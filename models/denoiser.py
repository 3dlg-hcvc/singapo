import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import torch
import models
from torch import nn
from diffusers.models.attention import Attention, FeedForward
from models.utils import (
    PEmbeder,
    FinalLayer,
    VisAttnProcessor,
    MyAdaLayerNormZero
)


class Attn_Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: int = None,
        attention_bias: bool = False,
        norm_elementwise_affine: bool = True,
        final_dropout: bool = False,
        class_dropout_prob: float = 0.0,  # for classifier-free
        img_emb_dims=None,

    ):
        super().__init__()

        self.norm1 = MyAdaLayerNormZero(dim, num_embeds_ada_norm, class_dropout_prob)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.norm4 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.norm5 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)

        self.local_attn = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
        )

        self.global_attn = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
        )

        self.graph_attn = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
        )

        self.img_attn = Attention(
            query_dim=dim,
            cross_attention_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_norm="layer_norm",
            processor=VisAttnProcessor(), 
        )

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
        )

        # image embedding layers
        layers = []
        for i in range(len(img_emb_dims) - 1):
            layers.append(nn.Linear(img_emb_dims[i], img_emb_dims[i + 1]))
            layers.append(nn.LeakyReLU(inplace=True))
        layers.pop(-1)
        self.img_emb = nn.Sequential(*layers)
        self.init_img_emb_weights()

    def init_img_emb_weights(self):
        for m in self.img_emb.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        hidden_states,
        img_patches,
        pad_mask,
        attr_mask,
        graph_mask,
        timestep,
        class_labels,
        label_free=False,
    ):
        # image patches embedding
        img_emb = self.img_emb(img_patches)

        # adaptive normalization, taken timestep and class_labels as input condition
        norm_hidden_states, gate_1, shift_mlp, scale_mlp, gate_mlp, gate_2, gate_3 = (
            self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype,
                label_free=label_free
            )
        )

        # local attribute self-attention
        attr_out = self.local_attn(norm_hidden_states, attention_mask=attr_mask)
        attr_out = gate_1.unsqueeze(1) * attr_out
        hidden_states = hidden_states + attr_out

        # global attribute self-attention
        norm_hidden_states = self.norm2(hidden_states)
        global_out = self.global_attn(norm_hidden_states, attention_mask=pad_mask)
        global_out = gate_2.unsqueeze(1) * global_out
        hidden_states = hidden_states + global_out

        # graph relation self-attention
        norm_hidden_states = self.norm3(hidden_states)
        graph_out = self.graph_attn(norm_hidden_states, attention_mask=graph_mask)
        graph_out = gate_3.unsqueeze(1) * graph_out
        hidden_states = hidden_states + graph_out

        # cross attention with image patches
        norm_hidden_states = self.norm4(hidden_states)
        B, Na, D = norm_hidden_states.shape
        Np = img_emb.shape[1] # number of image patches
        reshaped = norm_hidden_states.reshape(B, Na // 5, 5, D)
        bboxes = reshaped[:, :, 0, :] # (B, K, D)
        # cross attention between bbox attributes and image patches
        bbox_img_out, bbox_cross_attn_map = self.img_attn(
            bboxes,
            encoder_hidden_states=img_emb,
            attention_mask=None,
        )  # cross_attn_map: (B, n_head, K, Np)

        # to reshape the cross_attn_map back to (B, n_head, Na*5, Np), reduntant for other attributes, fix later
        cross_attn_map_reshape = torch.zeros(size=(B, bbox_cross_attn_map.shape[1], Na // 5, 5, Np), device=bbox_cross_attn_map.device)
        cross_attn_map_reshape[:, :, :, 0, :] = bbox_cross_attn_map
        cross_attn_map = cross_attn_map_reshape.reshape(B, bbox_cross_attn_map.shape[1], Na, Np)

        # assemble the output of cross attention with bbox attributes and other attributes
        img_out = torch.empty(size=(B, Na // 5, 5, D), device=hidden_states.device, dtype=hidden_states.dtype)
        img_out[:, :, 0, :] = bbox_img_out
        img_out[:, :, 1:, :] = reshaped[:, :, 1:, :]
        img_out = img_out.reshape(B, Na, D)
        hidden_states = hidden_states + img_out

        # feed-forward
        norm_hidden_states = self.norm5(hidden_states)
        norm_hidden_states = (
            norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        )
        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = ff_output + hidden_states

        return hidden_states, cross_attn_map


@models.register("denoiser")
class Denoiser(nn.Module):
    """
    Denoiser based on CAGE's attribute attention block + our ICA module, with 4 sequential attentions: LA -> GA -> GRA -> ICA
    Different image adapters for each layer.
    The image cross attention is with key-padding masks (object mask, part mask)
    *** The ICA only applies to the bbox attributes, not other attributes such as motion params.***
    """

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.K = self.hparams.get("K", 32)

        in_ch = hparams.in_ch
        attn_dim = hparams.attn_dim
        mid_dim = attn_dim // 2
        n_head = hparams.n_head
        head_dim = attn_dim // n_head
        num_embeds_ada_norm = 6 * attn_dim
        
        # embedding layers for different node attributes
        self.aabb_emb = nn.Sequential(
            nn.Linear(in_ch, mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, attn_dim),
        )
        self.jaxis_emb = nn.Sequential(
            nn.Linear(in_ch, mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, attn_dim),
        )
        self.range_emb = nn.Sequential(
            nn.Linear(in_ch, mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, attn_dim),
        )
        self.label_emb = nn.Sequential(
            nn.Linear(in_ch, mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, attn_dim),
        )
        self.jtype_emb = nn.Sequential(
            nn.Linear(in_ch, mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, attn_dim),
        )
        # positional encoding for nodes and attributes
        self.pe_node = PEmbeder(self.K, attn_dim)
        self.pe_attr = PEmbeder(5, attn_dim)

        # attention layers
        self.attn_layers = nn.ModuleList(
            [
                Attn_Block(
                    dim=attn_dim,
                    num_attention_heads=n_head,
                    attention_head_dim=head_dim,
                    class_dropout_prob=hparams.get("cat_drop_prob", 0.0),
                    dropout=hparams.dropout,
                    activation_fn="geglu",
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=False,
                    norm_elementwise_affine=True,
                    final_dropout=False,
                    img_emb_dims=hparams.get("img_emb_dims", None),
                )
                for d in range(hparams.n_layers)
            ]
        )

        self.final_layer = FinalLayer(attn_dim, in_ch)

    def forward(
        self,
        x,
        cat,
        timesteps,
        feat,
        key_pad_mask=None,
        graph_mask=None,
        attr_mask=None,
        label_free=False,
    ):
        B = x.shape[0]
        x = x.view(B, self.K, 5 * 6)

        # embedding layers for different attributes
        x_aabb = self.aabb_emb(x[..., :6])
        x_jtype = self.jtype_emb(x[..., 6:12])
        x_jaxis = self.jaxis_emb(x[..., 12:18])
        x_range = self.range_emb(x[..., 18:24])
        x_label = self.label_emb(x[..., 24:30])

        # concatenate all attribute embeddings
        x_ = torch.cat(
            [x_aabb, x_jtype, x_jaxis, x_range, x_label], dim=2
        )  # (B, K, 5*attn_dim)
        x_ = x_.view(B, self.K * 5, self.hparams.attn_dim)

        # positional encoding for nodes and attributes
        idx_attr = torch.tensor(
            [0, 1, 2, 3, 4], device=x.device, dtype=torch.long
        ).repeat(self.K)
        idx_node = torch.arange(
            self.K, device=x.device, dtype=torch.long
        ).repeat_interleave(5)
        x_ = self.pe_attr(self.pe_node(x_, idx=idx_node), idx=idx_attr)


        # init tensor to store attention maps
        Np = feat.shape[1]
        attn_maps = torch.empty(
            size=(B * self.hparams.n_layers, self.hparams.n_head, self.K*5, Np),
            device=x.device,
        )

        # attention layers
        for i, attn_layer in enumerate(self.attn_layers):
            x_, attn_map = attn_layer(
                hidden_states=x_,
                img_patches=feat,
                timestep=timesteps,
                class_labels=cat,
                pad_mask=key_pad_mask,
                graph_mask=graph_mask,
                attr_mask=attr_mask,
                label_free=label_free,
            )
            # store attention maps
            attn_maps[i * B : i * B + B] = attn_map

        y = self.final_layer(x_, timesteps, cat)
        return {
            'noise_pred': y,
            'attn_maps': attn_maps,
        }
