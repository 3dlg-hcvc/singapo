import torch
from torch import nn
from typing import Optional
from diffusers.models.embeddings import Timesteps, TimestepEmbedding, LabelEmbedding

class FinalLayer(nn.Module):
    """
    Final layer of the diffusion model that outputs the final logits.
    """
    def __init__(self, in_ch, out_ch=None, dropout=0.0):
        super().__init__()
        out_ch = in_ch if out_ch is None else out_ch
        self.linear = nn.Linear(in_ch, out_ch)
        self.norm = AdaLayerNormTC(in_ch, 2 * in_ch, dropout)

    def forward(self, x, t, cond=None):
        assert cond is not None
        x = self.norm(x, t, cond)
        x = self.linear(x)
        return x


class AdaLayerNormTC(nn.Module):
    """
    Norm layer modified to incorporate timestep and condition embeddings.
    """

    def __init__(self, embedding_dim, num_embeddings, dropout):
        super().__init__()
        self.emb = CombinedTimestepLabelEmbeddings(
            num_embeddings, embedding_dim, dropout
        )
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(
            embedding_dim, elementwise_affine=False, eps=torch.finfo(torch.float16).eps
        )

    def forward(self, x, timestep, cond):
        emb = self.linear(self.silu(self.emb(timestep, cond, hidden_dtype=None)))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x


class PEmbeder(nn.Module):
    """
    Positional embedding layer.
    """
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.embed.weight, mode="fan_in")

    def forward(self, x, idx=None):
        if idx is None:
            idx = torch.arange(x.shape[1], device=x.device, dtype=torch.long)
        return x + self.embed(idx)

class CombinedTimestepLabelEmbeddings(nn.Module):
    '''Modified from diffusers.models.embeddings.CombinedTimestepLabelEmbeddings'''
    def __init__(self, num_classes, embedding_dim, class_dropout_prob=0.1):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.class_embedder = LabelEmbedding(num_classes, embedding_dim, class_dropout_prob)

    def forward(self, timestep, class_labels, hidden_dtype=None, label_free=False):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, D)
        force_drop_ids = None # training mode
        if label_free: # inference mode, force_drop_ids is set to all ones to be dropped in class_embedder
            force_drop_ids = torch.ones_like(class_labels, dtype=torch.bool, device=class_labels.device)
        class_labels = self.class_embedder(class_labels, force_drop_ids)  # (N, D)
        conditioning = timesteps_emb + class_labels  # (N, D)
        return conditioning


class MyAdaLayerNormZero(nn.Module):
    """
    Adaptive layer norm zero (adaLN-Zero), borrowed from diffusers.models.attention.AdaLayerNormZero.
    Extended to incorporate scale parameters (gate_2, gate_3) for intermidate attention layers.
    """

    def __init__(self, embedding_dim, num_embeddings, class_dropout_prob):
        super().__init__()

        self.emb = CombinedTimestepLabelEmbeddings(
            num_embeddings, embedding_dim, class_dropout_prob
        )
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 8 * embedding_dim, bias=True)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, timestep, class_labels, hidden_dtype=None, label_free=False):
        emb_t_cls = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype, label_free=label_free)
        emb = self.linear(self.silu(emb_t_cls))
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            gate_2,
            gate_3,
        ) = emb.chunk(8, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp, gate_2, gate_3


class VisAttnProcessor:
    r"""
    This code is adapted from diffusers.models.attention_processor.AttnProcessor.
    Used for visualizing the attention maps when testing, NOT for training.
    """

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # Removed
        # if len(args) > 0 or kwargs.get("scale", None) is not None:
        #     deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
        #     deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query) # (40, 160, 16)
        key = attn.head_to_batch_dim(key) # (40, 256, 16)
        value = attn.head_to_batch_dim(value)  # (40, 256, 16)
        
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                attn_mask = torch.zeros_like(attention_mask, dtype=query.dtype, device=query.device)
                attn_mask = attn_mask.masked_fill_(attention_mask.logical_not(), float("-inf"))
            else:
                attn_mask = attention_mask
                assert attn_mask.dtype == query.dtype, f"query and attention_mask must have the same dtype, but got {query.dtype} and {attention_mask.dtype}."
        else:
            attn_mask = None
        attention_probs = attn.get_attention_scores(query, key, attn_mask) # (40, 160, 256)
        hidden_states = torch.bmm(attention_probs, value) # (40, 160, 16)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        attention_probs = attention_probs.reshape(batch_size, attn.heads, query.shape[1], sequence_length)

        return hidden_states, attention_probs

