import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import torch
import subprocess
import numpy as np
import models
import systems
import torch.nn.functional as F
from diffusers import DDPMScheduler
from systems.base import BaseSystem
from utils.lr_schedulers import LinearWarmupCosineAnnealingLR


@systems.register("sys_singapo")
class SingapoSystem(BaseSystem):
    """Trainer for the B9 model, incorporating the classifier-free for image condition."""

    def __init__(self, hparams):
        super().__init__(hparams)
        self.model = models.make(hparams.model.name, hparams.model)
        # configure the scheduler of DDPM
        self.scheduler = DDPMScheduler(**self.hparams.scheduler.config)
        # load the dummy DINO features
        self.dummy_dino = np.load('systems/dino_dummy.npy').astype(np.float32)
        # use the manual optimization
        self.automatic_optimization = False
        # save the hyperparameters
        self.save_hyperparameters()

    def load_cage_weights(self, pretrained_ckpt=None):
        ckpt = torch.load(pretrained_ckpt)
        state_dict = ckpt["state_dict"]
        # remove the "model." prefix from the keys
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        # load the weights
        self.model.load_state_dict(state_dict, strict=False) 
        # separate the weights of CAGE and our new modules
        self.cage_params = self.adapter_params = []
        for name, param in self.model.named_parameters():
            if "img" in name or "norm5" in name:
                self.adapter_params.append(param)
            else:
                self.cage_params.append(param)
        print("[INFO] loaded model weights of the pretrained CAGE.")
        

    def fg_loss(self, all_attn_maps, loss_masks):
        """
        Excite the attention maps within the object regions, while weaken the attention outside the object regions.
        
        Args:
            all_attn_maps: cross-attention maps from all layers, shape (B*L, H, 160, 256)
            loss_masks: object seg mask on the image patches, shape (B, 160, 256)

        Returns:
            loss: loss on the attention maps
        """
        valid_mask = loss_masks['valid_nodes']
        fg_mask = loss_masks['fg']
        # get the number of layers and batch size
        L = self.hparams.model.n_layers
        H = all_attn_maps.shape[1]
        # Reshape all the masks to the shape of the attention maps
        valid_node = valid_mask[:, :, 0].unsqueeze(1).expand(-1, H, -1).unsqueeze(-1).expand(-1, -1, -1, 256).repeat(L, 1, 1, 1)
        obj_region = fg_mask.unsqueeze(1).expand(-1, H, -1, -1).repeat(L, 1, 1, 1)
        # construct masks for the object and non-object regions
        fg_region = torch.logical_and(valid_node, obj_region)
        bg_region = torch.logical_and(valid_node, ~obj_region)
        # loss to excite the foreground regions
        loss = 1. - all_attn_maps[fg_region].mean() + all_attn_maps[bg_region].mean()
        return loss
    
    def diffuse_process(self, inputs):
        x = inputs["x"]
        # Sample Gaussian noise
        noise = torch.randn(x.shape, device=self.device, dtype=x.dtype)
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (x.shape[0],),
            device=self.device,
            dtype=torch.long,
        )
        # Add Gaussian noise to the input
        noisy_x = self.scheduler.add_noise(x, noise, timesteps)
        # update the inputs
        inputs["noise"] = noise
        inputs["timesteps"] = timesteps
        inputs["noisy_x"] = noisy_x

    def prepare_inputs(self, batch, mode='train', n_samples=1):
        x, c, f = batch

        cat = c["cat"]                   # object category
        attr_mask = c["attr_mask"]       # attention mask for local self-attention (follow the CAGE)
        key_pad_mask = c["key_pad_mask"] # key padding mask for global self-attention (follow the CAGE)
        graph_mask = c["adj_mask"]       # attention mask for graph relation self-attention (follow the CAGE)

        inputs = {}
        if mode == 'train':
            # the number of sampled timesteps per iteration
            n_repeat = self.hparams.n_time_samples
            # for sampling multiple timesteps
            x = x.repeat(n_repeat, 1, 1)
            cat = cat.repeat(n_repeat)
            f = f.repeat(n_repeat, 1, 1)
            key_pad_mask = key_pad_mask.repeat(n_repeat, 1, 1)
            graph_mask = graph_mask.repeat(n_repeat, 1, 1)
            attr_mask = attr_mask.repeat(n_repeat, 1, 1)
        elif mode == 'val':
            noisy_x = torch.randn(x.shape, device=x.device)
            dummy_f = torch.tensor(self.dummy_dino, device=self.device).unsqueeze(0).expand_as(f)
            inputs["noisy_x"] = noisy_x
            inputs["dummy_f"] = dummy_f
        elif mode == 'test':
            # for sampling multiple outputs
            x = x.repeat(n_samples, 1, 1)
            cat = cat.repeat(n_samples)
            f = f.repeat(n_samples, 1, 1)
            key_pad_mask = key_pad_mask.repeat(n_samples, 1, 1)
            graph_mask = graph_mask.repeat(n_samples, 1, 1)
            attr_mask = attr_mask.repeat(n_samples, 1, 1)
            noisy_x = torch.randn(x.shape, device=x.device)
            dummy_f = torch.tensor(self.dummy_dino, device=self.device).unsqueeze(0).expand_as(f)
            inputs["noisy_x"] = noisy_x
            inputs["dummy_f"] = dummy_f
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        inputs["x"] = x
        inputs["f"] = f
        inputs["cat"] = cat
        inputs["key_pad_mask"] = key_pad_mask
        inputs["graph_mask"] = graph_mask
        inputs["attr_mask"] = attr_mask
        
        return inputs
    
    def prepare_loss_mask(self, batch):
        x, c, _ = batch
        n_repeat = self.hparams.n_time_samples # the number of sampled timesteps per iteration

        # mask on the image patches for the foreground regions
        mask_fg = c["img_obj_mask"] 
        mask_fg = mask_fg.repeat(n_repeat, 1, 1)
        
        # mask on the valid nodes
        index_tensor = torch.arange(x.shape[1], device=self.device, dtype=torch.int32).unsqueeze(0)  # (1, N)
        valid_nodes = index_tensor < (c['n_nodes'] * 5).unsqueeze(-1)  
        mask_valid_nodes = valid_nodes.unsqueeze(-1).expand_as(x)
        mask_valid_nodes = mask_valid_nodes.repeat(n_repeat, 1, 1)

        return {"fg": mask_fg, "valid_nodes": mask_valid_nodes}
    
    def manage_cfg(self, inputs):
        '''
        Manage the classifier-free training for the image and graph condition.
        The CFG for object category is managed by the model (i.e. the CombinedTimestepLabelEmbeddings module in norm1 for each attention block)
        '''
        img_drop_prob = self.hparams.get("img_drop_prob", 0.0)
        graph_drop_prob = self.hparams.get("graph_drop_prob", 0.0)
        drop_img, drop_graph = False, False

        if img_drop_prob > 0.0:
            drop_img = torch.rand(1) < img_drop_prob
            if drop_img.item():
                dummy_batch = torch.tensor(self.dummy_dino, device=self.device).unsqueeze(0).expand_as(inputs['f'])
                inputs['f'] = dummy_batch  # use the dummy DINO features

        if graph_drop_prob > 0.0:
            if not drop_img:
                drop_graph = torch.rand(1) < graph_drop_prob
                if drop_graph.item():
                    inputs['graph_mask'] = None # for varify the model only, replace with the below line later and retrain the model
                    # inputs['graph_mask'] = inputs['key_pad_mask'] # use the key padding mask

    def compute_loss(self, batch, inputs, outputs):
        loss_dict = {}
        loss_weight = self.hparams.get("loss_fg_weight", 1.0)

        # prepare the loss masks
        loss_masks = self.prepare_loss_mask(batch)

        # diffusion model loss: MSE on the residual noise
        loss_mse = F.mse_loss(outputs['noise_pred'] * loss_masks['valid_nodes'], inputs['noise'] * loss_masks['valid_nodes'])
        # attention mask loss: BCE loss on the attention maps
        loss_fg = loss_weight * self.fg_loss(outputs['attn_maps'], loss_masks)
        
        # total loss
        loss = loss_mse + loss_fg 
        
        # log the losses
        loss_dict["train/loss_mse"] = loss_mse
        loss_dict["train/loss_fg"] = loss_fg
        loss_dict["train/loss_total"] = loss

        return loss, loss_dict
 
    def training_step(self, batch, batch_idx):
        # prepare the inputs and GT
        inputs = self.prepare_inputs(batch, mode='train')
        
        # manage the classifier-free training
        self.manage_cfg(inputs)

        # forward: diffusion process
        self.diffuse_process(inputs)

        # reverse: denoising process
        outputs = self.model(
            x=inputs['noisy_x'],
            cat=inputs['cat'],
            timesteps=inputs['timesteps'],
            feat=inputs['f'],
            key_pad_mask=inputs['key_pad_mask'],
            graph_mask=inputs['graph_mask'],
            attr_mask=inputs['attr_mask'],
        )

        # compute the loss
        loss, loss_dict = self.compute_loss(batch, inputs, outputs)

        # manual backward
        opt1, opt2 = self.optimizers()
        opt1.zero_grad()
        opt2.zero_grad()
        self.manual_backward(loss)
        opt1.step()
        opt2.step()

        # logging
        self.log_dict(loss_dict, sync_dist=True, on_step=True, on_epoch=False)

    def on_train_epoch_end(self):
        # step the lr scheduler every epoch
        sch1, sch2 = self.lr_schedulers()
        sch1.step()
        sch2.step()

    def inference(self, inputs, is_label_free=False):
        device = inputs['x'].device
        omega = self.hparams.get("guidance_scaler", 0)
        noisy_x = inputs['noisy_x']

        # set scheduler to denoise every 100 steps
        self.scheduler.set_timesteps(100)
        # denoising process
        for t in self.scheduler.timesteps:
            timesteps = torch.tensor([t], device=device)
            outputs_cond = self.model(
                x=noisy_x,
                cat=inputs['cat'],
                timesteps=timesteps,
                feat=inputs['f'], 
                key_pad_mask=inputs['key_pad_mask'],
                graph_mask=inputs['graph_mask'],
                attr_mask=inputs['attr_mask'],
                label_free=is_label_free,
            ) # take condtional image as input
            if omega != 0:
                outputs_free = self.model(
                    x=noisy_x,
                    cat=inputs['cat'],
                    timesteps=timesteps,
                    feat=inputs['dummy_f'], 
                    key_pad_mask=inputs['key_pad_mask'],
                    graph_mask=inputs['graph_mask'],
                    attr_mask=inputs['attr_mask'],
                    label_free=is_label_free,
                ) # take the dummy DINO features for the condition-free mode
                noise_pred = (1 + omega) * outputs_cond['noise_pred'] - omega * outputs_free['noise_pred']
            else:
                noise_pred = outputs_cond['noise_pred']
            noisy_x = self.scheduler.step(noise_pred, t, noisy_x).prev_sample

        return noisy_x

    def validation_step(self, batch, batch_idx):
        # prepare the inputs and GT
        inputs = self.prepare_inputs(batch, mode='val')
        # denoising process for inference
        out = self.inference(inputs)
        # compute the metrics
        log_dict = self.val_compute_metrics(out, inputs['x'], batch[1])
        self.log_dict(log_dict, on_step=True)

        # visualize the first 10 results
        self.save_val_img(out[:10], inputs['x'][:10], batch[1])

    def test_step(self, batch, batch_idx):
        is_label_free = self.hparams.get("test_label_free", False)
        # prepare the inputs and GT
        inputs = self.prepare_inputs(batch, mode='test', n_samples=5)
        # denoising process for inference
        out = self.inference(inputs, is_label_free)
        # save the results
        self.save_test_step(out, inputs['x'], batch[1], batch_idx)

    def on_test_end(self):
        # only run the single GPU
        if self.global_rank == 0:
            exp_name = self._get_exp_name()
            # retrieve parts
            subprocess.run(['python', 'scripts/mesh_retrieval/run_retrieve.py', '--src', self.get_save_path(exp_name), '--json_name', 'object.json', '--gt_data_root', self.hparams.data_root])
            # save metrics
            if not self.hparams.get("test_no_GT", False):
                subprocess.run(['python', 'scripts/eval_metrics.py', '--exp_dir', self.get_save_path(exp_name), '--gt_root', self.hparams.data_root])
            # save html
            self._save_html_end()

    def configure_optimizers(self):
        optimizer_adapter = torch.optim.AdamW(
            self.adapter_params, **self.hparams.optimizer_adapter.args
        )
        lr_scheduler_adapter = LinearWarmupCosineAnnealingLR(
            optimizer_adapter,
            warmup_epochs=self.hparams.lr_scheduler_adapter.warmup_epochs,
            max_epochs=self.hparams.lr_scheduler_adapter.max_epochs,
            warmup_start_lr=self.hparams.lr_scheduler_adapter.warmup_start_lr,
            eta_min=self.hparams.lr_scheduler_adapter.eta_min,
        )

        optimizer_cage = torch.optim.AdamW(
            self.cage_params, **self.hparams.optimizer_cage.args
        )
        lr_scheduler_cage = LinearWarmupCosineAnnealingLR(
            optimizer_cage,
            warmup_epochs=self.hparams.lr_scheduler_cage.warmup_epochs,
            max_epochs=self.hparams.lr_scheduler_cage.max_epochs,
            warmup_start_lr=self.hparams.lr_scheduler_cage.warmup_start_lr,
            eta_min=self.hparams.lr_scheduler_cage.eta_min,
        )
        return (
            {"optimizer": optimizer_adapter, "lr_scheduler": lr_scheduler_adapter},
            {"optimizer": optimizer_cage, "lr_scheduler": lr_scheduler_cage},
        )

    