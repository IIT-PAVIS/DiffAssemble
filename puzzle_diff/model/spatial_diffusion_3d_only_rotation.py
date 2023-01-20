import colorsys
import enum
import logging
import math
import torch.optim as optim


# from .backbones.Transformer_GNN import Transformer_GNN
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any

import einops
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL
import pytorch_lightning as pl
import scipy
import timm

from .distributions import IsotropicGaussianSO3
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix


# from .network_modules import (
#     default,
#     partial,
#     SinusoidalPositionEmbeddings,
#     PreNorm,
#     Downsample,
#     Upsample,
#     Residual,
#     LinearAttention,
#     ConvNextBlock,
#     ResnetBlock,
#     Attention,
#     exists,
# )
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn.models
import torchmetrics
import torchvision
import torchvision.transforms.functional as trF
from kornia.geometry.transform import Rotate as krot
from PIL import Image
from torch import Tensor
from torch.optim import Adam
from tqdm import tqdm
from transformers.optimization import Adafactor
from torchmetrics import Metric

import wandb

from .backbones import Eff_GAT_3d
from .utils_3d import (
    rot_cosine_loss,
    rot_l2_loss,
    rot_metrics,
    rot_points_cd_loss,
    rot_points_l2_loss,
    shape_cd_loss,
    trans_l2_loss,
    trans_metrics,
    test_loss,
    so3_scale,
    skew_to_rmat,
    CosineAnnealingWarmupRestarts
)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelScheduler(enum.Enum):
    """
    Which type of output the model predicts.
    """

    LINEAR = enum.auto()  # the model predicts x_{t-1}
    COSINE = enum.auto()  # the model predicts x_0
    COSINE_DISCRETE = enum.auto()  # the model predicts epsilon


def interpolate_color1d(color1, color2, fraction):
    # color1 = [float(x) / 255 for x in color1]
    # color2 = [float(x) / 255 for x in color2]
    hsv1 = color1  #  colorsys.rgb_to_hsv(*color1)
    hsv2 = color2  #  colorsys.rgb_to_hsv(*color2)
    h = hsv1[0] + (hsv2[0] - hsv1[0]) * fraction
    s = hsv1[1] + (hsv2[1] - hsv1[1]) * fraction
    v = hsv1[2] + (hsv2[2] - hsv1[2]) * fraction
    return tuple(x for x in (h, s, v))


def interpolate_color(
    pos, col_1=(1, 0, 0), col_2=(1, 1, 0), col_3=(0, 0, 1), col_4=(0, 1, 0)
):
    f1 = float((pos[0] + 1) / 2)
    f2 = float((pos[1] + 1) / 2)
    c1 = interpolate_color1d(col_1, col_2, f1)
    c2 = interpolate_color1d(col_3, col_4, f1)
    return interpolate_color1d(c1, c2, f2)


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def cosine_discrete_beta_schedule(timesteps, s=0.08):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """

    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps)
    alphas_cumprod = lambda t: torch.cos(((t / timesteps) + s) / (1 + s) + np.pi / 2)
    betas = 1 - alphas_cumprod(t + 1) / alphas_cumprod(t)
    return torch.clip(betas, 0.0001, 0.9999)


def cosine_beta_schedule(timesteps, s=0.08):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def extract(a, t, x_shape=None):
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out[:, None]  # out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def extract_rot(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

@torch.jit.script
def greedy_cost_assignment(pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
    # Compute pairwise distances between positions
    dist = torch.norm(pos1[:, None] - pos2, dim=2)

    # Create a tensor to store the assignments
    assignments = torch.zeros(dist.size(0), 3, dtype=torch.int64)

    # Create a mask to keep track of assigned positions
    mask = torch.ones_like(dist, dtype=torch.bool)

    # Counter for keeping track of the number of assignments
    counter = 0

    # While there are still unassigned positions
    while mask.sum() > 0:
        # Find the minimum distance
        min_val, min_idx = dist[mask].min(dim=0)

        # Get the indices of the two dimensions
        idx = int(min_idx.item())
        ret = mask.nonzero()[idx, :]
        i = ret[0]
        j = ret[1]

        # Add the assignment to the tensor
        assignments[counter, 0] = i
        assignments[counter, 1] = j
        assignments[counter, 2] = min_val

        # Increase the counter
        counter += 1

        # Remove the assigned positions from the distance matrix and the mask
        mask[i, :] = 0
        mask[:, j] = 0

    return assignments[:counter]


class GNN_Diffusion(pl.LightningModule):
    def __init__(
        self,
        steps=600,
        inference_ratio=1,
        sampling="DDPM",
        learning_rate=1e-4,
        save_and_sample_every=1000,
        classifier_free_prob=0,
        classifier_free_w=0,
        noise_weight=0.0,
        model_mean_type: ModelMeanType = ModelMeanType.EPSILON,
        input_channels=4,
        output_channels=7,
        scheduler: ModelScheduler = ModelScheduler.LINEAR,
        visual_pretrained: bool = True,
        freeze_backbone: bool = True,
        n_layers: int = 4,
        loss_type="all",
        backbone='vnn',
        max_epochs=200,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.visual_pretrained = visual_pretrained
        self.free_backbone = freeze_backbone
        self.model_mean_type = model_mean_type
        self.learning_rate = learning_rate
        self.save_and_sample_every = save_and_sample_every
        self.classifier_free_prob = classifier_free_prob
        self.classifier_free_w = classifier_free_w
        self.noise_weight = noise_weight
        self.backbone=backbone
        self.max_epochs=max_epochs
        ### DIFFUSION STUFF


        if sampling == "DDIM":
            self.inference_ratio = inference_ratio
            self.p_sample = partial(
                self.p_sample,
                sampling_func=self.p_sample_ddim,
            )
            self.eta = 0
        # define beta schedule
        betas = {
            ModelScheduler.LINEAR: linear_beta_schedule,
            ModelScheduler.COSINE: cosine_beta_schedule,
            ModelScheduler.COSINE_DISCRETE: cosine_discrete_beta_schedule,
        }[scheduler](timesteps=steps)

        # self.timesteps = torch.arange(0, 700).flip(0)
        self.register_buffer("betas", betas)
        # self.betas = cosine_beta_schedule(timesteps=steps)
        # define alphas
        alphas = 1.0 - self.betas
        self.register_buffer("alphas", alphas)
        alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.register_buffer("sqrt_recip_alphas", sqrt_recip_alphas)
        self.register_buffer("identity", torch.eye(3))
        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)

        self.register_buffer(
            "sqrt_recip_alphas_cumprod", np.sqrt(1.0 / self.alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", np.sqrt(1.0 / self.alphas_cumprod - 1)
        )

        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

        self.steps = steps
        self.input_channels = input_channels

        self.n_layers = n_layers
        self.init_backbone()

        self.save_hyperparameters()

    def init_backbone(self):

        self.model = Eff_GAT_3d(
            steps=self.steps,
            input_channels=self.input_channels,
            freeze_backbone=self.free_backbone,
            n_layers=self.n_layers,
            backbone=self.backbone
        )
    
    def initialize_torchmetrics(self):
        metrics = {}

        metrics["rmse_t"] = torchmetrics.MeanMetric()
        metrics["rmse_r"] = torchmetrics.MeanMetric()
        metrics["overall_acc"] = torchmetrics.MeanMetric()
        metrics["overall_nImages"] = torchmetrics.SumMetric()
        metrics["test_loss"] = torchmetrics.MeanMetric()
        self.metrics = nn.ModuleDict(metrics)

    def forward(self, xy_pos, time, patch_rgb, edge_index, batch) -> Any:
        return self.model(xy_pos, time, patch_rgb, edge_index, batch)

    def forward_with_feats(
        self,
        xy_pos: Tensor,
        time: Tensor,
        edge_index: Tensor,
        pcd_feats: Tensor,
        batch,
        return_attentions=False,
    ) -> Any:
        out, attentions = self.model.forward_with_feats(
            xy_pos, time, edge_index, pcd_feats, batch
        )

        return out, attentions

    def pcd_features(self, pcd):
        return self.model.pcd_features(pcd)

    # forward diffusion
    def q_sample_tr(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    # forward diffusion
    def q_sample_quat(self, x_start, t, noise=None):
        if noise is None:
            eps = extract(self.sqrt_one_minus_alphas_cumprod, t, t.shape)
            noise = IsotropicGaussianSO3(eps).sample()

        scale = extract_rot(self.sqrt_alphas_cumprod, t, t.shape)
        x_blend = so3_scale(x_start, scale)
        return x_blend @ noise

    def p_losses(
        self,
        x_start,
        t,
        noise=None,
        loss_type="l1",
        cond=None,
        edge_index=None,
        batch=None,
        n_batch=None,
        valids=None
    ):  
        x_start_tr = x_start[:, 4:]
        x_start_rot = quaternion_to_matrix(x_start[:, :4]) # input ad a rotation amtrix
        if noise is None:
            noise_tr = torch.randn_like(x_start_tr)
            eps = extract(self.sqrt_one_minus_alphas_cumprod, t, t.shape)
            noisedist = IsotropicGaussianSO3(eps.flatten()) # [6x1] ---> [6]
            noise_quat = noisedist.sample()

        #x_noisy_tr = self.q_sample_tr(x_start=x_start_tr, t=t, noise=noise_tr)
        #x_noisy_quat_1 = self.q_sample_quat(x_start=x_start_rot, t=t, noise=noise_quat)
        x_noisy = matrix_to_quaternion(self.q_sample_quat(x_start=x_start_rot, t=t, noise=noise_quat))
        # add projection of the point cloud?

        # Transformer case
        if self.steps == 1:  
            x_noisy = torch.zeros_like(x_noisy)

        pcd_feats = self.pcd_features(cond)

        prediction, _ = self.forward_with_feats(
            x_noisy,
            t,
            edge_index,
            pcd_feats=pcd_feats,
            batch=batch,
            return_attentions=False,
        )

        target = {
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
        }[self.model_mean_type]

        # Compute loss
        pred_r = prediction#[:, :4]
        # Rotation transformation
        #pred_r = matrix_to_quaternion(skew_to_rmat(prediction[:, :3])) #we predict a vector [x, y, z] --> convert to rmat ---> convert to quaternion


        gt_r = target#[:, :4]

        # parameter for loss sum

        trans_loss_w = 1.
        rot_pt_cd_loss_w = 10.
        transform_pt_cd_loss_w = 10.
        # cosine regression loss on rotation
        rot_loss_w = 0.2
        # per-point l2 loss between rotated part point clouds
        rot_pt_l2_loss_w = 1.

        if loss_type == "all":

            #trans_loss = trans_l2_loss(pred_t, gt_t, n_batch=n_batch, valids=valids).mean()
            rot_pt_cd_loss = rot_points_cd_loss(cond, pred_r, gt_r, n_batch=n_batch, valids=valids).mean()
            #transform_pt_cd_loss = shape_cd_loss(cond, pred_t, gt_t, pred_r, gt_r, n_batch=n_batch, valids=valids).mean()
            rot_loss = rot_cosine_loss(pred_r, gt_r,n_batch=n_batch, valids=valids).mean()
            rot_pt_l2_loss = rot_points_l2_loss(cond, pred_r, gt_r,n_batch=n_batch, valids=valids).mean()

        
            #loss = F.smooth_l1_loss(target, prediction)
            loss = rot_pt_cd_loss * rot_pt_cd_loss_w  + rot_loss * rot_loss_w +     rot_pt_l2_loss * rot_pt_l2_loss_w
            
            loss_dict = {
            'rot_pt_cd_loss': rot_pt_cd_loss * rot_pt_cd_loss_w,
            'rot_loss': rot_loss * rot_loss_w,
            'rot_pt_l2_loss':rot_pt_l2_loss * rot_pt_l2_loss_w,
            #'loss':loss
                }  # all loss are of shape [B]


        elif loss_type == "split":
            rot_loss = rot_l2_loss(pred_r, gt_r).mean()
            t_loss = trans_l2_loss(pred_t, gt_t).mean()

            loss = rot_loss + t_loss
        else:
            raise NotImplementedError()

        return loss_dict
        #return loss

    def _get_variance(self, timestep, prev_timestep):
        alpha_prod_t = extract(
            self.alphas_cumprod, timestep
        )  # self.alphas_cumprod[timestep]

        alpha_prod_t_prev = (
            extract(self.alphas_cumprod, prev_timestep)
            if (prev_timestep >= 0).all()
            else alpha_prod_t * 0 + 1
        )

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (
            1 - alpha_prod_t / alpha_prod_t_prev
        )

        return variance

    @torch.no_grad()
    def p_sample_ddim(
        self, x, t, t_index, edge_index, pcd_feats, batch
    ):  # (self, x, t, t_index, cond):
        # if t[0] == 0:
        #     return x
        prev_timestep = t - self.inference_ratio

        eta = self.eta
        # Si tratta della catena di gaussiane che si moltiplicano l'una con l'altra
        alpha_prod = extract(self.alphas_cumprod, t, x.shape)

        if (prev_timestep >= 0).all():
            alpha_prod_prev = extract(self.alphas_cumprod, prev_timestep, x.shape)
        else:
            alpha_prod_prev = alpha_prod * 0 + 1

        beta = 1 - alpha_prod
        beta_prev = 1 - alpha_prod_prev
        model_output, attentions = self.forward_with_feats(
            x,
            t,
            edge_index,
            pcd_feats=pcd_feats,
            batch=batch,
            return_attentions=True,
        )

        # estimate x_0

        x_0 = {
            ModelMeanType.EPSILON: (x - beta**0.5 * model_output) / alpha_prod**0.5,
            ModelMeanType.START_X: model_output,
        }[self.model_mean_type]

        #eps = self._predict_eps_from_xstart(x, t, x_0)
        # 
        #x_0_tr = x_0[:, 4:]
        x_0_r = x_0#[:, :4] 
        #x_tr = x[:, 4:]
        x_quater = x#[:, :4]
        #eps_tr = self._predict_eps_from_xstart(x_tr, t, x_0_tr)
        eps_rot = matrix_to_quaternion(self._predict_eps_from_xstart_rot(x_quater, t, x_0_r))

        # estimate "direction to x_t"
        # Why eps not N(0, 1)?
        #pred_sample_direction = (1 - alpha_prod_prev) ** (0.5) * eps

        #pred_sample_direction_tr = (1 - alpha_prod_prev) ** (0.5) * eps_tr
        pred_sample_direction_rot = so3_scale(quaternion_to_matrix(eps_rot), ((1 - alpha_prod_prev) ** (0.5)).view(-1)) #(1 - alpha_prod_prev) ** (0.5) * eps_rot # this operation required the skew?

        # x_t-1 = a * x_0 + b * eps
        #prev_sample = alpha_prod_prev ** (0.5) * x_0 + pred_sample_direction

        # in questo caso forse e' necessario nuovamente tornare in so(3)
        #prev_sample_tr = alpha_prod_prev ** (0.5) * x_0_tr + pred_sample_direction_tr
        prev_sample = matrix_to_quaternion(so3_scale(quaternion_to_matrix(x_0_r), (alpha_prod_prev ** (0.5)).view(-1)) @ pred_sample_direction_rot)
        #prev_sample_r_1 = prev_sample_r @ pred_sample_direction_rot
        #prev_sample = torch.concat([prev_sample_r, prev_sample_tr], axis = 1) # combinazione 
        return prev_sample, attentions

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _predict_eps_from_xstart_rot(self, x_t, t, pred_xstart):
        # devo trasformare x_t in rotazione
        x_t_term = so3_scale(quaternion_to_matrix(x_t), (extract_rot(self.sqrt_recip_alphas_cumprod, t, t.shape)/extract_rot(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape).flatten()))
        # devo trasformare noise in rotazione
        pred_xstart = so3_scale(quaternion_to_matrix(pred_xstart), 1/ extract_rot(self.sqrt_recipm1_alphas_cumprod, t, t.shape)) #[..., None]
        # Rotation = multiply by inverse op (matrices, so transpose)
        return x_t_term @ pred_xstart.transpose(-1, -2)

    # Algorithm 2 but save all images:
    @torch.no_grad()
    def p_sample_loop(self, shape, cond, edge_index, batch):
        # device = next(model.parameters()).device
        device = self.device

        
        b = shape[0]
        shape = torch.Size([b, 3])
        # start from pure noise (for each example in the batch)
        #img = torch.randn(shape, device=device) * self.noise_weight 
        imgs = []
        attentions = []
        pcd_feats = self.pcd_features(cond)
        # dimension problem..
        #x = IsotropicGaussianSO3(eps=torch.ones([], device=device)).sample(shape) # * self.noise_weight (qui definsico da dove partono tutte le rotazioni)
        # aggiungere qui zero starting
        #x, _ = torch.qr(torch.randn((b, 3, 3), device=device)) # anche questo restituisce una matrice R causale
        # This case is equal to noise_weight = 0
        truepos = torch.eye(3).to(device)
        x = truepos.repeat(b, 1, 1)
        
        img = matrix_to_quaternion(x)
        #img = torch.concat([x, img], axis = 1) # combinazione 
        # scomporre img in base alle due generazioni + combinarle dopo (coprire le dimensioni di b)
        for i in tqdm(
            list(reversed(range(0, self.steps, self.inference_ratio))),
            desc="sampling loop time step",
        ):
            img, atts = self.p_sample(
                img,
                torch.full((b,), i, device=device, dtype=torch.long),
                # time_t + i,
                i,
                edge_index=edge_index,
                pcd_feats=pcd_feats,
                batch=batch,
            )
            attentions.append(atts)
            imgs.append(img)
        return imgs, attentions

    @torch.no_grad()
    def p_sample(self, x, t, t_index, edge_index, sampling_func, pcd_feats, batch):
        return sampling_func(x, t, t_index, edge_index, pcd_feats, batch) # type of diffusion

    @torch.no_grad()
    def sample(
        self,
        image_size,
        batch_size=16,
        channels=3,
        cond=None,
        edge_index=None,
        batch=None,
    ):
        return self.p_sample_loop(
            shape=(batch_size, channels, image_size, image_size),
            cond=cond,
            edge_index=edge_index,
            batch=batch,
        )



    def configure_optimizers(self):
        #optimizer = Adafactor(self.parameters())

        #optimizer = Adafactor(self.parameters(), lr=self.learning_rate)
        
        """Build optimizer and lr scheduler."""
        #lr = 1e-3
        wd = 0
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.)
        scheduler = 'cosine'
        if scheduler:
            assert scheduler in ['cosine']
            total_epochs = self.max_epochs
            warmup_epochs = int(total_epochs * 0.)
            scheduler = CosineAnnealingWarmupRestarts(
                optimizer,
                total_epochs,
                max_lr=self.learning_rate,
                min_lr=self.learning_rate / 100.,
                warmup_steps=warmup_epochs,
            )
            return (
                [optimizer],
                [{
                    'scheduler': scheduler,
                    'interval': 'epoch',
                }],
            )
        #optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        # return super().training_step(*args, **kwargs)
        batch_size = batch.batch.max().item() + 1
        t = torch.randint(0, self.steps, (batch_size,), device=self.device).long()
        new_t = torch.gather(t, 0, batch.batch)
        loss_dict = self.p_losses(
            batch.x,
            new_t,
            loss_type=self.loss_type,
            cond=batch.pcds,
            edge_index=batch.edge_index,
            batch=batch.batch,
            n_batch=len(batch.data_id),
            valids=batch.valids
        )
        loss = sum([i for i in loss_dict.values()])
        if batch_idx == 0 and self.local_rank == 0:
            sampled_pos, _ = self.p_sample_loop(
                batch.x.shape, batch.pcds, batch.edge_index, batch=batch.batch
            )
            final_pos = sampled_pos[-1]

            save_path = Path(f"results/{self.logger.experiment.name}/train")
            for i in range(
                min(batch.batch.max().item(), 4)
            ):  # save max 4 images during training loop
                idx = torch.where(batch.batch == i)[0]
                pcds = batch.pcds[idx]
                gt_pos = batch.x[idx]
                pred_pos = final_pos[idx]

                # todo Save 3d pcd

        self.log_dict(loss_dict)
        #self.log("loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            sampled_pos, _ = self.p_sample_loop(
                batch.x.shape, batch.pcds, batch.edge_index, batch=batch.batch
            )
            final_pos = sampled_pos[-1]
            for i in range(batch.batch.max() + 1):
                idx = torch.where(batch.batch == i)[0]
                pcds = batch.pcds[idx]
                gt_pos = batch.x[idx]
                pred_pos = final_pos[idx]

                # Compute metrics
                #pred_t = pred_pos[:, 4:]
                pred_r = pred_pos#[:, : 4]
                #pred_r = matrix_to_quaternion(skew_to_rmat(pred_pos[:, :3] ))# applico trasformazioni.

                #gt_t = gt_pos[:, 4:]
                gt_r = gt_pos#[:, :4]
                #t_metric = trans_metrics(pred_t, gt_t) 
                rot_metric = rot_metrics(pred_r, gt_r)
                #loss = test_loss(pcds, pred_t, gt_t, pred_r, gt_r, training=False)

                #self.metrics["rmse_t"].update(t_metric)
                self.metrics["rmse_r"].update(rot_metric)
                #self.metrics["test_loss"].update(loss)

                # Log PCD

            self.log_dict(self.metrics)
        # return accuracy_dict

    def validation_epoch_end(self, outputs) -> None:
        self.log_dict(self.metrics)

    def test_epoch_end(self, outputs) -> None:
        return self.validation_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_predict_epoch_start(self):
        logging.info(f"Saving to results/{self.logger.experiment.name}/preds")

    def save_3d_image(
        self,
        pcds,
        pos,
        gt_pos,
        patches_dim,
        ind_name,
        file_name: Path,
        correct=None,
    ):
        file_name.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(2, 2)

        pass

        im = PIL.Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )
        im = wandb.Image(im)
        self.logger.experiment.log(
            {f"{file_name.stem}": im, "global_step": self.global_step}
        )

        plt.savefig(f"{file_name}/asd_{self.current_epoch}-{ind_name}.png")
        plt.close()

    def to_batch_dim(self, part_pcs, part_valids):
        """Extract per-part point cloud features."""
        T, N, _ = part_pcs.shape  # [T, N, 3] dove T = N * P
        valid_mask = (part_valids == 1)
        # shared-weight encoder
        valid_pcs = part_pcs[valid_mask]  # [n, N, 3]
        valid_feats = self.encoder(valid_pcs)  # [n, C]
        pc_feats = torch.zeros(B, P, self.pc_feat_dim).type_as(valid_feats) # matrice degli zeri con le dimensioni degli output
        pc_feats[valid_mask] = valid_feats
        return pc_feats
