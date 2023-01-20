from copy import deepcopy
import os
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
import trimesh

from .distributions import IsotropicGaussianSO3
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix, matrix_to_euler_angles


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
    Rotation3D,
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
    projection,
    translation,
    CosineAnnealingWarmupRestarts,
    geodesic_distance,
    transform_pc,
    calc_part_acc,
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
        input_channels=7,
        output_channels=7,
        scheduler: ModelScheduler = ModelScheduler.LINEAR,
        visual_pretrained: bool = True,
        freeze_backbone: bool = True,
        n_layers: int = 4,
        loss_type="all",
        backbone="vnn",
        max_epochs=200,
        use_vn_dgcnn_equiv_inv_mp: bool = False,
        max_num_part: int = 20,
        use_6dof: bool = False,
        architecture="transformer",
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
        self.backbone = backbone
        self.max_epochs = max_epochs
        self.use_vn_dgcnn_equiv_inv_mp = use_vn_dgcnn_equiv_inv_mp
        self.max_num_part = max_num_part
        self.use_6dof = use_6dof
        self.save_eval_images = False
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
        if self.use_6dof:
            self.input_channels = 13  # 4 + 3 + 6 , quat (unused) + trans + 6dof_rot

        self.architecture = architecture
        self.n_layers = n_layers
        self.init_backbone()

        self.save_hyperparameters()

    def init_backbone(self):
        self.model = Eff_GAT_3d(
            steps=self.steps,
            input_channels=self.input_channels,
            freeze_backbone=self.free_backbone,
            n_layers=self.n_layers,
            backbone=self.backbone,
            use_vn_dgcnn_equiv_inv_mp=self.use_vn_dgcnn_equiv_inv_mp,
            t_channels=9 if self.use_6dof else 3,
            architecture=self.architecture,
        )

    def initialize_torchmetrics(self, categories):
        metrics = {}

        metrics_AVG = {}

        for i in categories:
            metrics[f"rmse_t_{i}"] = torchmetrics.MeanMetric()
            metrics[f"rmse_r_{i}"] = torchmetrics.MeanMetric()
            metrics[f"gd_r_{i}"] = torchmetrics.MeanMetric()
            metrics[f"part_acc_{i}"] = torchmetrics.MeanMetric()

        self.metrics = nn.ModuleDict(metrics)

        metrics_AVG[f"rmse_t_AVG"] = torchmetrics.MeanMetric()
        metrics_AVG[f"rmse_r_AVG"] = torchmetrics.MeanMetric()
        metrics_AVG[f"gd_r_AVG"] = torchmetrics.MeanMetric()
        metrics_AVG[f"part_acc_AVG"] = torchmetrics.MeanMetric()
        self.avg_metrics = nn.ModuleDict(metrics_AVG)

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
        valids=None,
    ):
        x_start_tr = x_start[:, 4:]
        x_start_rot = quaternion_to_matrix(x_start[:, :4])  # input ad a rotation amtrix

        if self.use_6dof:
            x_start_tr = torch.cat(
                [x_start[:, 4:], x_start_rot[:, :, 0], x_start_rot[:, :, 1]], -1
            )
        if noise is None:
            noise_tr = torch.randn_like(x_start_tr)
            eps = extract(self.sqrt_one_minus_alphas_cumprod, t, t.shape)
            noisedist = IsotropicGaussianSO3(eps.flatten())  # [6x1] ---> [6]
            noise_quat = noisedist.sample()

        x_noisy_tr = self.q_sample_tr(x_start=x_start_tr, t=t, noise=noise_tr)
        x_noisy_quat_1 = self.q_sample_quat(x_start=x_start_rot, t=t, noise=noise_quat)
        # x_noisy_quat = matrix_to_quaternion(x_noisy_quat_1)
        # add projection of the point cloud?
        # cond = projection(x_noisy_quat_1, translation(x_noisy_tr,cond))
        # Back to quaternion
        x_noisy_quat = matrix_to_quaternion(x_noisy_quat_1)
        x_noisy = torch.concat([x_noisy_quat, x_noisy_tr], axis=1)
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
        pred_t = prediction[:, 4:7]
        pred_r = prediction[:, :4]
        # Rotation transformation
        # pred_r = matrix_to_quaternion(skew_to_rmat(prediction[:, :3])) #we predict a vector [x, y, z] --> convert to rmat ---> convert to quaternion

        gt_t = target[:, 4:7]
        gt_r = target[:, :4]

        # parameter for loss sum

        trans_loss_w = 1.0
        rot_pt_cd_loss_w = 0.0  # 10.0
        transform_pt_cd_loss_w = 10.0
        # cosine regression loss on rotation
        rot_loss_w = 0.2  # 10.0
        # per-point l2 loss between rotated part point clouds
        rot_pt_l2_loss_w = 0.0
        if self.use_6dof:
            rot_mat = quaternion_to_matrix(gt_r)
            rot_vec = torch.cat([rot_mat[:, :, 0], rot_mat[:, :, 1]], -1)
            # pred_6dof = pred_t[:, 3:]
            # pred_t = pred_t[:, :3]

            a1 = prediction[:, 7:10]
            a2 = prediction[:, 10:13]
            b1 = F.normalize(a1, dim=-1, p=2)
            # b1 = a1 / a1.norm(dim=-1)[:, None]
            b2 = a2 - (a2 * b1).sum(-1)[:, None] * b1

            b2 = F.normalize(b2, dim=-1, p=2)
            # b2 = b2 / b2.norm(dim=-1)[:, None]
            b3 = torch.cross(b1, b2, dim=1)
            rot = torch.stack([b1, b2, b3], dim=-1)
            pred_r = matrix_to_quaternion(rot)

        # m2 = rot_mat @ rot.transpose(1, 2)

        if loss_type == "all":
            trans_loss = trans_l2_loss(
                pred_t,
                gt_t,
                n_batch=n_batch,
                valids=valids,
                n_parts=self.max_num_part,  # 20
            ).mean()
            rot_pt_cd_loss = rot_points_cd_loss(
                cond,
                pred_r,
                gt_r,
                n_batch=n_batch,
                valids=valids,
                n_parts=self.max_num_part,  # 20
            ).mean()
            transform_pt_cd_loss = shape_cd_loss(
                cond,
                pred_t,
                gt_t,
                pred_r,
                gt_r,
                n_batch=n_batch,
                valids=valids,
                n_parts=self.max_num_part,  # 20,
            ).mean()
            rot_loss = rot_cosine_loss(
                pred_r,
                gt_r,
                n_batch=n_batch,
                valids=valids,
                n_parts=self.max_num_part,  # 20
            ).mean()
            rot_pt_l2_loss = rot_points_l2_loss(
                cond,
                pred_r,
                gt_r,
                n_batch=n_batch,
                valids=valids,
                n_parts=self.max_num_part,  # 20
            ).mean()
            rot_geodesic = geodesic_distance(
                pred_r,
                gt_r,
            ).mean()
            # loss = F.smooth_l1_loss(target, prediction)
            loss = (
                trans_loss * trans_loss_w
                + rot_pt_cd_loss * rot_pt_cd_loss_w
                + transform_pt_cd_loss * transform_pt_cd_loss_w
                + rot_loss * rot_loss_w
                + rot_pt_l2_loss * rot_pt_l2_loss_w
            )

            loss_dict = {
                "trans_loss": trans_loss * trans_loss_w,
                "rot_pt_cd_loss": rot_pt_cd_loss * rot_pt_cd_loss_w,
                "transform_pt_cd_loss": transform_pt_cd_loss * transform_pt_cd_loss_w,
                "rot_loss": rot_loss * rot_loss_w,
                # "rot_geodesic": rot_geodesic * rot_loss_w,
                "rot_pt_l2_loss": rot_pt_l2_loss * rot_pt_l2_loss_w,
                #'loss':loss
            }  # all loss are of shape [B]

        elif loss_type == "split":
            rot_loss = rot_l2_loss(pred_r, gt_r).mean()
            t_loss = trans_l2_loss(pred_t, gt_t).mean()

            loss = rot_loss + t_loss
        else:
            raise NotImplementedError()

        return loss_dict
        # return loss

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

        # eps = self._predict_eps_from_xstart(x, t, x_0)
        #
        x_0_tr = x_0[:, 4:]
        x_0_r = x_0[:, :4]
        x_tr = x[:, 4:]
        x_quater = x[:, :4]
        eps_tr = self._predict_eps_from_xstart(x_tr, t, x_0_tr)
        eps_rot = matrix_to_quaternion(
            self._predict_eps_from_xstart_rot(x_quater, t, x_0_r)
        )

        # estimate "direction to x_t"
        # Why eps not N(0, 1)?
        # pred_sample_direction = (1 - alpha_prod_prev) ** (0.5) * eps

        pred_sample_direction_tr = (1 - alpha_prod_prev) ** (0.5) * eps_tr
        pred_sample_direction_rot = so3_scale(
            quaternion_to_matrix(eps_rot), ((1 - alpha_prod_prev) ** (0.5)).view(-1)
        )  # (1 - alpha_prod_prev) ** (0.5) * eps_rot # this operation required the skew?

        # x_t-1 = a * x_0 + b * eps
        # prev_sample = alpha_prod_prev ** (0.5) * x_0 + pred_sample_direction

        # in questo caso forse e' necessario nuovamente tornare in so(3)
        prev_sample_tr = alpha_prod_prev ** (0.5) * x_0_tr + pred_sample_direction_tr
        prev_sample_r = matrix_to_quaternion(
            so3_scale(quaternion_to_matrix(x_0_r), (alpha_prod_prev ** (0.5)).view(-1))
            @ pred_sample_direction_rot
        )
        # prev_sample_r_1 = prev_sample_r @ pred_sample_direction_rot
        prev_sample = torch.concat(
            [prev_sample_r, prev_sample_tr], axis=1
        )  # combinazione
        return prev_sample, attentions

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _predict_eps_from_xstart_rot(self, x_t, t, pred_xstart):
        # devo trasformare x_t in rotazione
        x_t_term = so3_scale(
            quaternion_to_matrix(x_t),
            (
                extract_rot(self.sqrt_recip_alphas_cumprod, t, t.shape)
                / extract_rot(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape).flatten()
            ),
        )
        # devo trasformare noise in rotazione
        pred_xstart = so3_scale(
            quaternion_to_matrix(pred_xstart),
            1 / extract_rot(self.sqrt_recipm1_alphas_cumprod, t, t.shape),
        )  # [..., None]
        # Rotation = multiply by inverse op (matrices, so transpose)
        return x_t_term @ pred_xstart.transpose(-1, -2)

    # Algorithm 2 but save all images:
    @torch.no_grad()
    def p_sample_loop(self, shape, cond, edge_index, batch):
        # device = next(model.parameters()).device
        device = self.device

        b = shape[0]
        num_t = 9 if self.use_6dof else 3
        shape = torch.Size([b, num_t])
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device) * self.noise_weight
        imgs = []
        attentions = []
        pcd_feats = self.pcd_features(cond)
        # dimension problem..
        # x = IsotropicGaussianSO3(eps=torch.ones([], device=device)).sample(shape) # * self.noise_weight (qui definsico da dove partono tutte le rotazioni)
        # aggiungere qui zero starting
        # x, _ = torch.qr(torch.randn((b, 3, 3), device=device)) # anche questo restituisce una matrice R causale
        # This case is equal to noise_weight = 0
        truepos = torch.eye(3).to(device)
        x = truepos.repeat(b, 1, 1)

        x = matrix_to_quaternion(x)
        img = torch.concat([x, img], axis=1)  # combinazione
        # scomporre img in base alle due generazioni + combinarle dopo (coprire le dimensioni di b)
        # for i in tqdm(
        #     list(reversed(range(0, self.steps, self.inference_ratio))),
        #     desc="sampling loop time step",
        # ):

        for i in list(reversed(range(0, self.steps, self.inference_ratio))):
            img, atts = self.p_sample(
                img,
                torch.full((b,), i, device=device, dtype=torch.long),
                # time_t + i,
                i,
                edge_index=edge_index,
                pcd_feats=pcd_feats,
                batch=batch,
            )
            # projection of the point cloud? They don't apply this operation.
            # pcd_feats = self.pcd_features(projection(quaternion_to_matrix(img[:,:4]), translation(img[:,4:],cond)))
            attentions.append(atts)
            imgs.append(img)
        return imgs, attentions

    @torch.no_grad()
    def p_sample(self, x, t, t_index, edge_index, sampling_func, pcd_feats, batch):
        return sampling_func(
            x, t, t_index, edge_index, pcd_feats, batch
        )  # type of diffusion

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
        optimizer = Adafactor(self.parameters())
        return optimizer

        # optimizer = Adafactor(self.parameters(), lr=self.learning_rate)

        # """Build optimizer and lr scheduler."""
        # # lr = 1e-3
        # wd = 0
        # optimizer = optim.Adam(
        #     self.parameters(), lr=self.learning_rate, weight_decay=0.0
        # )
        # scheduler = "cosine"
        # if scheduler:
        #     assert scheduler in ["cosine"]
        #     total_epochs = self.max_epochs
        #     warmup_epochs = int(total_epochs * 0.0)
        #     scheduler = CosineAnnealingWarmupRestarts(
        #         optimizer,
        #         total_epochs,
        #         max_lr=self.learning_rate,
        #         min_lr=self.learning_rate / 100.0,
        #         warmup_steps=warmup_epochs,
        #     )
        #     return (
        #         [optimizer],
        #         [
        #             {
        #                 "scheduler": scheduler,
        #                 "interval": "epoch",
        #             }
        #         ],
        #     )
        # # optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        # return optimizer

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
            valids=batch.valids,
        )
        loss = sum([i for i in loss_dict.values()])
        if batch_idx == 0 and self.local_rank == 0:
            sampled_pos, _ = self.p_sample_loop(
                batch.x.shape, batch.pcds, batch.edge_index, batch=batch.batch
            )
            final_pos = sampled_pos[-1]

            save_path = Path(f"results/{self.logger.experiment.name}/train")
            for i in range(
                min(batch.batch.max().item() + 1, 4)
            ):  # save max 4 images during training loop
                idx = torch.where(batch.batch == i)[0]
                pcds = batch.pcds[idx]
                gt_pos = batch.x[idx]
                pred_pos = final_pos[idx]
                pred_t = pred_pos[:, 4:7]

                pred_r = pred_pos[:, :4]

                if self.use_6dof:
                    a1 = pred_pos[:, 7:10]
                    a2 = pred_pos[:, 10:13]
                    b1 = a1 / a1.norm(dim=-1)[:, None]
                    b2 = a2 - (a2 @ b1.T).diag()[:, None] * b1
                    b2 = b2 / b2.norm(dim=-1)[:, None]
                    b3 = torch.cross(b1, b2, dim=1)
                    rotm = torch.stack([b1, b2, b3], dim=-1)
                    pred_r = matrix_to_quaternion(rotm)
                points = []

                for idx_p, pcd in enumerate(pcds):
                    if self.use_6dof:
                        assert (
                            torch.isclose(
                                rotm[idx_p].norm(dim=0), torch.tensor(1.0)
                            ).all()
                            and torch.isclose(
                                rotm[idx_p].norm(dim=1), torch.tensor(1.0)
                            ).all()
                        ), f"Rotation matrix is not orthogonal {rot[idx_p]}"
                    col = matplotlib.colormaps["tab20"](idx_p % 20)
                    col = np.array(col)[None, :].repeat(1000, 0) * 255

                    # rot = gt_r[idx_p].cpu()[None, :]

                    # rot = ut3d.Rotation3D(rot)
                    # tras = gt_t[idx_p].cpu()[None, :]
                    # pcd_rot = ut3d.transform_pc(tras, rot, pcd[None, :].cpu())[
                    #     0
                    # ].numpy()
                    # pcd_rot = pcd.cpu().numpy() + pred_t[idx_p].cpu().numpy()

                    # rot = pred_r[idx_p].cpu()[None, :]

                    tras = pred_t[idx_p].cpu()[None, :]
                    rot = pred_r[idx_p].cpu()[None, :]

                    rot = Rotation3D(rot)
                    pcd_rot = transform_pc(tras, rot, pcd[None, :].cpu())[0].numpy()
                    # pcd_rot = (
                    #     pcd.cpu().numpy() @ pred_r[idx_p].cpu().numpy().T
                    #     + pred_t[idx_p].cpu().numpy()
                    # )

                    points.append(np.concatenate([pcd_rot, col[:, :3]], 1))

                    # pcd_vedo = vedo.Points(pcd_rot, c=col)
                #     pcds_vedo.append(pcd_vedo)
                #     plotter.add(pcd_vedo)

                # plotter.add_global_axes(2)
                # text = f"Frame: {id_t}"  # Replace this with your desired text
                # vedo.showText(
                #     text, pos=(0.2, 0.8), s=0.2, c="black", bg="white"
                # )  # Customize position, size, and appearance
            pcd_wandb = wandb.Object3D(np.concatenate(points, 0))

            self.logger.experiment.log(
                {f"train": pcd_wandb, "global_step": self.global_step}
            )

            # todo Save 3d pcd

        self.log_dict(loss_dict)
        # self.log("loss", loss)

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

                cat = batch.category[i]

                # Compute metrics
                pred_t = pred_pos[:, 4:7]
                pred_r = pred_pos[:, :4]

                # pred_r = matrix_to_quaternion(skew_to_rmat(pred_pos[:, :3] ))# applico trasformazioni.
                if self.use_6dof:
                    a1 = pred_pos[:, 7:10]
                    a2 = pred_pos[:, 10:13]
                    b1 = a1 / a1.norm(dim=-1)[:, None]
                    b2 = a2 - (a2 @ b1.T).diag()[:, None] * b1
                    b2 = b2 / b2.norm(dim=-1)[:, None]
                    b3 = torch.cross(b1, b2, dim=1)
                    rot = torch.stack([b1, b2, b3], dim=-1)
                    pred_r = matrix_to_quaternion(rot)

                gt_t = gt_pos[:, 4:]
                gt_r = gt_pos[:, :4]
                t_metric = trans_metrics(pred_t, gt_t)
                rot_metric = rot_metrics(pred_r, gt_r, metric="rmse")
                gd_metric = rot_metrics(pred_r, gt_r, metric="geodesic")
                part_acc = calc_part_acc(pcds, pred_t, gt_t, pred_r, gt_r, None)

                # loss = test_loss(pcds, pred_t, gt_t, pred_r, gt_r, training=False)

                # self.metrics["rmse_t"].update(t_metric)
                # self.metrics["rmse_r"].update(rot_metric)
                # self.metrics["gd_r"].update(gd_metric)
                # self.metrics["part_acc"].update(part_acc)

                self.metrics[f"rmse_t_{cat}"].update(t_metric)
                self.metrics[f"rmse_r_{cat}"].update(rot_metric)
                self.metrics[f"gd_r_{cat}"].update(gd_metric)
                self.metrics[f"part_acc_{cat}"].update(part_acc)
                # self.metrics["test_loss"].update(loss)

                if (
                    self.local_rank == 0
                    and batch_idx < 10
                    and i < min(batch.batch.max().item(), 4)
                ):
                    from . import utils_3d as ut3d

                    points = []

                    for id_t, denoised in enumerate(sampled_pos[-1:]):
                        pred_pos = denoised[idx]
                        # plotter.clear()

                        pred_t = pred_pos[:, 4:7]  # [:, 4:]
                        pred_r = pred_pos[:, :4]

                        if self.use_6dof:
                            a1 = pred_pos[:, 7:10]
                            a2 = pred_pos[:, 10:13]
                            b1 = a1 / a1.norm(dim=-1)[:, None]
                            b2 = a2 - (a2 @ b1.T).diag()[:, None] * b1
                            b2 = b2 / b2.norm(dim=-1)[:, None]
                            b3 = torch.cross(b1, b2, dim=1)
                            rot = torch.stack([b1, b2, b3], dim=-1)
                            pred_r = matrix_to_quaternion(rot)

                        pcds_vedo = []

                        for idx_p, pcd in enumerate(pcds):
                            col = matplotlib.colormaps["tab20"](idx_p % 20)
                            col = np.array(col)[None, :].repeat(1000, 0) * 255

                            # rot = gt_r[idx_p].cpu()[None, :]

                            # rot = ut3d.Rotation3D(rot)
                            # tras = gt_t[idx_p].cpu()[None, :]
                            # pcd_rot = ut3d.transform_pc(tras, rot, pcd[None, :].cpu())[
                            #     0
                            # ].numpy()
                            # pcd_rot = pcd.cpu().numpy() + pred_t[idx_p].cpu().numpy()

                            rot = pred_r[idx_p].cpu()[None, :]

                            rot = ut3d.Rotation3D(rot)
                            tras = pred_t[idx_p].cpu()[None, :]
                            pcd_rot = ut3d.transform_pc(tras, rot, pcd[None, :].cpu())[
                                0
                            ].numpy()

                            points.append(np.concatenate([pcd_rot, col[:, :3]], 1))

                            # pcd_vedo = vedo.Points(pcd_rot, c=col)
                        #     pcds_vedo.append(pcd_vedo)
                        #     plotter.add(pcd_vedo)

                        # plotter.add_global_axes(2)
                        # text = f"Frame: {id_t}"  # Replace this with your desired text
                        # vedo.showText(
                        #     text, pos=(0.2, 0.8), s=0.2, c="black", bg="white"
                        # )  # Customize position, size, and appearance
                    pcd_wandb = wandb.Object3D(np.concatenate(points, 0))

                    self.logger.experiment.log(
                        {f"val": pcd_wandb, "global_step": self.global_step}
                    )

                # Log PCD

            self.log_dict(self.metrics, prog_bar=True, on_step=False, on_epoch=True)
        # return accuracy_dict

    def validation_epoch_end(self, outputs) -> None:
        metrics = {"rmse_t", "rmse_r", "gd_r", "part_acc"}
        metrics_names = set(self.metrics) - {
            "rmse_t_AVG",
            "rmse_r_AVG",
            "gd_r_AVG",
            "part_acc_AVG",
        }

        for i in metrics:
            for j in metrics_names:
                if i in j:
                    self.avg_metrics[f"{i}_AVG"].update(
                        self.metrics[j].compute().item()
                    )

        self.log_dict(self.avg_metrics, prog_bar=True, on_step=False, on_epoch=True)

    def test_epoch_end(self, outputs) -> None:
        return self.validation_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
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
                cat = batch.category[i]

                # Compute metrics
                pred_t = pred_pos[:, 4:7]
                pred_r = pred_pos[:, :4]

                # pred_r = matrix_to_quaternion(skew_to_rmat(pred_pos[:, :3] ))# applico trasformazioni.
                if self.use_6dof:
                    a1 = pred_pos[:, 7:10]
                    a2 = pred_pos[:, 10:13]
                    b1 = a1 / a1.norm(dim=-1)[:, None]
                    b2 = a2 - (a2 @ b1.T).diag()[:, None] * b1
                    b2 = b2 / b2.norm(dim=-1)[:, None]
                    b3 = torch.cross(b1, b2, dim=1)
                    rot = torch.stack([b1, b2, b3], dim=-1)
                    pred_r = matrix_to_quaternion(rot)

                gt_t = gt_pos[:, 4:]
                gt_r = gt_pos[:, :4]
                t_metric = trans_metrics(pred_t, gt_t)
                rot_metric = rot_metrics(pred_r, gt_r, metric="rmse")
                gd_metric = rot_metrics(pred_r, gt_r, metric="geodesic")
                part_acc = calc_part_acc(pcds, pred_t, gt_t, pred_r, gt_r, None)

                # loss = test_loss(pcds, pred_t, gt_t, pred_r, gt_r, training=False)

                self.metrics[f"rmse_t_{cat}"].update(t_metric)
                self.metrics[f"rmse_r_{cat}"].update(rot_metric)
                self.metrics[f"gd_r_{cat}"].update(gd_metric)
                self.metrics[f"part_acc_{cat}"].update(part_acc)

                if self.save_eval_images:
                    index_dt = batch.data_id[i].item()
                    mesh_dir_path = (
                        Path("datasets")
                        / "breaking-bad"
                        / Path(self.test_dataset.dataset.data_list[index_dt])
                    )
                    num_pieces = (batch.batch == i).sum().item()

                    bb_path = Path(
                        self.test_dataset.dataset.data_list[index_dt]
                    ).parents._parts
                    bb_path.insert(3, f"{num_pieces}-parts")
                    bb_path[
                        4
                    ] = f"{rot_metric.item()}rot-{t_metric.item()}tras-{part_acc.item()}pa-{bb_path[4]}"

                    save_path = Path(
                        f"results/{self.logger.experiment.name}/viz_3d"
                    ) / "/".join(bb_path)

                    # Path(self.test_dataset.dataset.data_list[index_dt])
                    save_path.mkdir(parents=True, exist_ok=True)
                    mesh_list = os.listdir(mesh_dir_path)
                    mesh_list.sort()

                    for id_p in range((batch.batch == i).sum().item()):
                        px_all=[]
                        pr_all=[]
                        quat_all=[]
                        for id_t,smp in enumerate(sampled_pos):
                            pred= smp[idx][id_p]
                            px = pred[4:]
                            pr = pred[:4]
                            xyz=matrix_to_euler_angles(quaternion_to_matrix(pr),'XYZ')
                            px_all.append(px.cpu().numpy())
                            pr_all.append(xyz.cpu().numpy())
                            quat_all.append(pr.cpu().numpy())
                        px_all=np.stack(px_all)
                        pr_all=np.stack(pr_all)
                        np.savez(save_path / f"pred_{id_p}.npy",pos=px_all,rot=pr_all,quat=quat_all)




                        # mesh_path = mesh_dir_path / f"piece_{id_p}.obj"
                        mesh_path = mesh_dir_path / mesh_list[id_p]
                        mesh = trimesh.load(mesh_path)
                        mesh.export(save_path / f"original_{id_p}.ply")

                        # anti-rot trasl to move the mesh to the origin
                        gt_trasl = -gt_t[id_p].cpu().numpy()
                        gt_rot = quaternion_to_matrix(gt_r[id_p]).T.cpu().numpy()
                        transf = np.eye(4)
                        # transf[:3, :3] = gt_rot
                        transf[:3, 3] = gt_trasl
                        origin_mesh = mesh.apply_transform(transf)

                        transf = np.eye(4)
                        transf[:3, :3] = gt_rot
                        # transf[:3, 3] = gt_trasl

                        origin_mesh = origin_mesh.apply_transform(transf)
                        origin_mesh.export(save_path / f"init_{id_p}_origin.ply")
                        from copy import deepcopy

                        origin_mesh2 = deepcopy(origin_mesh)

                        gt_rot = quaternion_to_matrix(gt_r[id_p]).cpu().numpy()
                        gt_trasl = gt_t[id_p].cpu().numpy()

                        transf = np.eye(4)
                        transf[:3, :3] = gt_rot
                        transf[:3, 3] = gt_trasl

                        output_mesh_origin = origin_mesh.apply_transform(transf)
                        output_mesh_origin.export(
                            save_path / f"original_retransform_{id_p}.ply"
                        )

                        ours_rot = quaternion_to_matrix(pred_r[id_p]).cpu().numpy()
                        ours_trasl = pred_t[id_p].cpu().numpy()

                        transf = np.eye(4)
                        transf[:3, :3] = ours_rot
                        transf[:3, 3] = ours_trasl

                        output_mesh = origin_mesh2.apply_transform(transf)
                        output_mesh.export(save_path / f"predicted_{id_p}_origin.ply")

                        # cloud = trimesh.PointCloud(pcds[id_p].cpu().numpy())

                        # cloud.export(save_path / f"init_cloud_{id_p}_origin.ply")

                        # cloud = cloud.apply_transform(transf)
                        # cloud.export(save_path / f"predicted_cloud_{id_p}_origin.ply")

                        # rot1_r = Rotation3D(pred_r[id_p])

                        # pts1 = transform_pc(
                        #     pred_t[id_p], rot1_r, pcds[id_p]
                        # )  # [B, P, N, 3]

                        # cloud = trimesh.PointCloud(pts1.cpu().numpy())

                        # cloud.export(
                        #     save_path / f"predicted_cloud_trasform_{id_p}_origin_.ply"
                        # )

                # self.metrics["test_loss"].update(loss)

                # if (
                #     self.local_rank == 0
                #     and batch_idx < 10
                #     and i < min(batch.batch.max().item(), 4)
                # ):
                #     from . import utils_3d as ut3d

                #     points = []

                #     for id_t, denoised in enumerate(sampled_pos[-1:]):
                #         pred_pos = denoised[idx]
                #         # plotter.clear()

                #         pred_t = pred_pos[:, 4:7]  # [:, 4:]
                #         pred_r = pred_pos[:, :4]

                #         if self.use_6dof:
                #             a1 = pred_pos[:, 7:10]
                #             a2 = pred_pos[:, 10:13]
                #             b1 = a1 / a1.norm(dim=-1)[:, None]
                #             b2 = a2 - (a2 @ b1.T).diag()[:, None] * b1
                #             b2 = b2 / b2.norm(dim=-1)[:, None]
                #             b3 = torch.cross(b1, b2, dim=1)
                #             rot = torch.stack([b1, b2, b3], dim=-1)
                #             pred_r = matrix_to_quaternion(rot)

                #         pcds_vedo = []

                #         for idx_p, pcd in enumerate(pcds):
                #             col = matplotlib.colormaps["tab20"](idx_p % 20)
                #             col = np.array(col)[None, :].repeat(1000, 0) * 255

                #             # rot = gt_r[idx_p].cpu()[None, :]

                #             # rot = ut3d.Rotation3D(rot)
                #             # tras = gt_t[idx_p].cpu()[None, :]
                #             # pcd_rot = ut3d.transform_pc(tras, rot, pcd[None, :].cpu())[
                #             #     0
                #             # ].numpy()
                #             # pcd_rot = pcd.cpu().numpy() + pred_t[idx_p].cpu().numpy()

                #             rot = pred_r[idx_p].cpu()[None, :]

                #             rot = ut3d.Rotation3D(rot)
                #             tras = pred_t[idx_p].cpu()[None, :]
                #             pcd_rot = ut3d.transform_pc(tras, rot, pcd[None, :].cpu())[
                #                 0
                #             ].numpy()

                #             points.append(np.concatenate([pcd_rot, col[:, :3]], 1))

                #             # pcd_vedo = vedo.Points(pcd_rot, c=col)
                #         #     pcds_vedo.append(pcd_vedo)
                #         #     plotter.add(pcd_vedo)

                #         # plotter.add_global_axes(2)
                #         # text = f"Frame: {id_t}"  # Replace this with your desired text
                #         # vedo.showText(
                #         #     text, pos=(0.2, 0.8), s=0.2, c="black", bg="white"
                #         # )  # Customize position, size, and appearance
                #     pcd_wandb = wandb.Object3D(np.concatenate(points, 0))

                #     self.logger.experiment.log(
                #         {f"val": pcd_wandb, "global_step": self.global_step}
                #     )

                # Log PCD

            self.log_dict(self.metrics, prog_bar=True, on_step=False, on_epoch=True)

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
        valid_mask = part_valids == 1
        # shared-weight encoder
        valid_pcs = part_pcs[valid_mask]  # [n, N, 3]
        valid_feats = self.encoder(valid_pcs)  # [n, C]
        pc_feats = torch.zeros(B, P, self.pc_feat_dim).type_as(
            valid_feats
        )  # matrice degli zeri con le dimensioni degli output
        pc_feats[valid_mask] = valid_feats
        return pc_feats
