import colorsys
import enum
import logging
import math

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

import wandb

from . import backbones
from . import spatial_diffusion as sd

matplotlib.use("agg")


def matrix_cumprod(matrixes, dim):
    cumprods = []

    cumprod = torch.eye(matrixes[0].shape[0])
    for matrix in matrixes:
        cumprod = cumprod @ matrix
        cumprods.append(cumprod)
    return cumprods


class GNN_Diffusion(sd.GNN_Diffusion):
    def __init__(self, puzzle_sizes, loss_type="vb", lambda_loss=0.01, *args, **kwargs):
        K = puzzle_sizes[0][0] * puzzle_sizes[0][1]
        if "input_channels" not in kwargs:
            kwargs["input_channels"] = K
        if "output_channels" not in kwargs:
            kwargs["output_channels"] = K
        super().__init__(
            *args,
            **kwargs,
        )

        self.lambda_loss = lambda_loss
        self.puzzle_sizes = puzzle_sizes[0]
        self.loss_type = loss_type
        self.K = K
        Qs = []

        for t in range(self.steps):
            beta_t = self.betas[t]
            Q_t = (1 - beta_t) * torch.eye(self.K) + beta_t * torch.ones(
                (self.K, self.K)
            ) / self.K
            Qs.append(Q_t)

        self.register_buffer("Q_onestep", torch.stack(Qs))
        self.register_buffer("Q_onestep_transpose", torch.stack(Qs).transpose(1, 2))

        self.register_buffer(
            "overline_Q", torch.stack(matrix_cumprod(torch.stack(Qs), 0))
        )
        self.discrete = True
        self.save_hyperparameters()

    def init_backbone(self):
        self.model = backbones.Eff_GAT_Discrete(
            steps=self.steps,
            input_channels=self.input_channels,
            output_channels=self.output_channels,
        )

    def training_step(self, batch, batch_idx):
        # return super().training_step(*args, **kwargs)
        batch_size = batch.batch.max().item() + 1
        t = torch.randint(0, self.steps, (batch_size,), device=self.device).long()

        new_t = torch.gather(t, 0, batch.batch)

        loss = self.p_losses(
            batch.indexes % self.K,
            new_t,
            loss_type=self.loss_type,
            cond=batch.patches,
            edge_index=batch.edge_index,
            batch=batch.batch,
        )
        if batch_idx == 0 and self.local_rank == 0:
            indexes = self.p_sample_loop(
                batch.indexes.shape, batch.patches, batch.edge_index, batch=batch.batch
            )
            index = indexes[-1]

            save_path = Path(f"results/{self.logger.experiment.name}/train")
            for i in range(
                min(batch.batch.max().item(), 4)
            ):  # save max 4 images during training loop
                idx = torch.where(batch.batch == i)[0]
                patches_rgb = batch.patches[idx]
                gt_pos = batch.x[idx]
                n_patches = batch.patches_dim[i].tolist()
                y = torch.linspace(-1, 1, n_patches[0], device=self.device)
                x = torch.linspace(-1, 1, n_patches[1], device=self.device)
                xy = torch.stack(torch.meshgrid(x, y, indexing="xy"), -1)
                real_grid = einops.rearrange(xy, "x y c-> (x y) c")
                pos = real_grid[index[idx]]

                n_patches = batch.patches_dim[i]
                i_name = batch.ind_name[i]
                self.save_image(
                    patches_rgb=patches_rgb,
                    pos=pos,
                    gt_pos=gt_pos,
                    patches_dim=n_patches,
                    ind_name=i_name,
                    file_name=save_path,
                )

        self.log("loss", loss)

        return loss

    def on_predict_epoch_start(self):
        logging.info(f"Saving to results/{self.logger.experiment.name}/preds")

    def predict_step(self, batch, batch_idx):
        with torch.no_grad():
            preds = self.p_sample_loop(
                batch.indexes.shape, batch.patches, batch.edge_index, batch=batch.batch
            )
            for i in range(batch.batch.max() + 1):
                for loop_index, pred_last_index in enumerate(preds):
                    idx = torch.where(batch.batch == i)[0]
                    patches_rgb = batch.patches[idx]
                    gt_pos = batch.x[idx]
                    gt_index = batch.indexes[idx] % self.K

                    pred_index = pred_last_index[idx]
                    n_patches = batch.patches_dim[i].tolist()
                    i_name = f"{batch.ind_name[i]:03d}_{loop_index:03d}"

                    y = torch.linspace(-1, 1, n_patches[0], device=self.device)
                    x = torch.linspace(-1, 1, n_patches[1], device=self.device)
                    xy = torch.stack(torch.meshgrid(x, y, indexing="xy"), -1)
                    real_grid = einops.rearrange(xy, "x y c-> (x y) c")
                    pred_pos = real_grid[pred_index]

                    correct = (pred_index == gt_index).all()
                    save_path = Path(f"results/{self.logger.experiment.name}/val")
                    self.save_image(
                        patches_rgb=patches_rgb,
                        pos=pred_pos,
                        gt_pos=gt_pos,
                        patches_dim=n_patches,
                        ind_name=i_name,
                        file_name=save_path,
                        correct=correct,
                    )

    # forward diffusion
    def q_sample(self, x_start, t, overline_Q=None, eps=1e-9):
        if overline_Q is None:
            overline_Q = self.overline_Q
        noise = torch.rand(size=x_start.shape).to(x_start.device)
        noise = torch.clip(noise, torch.finfo(noise.dtype).tiny, 1.0)
        Q_t = overline_Q[t]
        q_logits = torch.log(
            torch.bmm(x_start.float().unsqueeze(1), Q_t) + eps
        ).squeeze()

        return torch.argmax(q_logits - torch.log(-torch.log(noise)), -1)

    def q_posterior_logits(
        self,
        x_t,
        x_start_logits,
        t,
        previous_t,
        K=None,
        overline_Q=None,
        eps=1e-8,
        use_x_start_logits=True,
    ):
        if overline_Q is None:
            overline_Q = self.overline_Q
        if K is None:
            K = self.K

        Q_ksteps_transpose = (
            overline_Q[t] @ torch.linalg.inv(overline_Q[previous_t])
        ).transpose(1, 2)
        Q_previous_t = overline_Q[previous_t]

        fact1 = torch.bmm(F.one_hot(x_t, K).float().unsqueeze(1), Q_ksteps_transpose)

        if use_x_start_logits:
            tzero_logits = x_start_logits
            fact2 = torch.bmm(F.softmax(x_start_logits).unsqueeze(1), Q_previous_t)
        else:
            tzero_logits = torch.log(x_start_logits + 1e-8)
            fact2 = torch.bmm(x_start_logits.unsqueeze(1), Q_previous_t)

        out = torch.log(fact1 + eps) + torch.log(fact2 + eps)

        return torch.where(
            t[:, None].tile(x_start_logits.shape[1]) == 0, tzero_logits, out.squeeze()
        )

    def p_losses(
        self,
        x_start,
        t,
        noise=None,
        loss_type="l1",
        cond=None,
        edge_index=None,
        batch=None,
    ):
        x_start_one_hot = torch.nn.functional.one_hot(x_start)

        x_noisy = self.q_sample(x_start=x_start_one_hot, t=t)

        patch_feats = self.visual_features(cond)
        batch_size = batch.max() + 1
        batch_one_hot = torch.nn.functional.one_hot(batch)
        prob = (
            batch_one_hot.float() @ torch.rand(batch_size, device=self.device)
            > self.classifier_free_prob
        )
        classifier_free_patch_feats = prob[:, None] * patch_feats

        prediction = self.forward_with_feats(
            x_noisy,
            t,
            cond,
            edge_index,
            patch_feats=classifier_free_patch_feats,
            batch=batch,
        )
        if loss_type == "cross_entropy":
            loss = F.cross_entropy(prediction, x_start, label_smoothing=1e-2)
        elif loss_type == "vb":
            model_logits = self.q_posterior_logits(x_noisy, prediction, t, t - 1)
            loss = self.vb_terms_bpd(prediction, model_logits, x_start, x_noisy, t)
        elif loss_type == "hybrid":
            xstart_loss = F.cross_entropy(prediction, x_start, label_smoothing=1e-2)
            model_logits = self.q_posterior_logits(x_noisy, prediction, t, t - 1)
            vb_loss = self.vb_terms_bpd(prediction, model_logits, x_start, x_noisy, t)
            loss = self.lambda_loss * xstart_loss + vb_loss
        else:
            raise Exception("Loss not implemented %s", loss_type)

        return loss

    @torch.no_grad()
    def p_sample(
        self, x, t, t_index, cond, edge_index, sampling_func, patch_feats, batch
    ):
        return sampling_func(x, t, t_index, cond, edge_index, patch_feats, batch)

    @torch.no_grad()
    def p_sample_ddpm(self, x, t, t_index, cond, edge_index, patch_feats, batch):
        prev_timestep = t - self.inference_ratio

        if self.classifier_free_prob > 0.0:
            model_output_cond = self.forward_with_feats(
                x, t, cond, edge_index, patch_feats=patch_feats, batch=batch
            )

            model_output_uncond = self.forward_with_feats(
                x,
                t,
                cond,
                edge_index,
                patch_feats=torch.zeros_like(patch_feats),
                batch=batch,
            )
            model_output = (
                1 + self.classifier_free_w
            ) * model_output_cond - self.classifier_free_w * model_output_uncond
        else:
            model_output = self.forward_with_feats(
                x, t, cond, edge_index, patch_feats=patch_feats, batch=batch
            )

        # estimate x_0

        logits = torch.where(
            t[:, None].tile(model_output.shape[1]) == 0,
            model_output,
            self.q_posterior_logits(x, model_output, t, prev_timestep),
        )

        mask = (t != 0).reshape(x.shape[0], *([1] * (len(x.shape)))).to(logits.device)
        noise = torch.rand(logits.shape).to(logits.device)
        noise = torch.clip(noise, torch.finfo(noise.dtype).tiny, 1.0)

        gumbel_noise = -torch.log(-torch.log(noise))
        sample = torch.argmax(logits + mask * gumbel_noise, -1)
        return sample

    # Algorithm 2 but save all images:
    @torch.no_grad()
    def p_sample_loop(self, shape, cond, edge_index, batch):
        # device = next(model.parameters()).device
        device = self.device

        b = shape[0]

        index = torch.randint(0, self.K, shape, device=device)

        imgs = []

        patch_feats = self.visual_features(cond)

        # time_t = torch.full((b,), i, device=device, dtype=torch.long)

        # time_t = torch.full((b,), 0, device=device, dtype=torch.long)

        for i in tqdm(
            list(reversed(range(0, self.steps, self.inference_ratio))),
            desc="sampling loop time step",
        ):
            index = self.p_sample(
                index,
                torch.full((b,), i, device=device, dtype=torch.long),
                # time_t + i,
                i,
                cond=cond,
                edge_index=edge_index,
                patch_feats=patch_feats,
                batch=batch,
            )

            imgs.append(index)
        return imgs

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            pred_indexes = self.p_sample_loop(
                batch.indexes.shape, batch.patches, batch.edge_index, batch=batch.batch
            )
            pred_last_index = pred_indexes[-1]

            for i in range(batch.batch.max() + 1):
                idx = torch.where(batch.batch == i)[0]
                patches_rgb = batch.patches[idx]
                gt_pos = batch.x[idx]
                gt_index = batch.indexes[idx] % self.K

                pred_index = pred_last_index[idx]
                n_patches = batch.patches_dim[i].tolist()
                i_name = batch.ind_name[i]

                y = torch.linspace(-1, 1, n_patches[0], device=self.device)
                x = torch.linspace(-1, 1, n_patches[1], device=self.device)
                xy = torch.stack(torch.meshgrid(x, y, indexing="xy"), -1)
                real_grid = einops.rearrange(xy, "x y c-> (x y) c")
                pred_pos = real_grid[pred_index]

                piece_acc = pred_index == gt_index
                correct = (pred_index == gt_index).all()

                if (
                    self.local_rank == 0
                    and batch_idx < 10
                    and i < min(batch.batch.max().item(), 4)
                ):
                    save_path = Path(f"results/{self.logger.experiment.name}/val")
                    self.save_image(
                        patches_rgb=patches_rgb,
                        pos=pred_pos,
                        gt_pos=gt_pos,
                        patches_dim=n_patches,
                        ind_name=i_name,
                        file_name=save_path,
                        correct=correct,
                    )

                self.metrics[f"{tuple(n_patches)}_nImages"].update(1)
                self.metrics["overall_nImages"].update(1)
                self.metrics["overall__piece_acc"].update(piece_acc)
                if correct:
                    # if (assignement[:, 0] == assignement[:, 1]).all():
                    self.metrics[f"{tuple(n_patches)}_acc"].update(1)
                    self.metrics["overall_acc"].update(1)
                    # accuracy_dict[tuple(n_patches)].append(1)
                else:
                    self.metrics[f"{tuple(n_patches)}_acc"].update(0)
                    self.metrics["overall_acc"].update(0)
                    # accuracy_dict[tuple(n_patches)].append(0)

            self.log_dict(self.metrics)
        # return accuracy_dict

    def vb_terms_bpd(
        self,
        pred_x_start_logits,
        model_logits,
        x_start,
        x_t,
        t,
        K=None,
        overline_Q=None,
    ):
        """Calculate specified terms of the variational bound.
        Args:
          model_fn: the denoising network
          x_start: original clean data
          x_t: noisy data
          t: timestep of the noisy data (and the corresponding term of the bound
            to return)
        Returns:
          a pair `(kl, pred_start_logits)`, where `kl` are the requested bound terms
          (specified by `t`), and `pred_x_start_logits` is logits of
          the denoised image.
        """
        if overline_Q is None:
            overline_Q = self.overline_Q
        if K is None:
            K = self.K

        true_logits = self.q_posterior_logits(
            x_t,
            F.one_hot(x_start).float(),
            t,
            t - 1,
            K=K,
            overline_Q=overline_Q,
            use_x_start_logits=False,
        )

        true_logits = torch.where(
            t[:, None] == 0, torch.log(F.one_hot(x_start) + 1e-8), true_logits
        )

        kl = categorical_kl_logits(logits1=true_logits, logits2=model_logits)
        assert kl.shape == x_start.shape
        kl = kl / torch.log(torch.tensor([2.0])).to(kl.device)

        decoder_nll = F.nll_loss(
            F.log_softmax(model_logits, dim=1), x_start, reduction="none"
        )

        decoder_nll = decoder_nll / torch.log(torch.tensor([2.0])).to(
            decoder_nll.device
        )

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_start) || p(x_{t-1}|x_t))

        return torch.where(t == 0, decoder_nll, kl).mean()


def categorical_kl_logits(logits1, logits2, eps=1.0e-6):
    """KL divergence between categorical distributions.
    Distributions parameterized by logits.
    Args:
      logits1: logits of the first distribution. Last dim is class dim.
      logits2: logits of the second distribution. Last dim is class dim.
      eps: float small number to avoid numerical issues.
    Returns:
      KL(C(logits1) || C(logits2)): shape: logits1.shape[:-1]
    """
    out = F.softmax(logits1 + eps, dim=-1) * (
        F.log_softmax(logits1 + eps, dim=-1) - F.log_softmax(logits2 + eps, dim=-1)
    )
    return torch.sum(out, dim=-1)


def meanflat(x: torch.Tensor):
    """Take the mean over all axes except the first batch dimension."""
    return torch.mean(x, dim=tuple(range(1, len(x.shape))))
