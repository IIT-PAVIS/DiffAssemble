import timm
import torch
import torch.nn as nn
from torch import Tensor

from .Transformer_GNN import Transformer_GNN


class Dark_TFConv(nn.Module):
    """
    This model has 45M parameters


    Args:
        nn (_type_): _description_
    """

    def __init__(self, steps) -> None:
        super().__init__()

        self.visual_backbone = timm.create_model(
            "cspdarknet53", pretrained=True, features_only=True
        )

        self.combined_features_dim = 4096 + 32 + 32

        # self.gnn_backbone = torch_geometric.nn.models.GAT(
        #     in_channels=self.combined_features_dim,
        #     hidden_channels=256,
        #     num_layers=2,
        #     out_channels=self.combined_features_dim,
        # )
        self.gnn_backbone = Transformer_GNN(
            self.combined_features_dim,
            hidden_dim=128 * 4,
            heads=4,
            output_size=self.combined_features_dim,
        )
        self.time_emb = nn.Embedding(steps, 32)
        self.pos_mlp = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, 32))
        self.mlp = nn.Sequential(
            nn.Linear(self.combined_features_dim, 128),
            nn.GELU(),
            nn.Linear(128, self.combined_features_dim),
        )
        self.final_mlp = nn.Sequential(
            nn.Linear(self.combined_features_dim, 64), nn.GELU(), nn.Linear(64, 2)
        )

        mean = torch.tensor([0.4850, 0.4560, 0.4060])[None, :, None, None]
        std = torch.tensor([0.2290, 0.2240, 0.2250])[None, :, None, None]
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, xy_pos, time, patch_rgb, edge_index, batch):
        patch_rgb = (patch_rgb - self.mean) / self.std

        # fe[3].reshape(fe[0].shape[0],-1)
        patch_feats = self.visual_backbone.forward(patch_rgb)[3].reshape(
            patch_rgb.shape[0], -1
        )
        # patch_feats = patch_feats
        time_feats = self.time_emb(time)
        pos_feats = self.pos_mlp(xy_pos)
        combined_feats = torch.cat([patch_feats, pos_feats, time_feats], -1)
        combined_feats = self.mlp(combined_feats)
        feats = self.gnn_backbone(x=combined_feats, edge_index=edge_index)
        final_feats = self.final_mlp(feats + combined_feats)

        return final_feats

    def forward_with_feats(
        self: nn.Module,
        xy_pos: Tensor,
        time: Tensor,
        patch_rgb: Tensor,
        edge_index: Tensor,
        patch_feats: Tensor,
        batch=None,
    ):
        # mean = patch_rgb.new_tensor([0.4850, 0.4560, 0.4060])[None, :, None, None]
        # std = patch_rgb.new_tensor([0.2290, 0.2240, 0.2250])[None, :, None, None]
        # if patch_feats == None:

        # patch_rgb = (patch_rgb - self.mean) / self.std

        # # fe[3].reshape(fe[0].shape[0],-1)
        # patch_feats = self.visual_backbone.forward(patch_rgb)[3].reshape(
        # patch_rgb.shape[0], -1
        # )
        # patch_feats = patch_feats
        time_feats = self.time_emb(time)
        pos_feats = self.pos_mlp(xy_pos)
        combined_feats = torch.cat([patch_feats, pos_feats, time_feats], -1)
        combined_feats = self.mlp(combined_feats)
        feats = self.gnn_backbone(x=combined_feats, edge_index=edge_index)
        final_feats = self.final_mlp(feats + combined_feats)

        return final_feats

    def visual_features(self, patch_rgb):
        patch_rgb = (patch_rgb - self.mean) / self.std

        # fe[3].reshape(fe[0].shape[0],-1)
        patch_feats = self.visual_backbone.forward(patch_rgb)[3].reshape(
            patch_rgb.shape[0], -1
        )
        return patch_feats
