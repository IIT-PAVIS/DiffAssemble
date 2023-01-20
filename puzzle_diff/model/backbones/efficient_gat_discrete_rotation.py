import timm
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GraphNorm

from .Transformer_GNN import Transformer_GNN


class Eff_GAT_Discrete_ROT(nn.Module):
    """
    This model has 45M parameters


    Args:
        nn (_type_): _description_
    """

    def __init__(
        self, steps, input_channels, output_channels, rotation_channels=4
    ) -> None:
        super().__init__()

        self.visual_backbone = timm.create_model(
            "efficientnet_b0", pretrained=True, features_only=True
        )
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.rotation_channels = rotation_channels
        # visual_feats = 448  # hardcoded

        self.combined_features_dim = 1088 + 32 + 32

        self.gnn_backbone = Transformer_GNN(
            self.combined_features_dim,
            hidden_dim=32 * 8,
            heads=8,
            output_size=self.combined_features_dim,
        )

        self.time_emb = nn.Embedding(steps, 32)
        self.pos_mlp = nn.Embedding(input_channels, 32)

        # self.GN = GraphNorm(self.combined_features_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.combined_features_dim, 128),
            nn.GELU(),
            nn.Linear(128, self.combined_features_dim),
        )
        self.final_mlp = nn.Sequential(
            nn.Linear(self.combined_features_dim, 32),
            nn.GELU(),
            nn.Linear(32, output_channels),
        )

        self.final_mlp_rot = nn.Sequential(
            nn.Linear(self.combined_features_dim, 32),
            nn.GELU(),
            nn.Linear(32, self.rotation_channels),
        )
        mean = torch.tensor([0.4850, 0.4560, 0.4060])[None, :, None, None]
        std = torch.tensor([0.2290, 0.2240, 0.2250])[None, :, None, None]
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, xy_pos, rot, time, patch_rgb, edge_index, batch):
        # patch_rgb = (patch_rgb - self.mean) / self.std

        ## fe[3].reshape(fe[0].shape[0],-1)
        # patch_feats = self.visual_backbone.forward(patch_rgb)[3].reshape(
        # patch_rgb.shape[0], -1
        # )
        # patch_feats = patch_feats

        patch_feats = self.visual_features(patch_rgb)
        final_feats = self.forward_with_feats(
            xy_pos,
            rot,
            time,
            patch_rgb,
            edge_index,
            patch_feats=patch_feats,
            batch=batch,
        )
        return final_feats

    def forward_with_feats(
        self: nn.Module,
        xy_pos: Tensor,
        rot: Tensor,
        time: Tensor,
        patch_rgb: Tensor,
        edge_index: Tensor,
        patch_feats: Tensor,
        batch,
    ):
        time_feats = self.time_emb(time)  # embedding, int -> 32

        pos_feats = self.pos_mlp(xy_pos)  # MLP, (x, y) -> 32

        # COMBINE  and transform with MLP
        combined_feats = torch.cat([patch_feats, pos_feats, time_feats], -1)
        combined_feats = self.mlp(combined_feats)

        # GNN
        feats = self.gnn_backbone(x=combined_feats, edge_index=edge_index)

        # Residual + final transform
        final_feats = self.final_mlp(
            feats + combined_feats
        )  # combined -> (err_x, err_y)

        final_rot = self.final_mlp_rot(feats + combined_feats)

        return final_feats, final_rot

    def visual_features(self, patch_rgb):
        patch_rgb = (patch_rgb - self.mean) / self.std

        feats = self.visual_backbone.forward(patch_rgb)
        patch_feats = torch.cat(
            [
                feats[2].reshape(patch_rgb.shape[0], -1),
                feats[3].reshape(patch_rgb.shape[0], -1),
            ],
            -1,
        )

        # patch_feats = self.visual_backbone.forward(patch_rgb)[3].reshape(
        # patch_rgb.shape[0], -1
        # )
        return patch_feats
