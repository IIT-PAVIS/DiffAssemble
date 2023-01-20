import timm
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GraphNorm
from torch.nn import functional as F

from .Transformer_GNN import Transformer_GNN
from .resnet_equivariant import ResNet18, ResNet34, ResNet50

class Eff_GAT(nn.Module):
    """
    This model has 45M parameters


    Args:
        nn (_type_): _description_
    """

    def __init__(self, steps, input_channels=2, output_channels=2) -> None:
        super().__init__()

        self.visual_backbone = ResNet18()
        #self.visual_backbone = timm.create_model(
        #    "efficientnet_b0", pretrained=True, features_only=True
        #)
        self.input_channels = input_channels
        self.output_channels = output_channels
        # visual_feats = 448  # hardcoded

        self.combined_features_dim = 1088 + 32 + 32 #+ 32 + 32 #resnet32
        #97792 + 32 + 32 resnet50
        

        # self.gnn_backbone = torch_geometric.nn.models.GAT(
        #     in_channels=self.combined_features_dim,
        #     hidden_channels=256,
        #     num_layers=2,
        #     out_channels=self.combined_features_dim,
        # )
        self.gnn_backbone = Transformer_GNN(
            self.combined_features_dim,
            hidden_dim=32 * 8,
            heads=8,
            output_size=self.combined_features_dim,
        )
        self.time_emb = nn.Embedding(steps, 32)
        self.pos_mlp = nn.Sequential(
            nn.Linear(input_channels, 16), nn.GELU(), nn.Linear(16, 32)
        )
        # self.GN = GraphNorm(self.combined_features_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.combined_features_dim, 128),
            nn.GELU(),
            nn.Linear(128, self.combined_features_dim),
        )

        self.mlp_t = nn.Sequential(
            nn.Linear(self.combined_features_dim, 256),
            nn.GELU(),
            nn.Linear(256, 2),
        )

        self.mlp_r = nn.Sequential(
            nn.Linear(self.combined_features_dim, 256),
            nn.GELU(),
            nn.Linear(256, 2),
        )

        mean = torch.tensor([0.4850, 0.4560, 0.4060])[None, :, None, None]
        std = torch.tensor([0.2290, 0.2240, 0.2250])[None, :, None, None]
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, xy_pos, time, patch_rgb, edge_index, batch):
        # patch_rgb = (patch_rgb - self.mean) / self.std

        ## fe[3].reshape(fe[0].shape[0],-1)
        # patch_feats = self.visual_backbone.forward(patch_rgb)[3].reshape(
        # patch_rgb.shape[0], -1
        # )
        # patch_feats = patch_feats

        patch_feats = self.visual_features(patch_rgb)
        final_feats = self.forward_with_feats(
            xy_pos, time, patch_rgb, edge_index, patch_feats=patch_feats, batch=batch
        )
        return final_feats

    def forward_with_feats(
        self: nn.Module,
        xy_pos: Tensor,
        time: Tensor,
        patch_rgb: Tensor,
        edge_index: Tensor,
        patch_feats: Tensor,
        batch,
    ):
        time_feats = self.time_emb(time)  # type: ignore # embedding, int -> 32
        pos_feats = self.pos_mlp(xy_pos)  # type: ignore # MLP, (x, y) -> 32


        # COMBINE  and transform with MLP
        combined_feats = torch.cat([patch_feats, pos_feats, time_feats], -1)
        combined_feats = self.mlp(combined_feats) # type: ignore

        # GNN
        feats = self.gnn_backbone(x=combined_feats, edge_index=edge_index) # type: ignore

        # Residual + final transform
        t_pred = self.mlp_t(feats + combined_feats)  # combined -> (err_x, err_y)
        r_pred = self.mlp_r(feats + combined_feats)  # combined -> (err_x, err_y)
        r_pred = F.normalize(r_pred, p=2, dim=-1)
        # t_pred = final_feats[:,3:]
        return torch.hstack((t_pred, r_pred))
        

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
