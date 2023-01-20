import timm
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GraphNorm

from .exophormer_gnn import Exophormer_GNN
from .Transformer_GNN import Transformer_GNN
from torchvision.transforms.functional import rotate

class Eff_GAT(nn.Module):
    """
    This model has 45M parameters


    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        steps,
        input_channels=2,
        output_channels=2,
        n_layers=4,
        visual_pretrained=True,
        freeze_backbone=False,
        model="efficientnet_b0",
        architecture="transformer",
        virt_nodes=4,
        all_equivariant=False
    ) -> None:
        super().__init__()
        self.visual_backbone = timm.create_model(
            model, pretrained=visual_pretrained, features_only=True
        )
        self.all_equivariant=all_equivariant
        self.model = model
        self.combined_features_dim = {
            "resnet18": 1088 + 32 + 32, #3136,
            "resnet50": 12352,
            "efficientnet_b0": 1088 + 32 + 32,
        }[model]

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.freeze_backbone = freeze_backbone

        if architecture == "transformer":
            self.gnn_backbone = Transformer_GNN(
                self.combined_features_dim,
                n_layers=n_layers,
                hidden_dim=32 * 8,
                heads=8,
                output_size=self.combined_features_dim,
            )
        elif architecture == "exophormer":
            self.gnn_backbone = Exophormer_GNN(
                self.combined_features_dim,
                n_layers=n_layers,
                hidden_dim=32 * 8,
                heads=8,
                output_size=self.combined_features_dim,
                virt_nodes=virt_nodes
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
        self.final_mlp = nn.Sequential(
            nn.Linear(self.combined_features_dim, 32),
            nn.GELU(),
            nn.Linear(32, output_channels),
        )
        if self.model == 'resnet18':
            self.linear1 = nn.Linear(8192, 544) # 2560 is the dimension of efficientnet. trovare modo per generalizzare
            self.linear2 = nn.Linear(4096, 544) # 1792 is the dimension of efficientnet. trovare modo per generalizzare               
        else:
            self.linear1 = nn.Linear(2560, 544) # 2560 is the dimension of efficientnet. trovare modo per generalizzare
            self.linear2 = nn.Linear(1792, 544) # 1792 is the dimension of efficientnet. trovare modo per generalizzare
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
        time_feats = self.time_emb(time)  # embedding, int -> 32
        pos_feats = self.pos_mlp(xy_pos)  # MLP, (x, y) -> 32
        # COMBINE  and transform with MLP
        combined_feats = torch.cat([patch_feats, pos_feats, time_feats], -1)
        combined_feats = self.mlp(combined_feats)

        # GNN
        feats, attentions = self.gnn_backbone(
            x=combined_feats, edge_index=edge_index, batch=batch
        )

        # Residual + final transform
        final_feats = self.final_mlp(
            feats + combined_feats
        )  # combined -> (err_x, err_y)

        return final_feats, attentions

    def visual_features(self, patch_rgb):
        patch_rgb = (patch_rgb - self.mean) / self.std
        if self.freeze_backbone:
            with torch.no_grad():
                feats = self.visual_backbone.forward(patch_rgb)
        else:
            if self.all_equivariant:
                # definition of the 4 different forward pass
                feats = [self.visual_backbone.forward(patch_rgb[:, i, :, :, :]) for i in range(4)]
                # Normalize operation
                #feats = [torch.nn.functional.normalize(rotate(feats[1][i], 270), dim = 3) + torch.nn.functional.normalize(rotate(feats[2][i], 180), dim = 3) + \
                #        torch.nn.functional.normalize(rotate(feats[3][i], 90), dim = 3) + torch.nn.functional.normalize(feats[0][i], dim= 3) for i in range(len(feats[1]))]
                # Creation of the 4 equivariant solution
                feats1 = [(rotate(feats[1][i], 270) + rotate(feats[2][i], 180) + rotate(feats[3][i], 90) + feats[0][i])/4 for i in range(len(feats[1]))]
                feats2 = [(feats[1][i] + rotate(feats[2][i], 270) + rotate(feats[3][i], 180) + rotate(feats[0][i], 90))/4 for i in range(len(feats[1]))]
                feats3 = [(rotate(feats[1][i], 90) + feats[2][i] + rotate(feats[3][i], 270) + rotate(feats[0][i], 180))/4 for i in range(len(feats[1]))]
                feats4 = [(rotate(feats[1][i], 180) + rotate(feats[2][i], 90) + feats[3][i] + rotate(feats[0][i], 270))/4 for i in range(len(feats[1]))]
                
                # Combination of these features in a 5 tensors
                feats = [torch.stack((feats1[i], feats2[i], feats3[i], feats4[i]), dim = 1) for i in range(len(feats[1]))]
                # The below construction works worse                
                #bs, n_rots, channels, H, W = patch_rgb.shape
                #feats = self.visual_backbone.forward(patch_rgb.view(bs * n_rots, channels, H, W))
                #feats = [torch.nn.functional.normalize(feats[i].view(bs, n_rots, *feats[i].size()[1:]), dim = 3) for i in range(len(feats))]
                #feats = [rotate(feats[i][:, 1, :, :, :], 270) + rotate(feats[i][:, 2, :, :, :], 180) + \
                #         rotate(feats[i][:, 3, :, :, :], 90) + feats[i][:, 0, :, :, :] for i in range(len(feats))]
            else:
                feats = self.visual_backbone.forward(patch_rgb)
        feats = {
            "efficientnet_b0": [
                feats[2].reshape(patch_rgb.shape[0], -1),
                feats[3].reshape(patch_rgb.shape[0], -1),
            ],
            "resnet50": [
                feats[2].reshape(patch_rgb.shape[0], -1),
                feats[3].reshape(patch_rgb.shape[0], -1),
            ],
            "resnet18": [
                feats[2].reshape(patch_rgb.shape[0], -1),
                feats[3].reshape(patch_rgb.shape[0], -1),
            ],
        }[self.model]
        if self.all_equivariant:
            if self.model == 'resnet18':
                feats[0] = self.linear1(feats[0].view(feats[0].size(0), -1))
                feats[1] = self.linear2(feats[1].view(feats[1].size(0), -1))
            else:                
                feats[0] = self.linear1(feats[0].view(feats[0].size(0), -1))
                feats[1] = self.linear2(feats[1].view(feats[1].size(0), -1))
            return torch.cat(feats, -1)
        else:
            patch_feats = torch.cat(feats, -1)
            return patch_feats
