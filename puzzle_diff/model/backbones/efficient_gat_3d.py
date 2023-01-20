import timm
import torch
import torch.nn as nn
from pytorch3d.transforms import matrix_to_quaternion
from torch import Tensor
from torch.nn import functional as F
from torch_geometric.nn import GraphNorm
from .exophormer_gnn import Exophormer_GNN
from .pointnet import PointNet, PretrainedPointnet, PointNetPlus
from .Transformer_GNN import Transformer_GNN
from .vnn.vn_pointnet import PointNetEncoder
from .vnn.vn_dgcnn import VN_DGCNN
from .gcn import GCN

def orthogonalise(mat):
    """Orthogonalise rotation/affine matrices

    Ideally, 3D rotation matrices should be orthogonal,
    however during creation, floating point errors can build up.
    We SVD decompose our matrix as in the ideal case S is a diagonal matrix of 1s
    We then round the values of S to [-1, 0, +1],
    making U @ S_rounded @ V.T an orthonormal matrix close to the original.
    """
    orth_mat = mat.clone()
    u, s, v = torch.svd(mat[..., :3, :3])
    orth_mat[..., :3, :3] = u @ torch.diag_embed(s.round()) @ v.transpose(-1, -2)
    return orth_mat


def vec2skew(vec: torch.Tensor) -> torch.Tensor:
    skew = torch.repeat_interleave(torch.zeros_like(vec).unsqueeze(-1), 3, dim=-1)
    skew[..., 2, 1] = vec[..., 0]
    skew[..., 2, 0] = -vec[..., 1]
    skew[..., 1, 0] = vec[..., 2]
    return skew - skew.transpose(-1, -2)


def skew_to_rmat(vmat, check=False):
    """ """
    rmat = vec2skew(vmat)
    out = torch.matrix_exp(rmat)
    if not check:
        return out
    else:
        return orthogonalise(out)


class Eff_GAT_3d(nn.Module):
    """
    This model has 45M parameters


    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        steps,
        input_channels=7,  # 3d : 3 trans and 4 rot
        t_channels=3,
        r_channels=3,
        n_layers=4,
        architecture='transformer',
            virt_nodes=8,
        backbone="pointnet",
        freeze_backbone=False,
        use_vn_dgcnn_equiv_inv_mp=False,
    ) -> None:
        super().__init__()
        self.use_vn_dgcnn_equiv_inv_mp = use_vn_dgcnn_equiv_inv_mp

        if backbone == "pointnet_inv":
            net = PretrainedPointnet()
            self.pcd_backbone = net.feat
            feat_dim = 1024
        elif backbone == "pointnet":
            feat_dim = 128
            self.pcd_backbone = PointNet(feat_dim=feat_dim)
        elif backbone == "pointnet_plus":
            self.pcd_backbone = PointNetPlus()
            feat_dim = 256
        elif backbone == "vn_dgcnn":
            feat_dim_out = 128
            feat_dim = 768 #1024  # 768 equivariant feats + 256 invariant feats [equi, inv]
            self.pcd_backbone = VN_DGCNN(feat_dim=feat_dim_out)  # vn_dgcnn PointNet
        elif backbone == "vn_dgcnn_inv":
            feat_dim_out = 128
            feat_dim = 256 # 768 equivariant feats + 256 invariant feats [equi, inv]
            self.pcd_backbone = VN_DGCNN(feat_dim=feat_dim_out, inv=True)  # vn_dgcnn PointNet
        elif backbone == "vnn":
            feat_dim = 2104
            self.pcd_backbone = nn.Sequential(
                PointNetEncoder(), nn.Linear(2046, 2104)  # vnn PointNet
            )
        else:
            raise Exception(f"Backbone not implemented {backbone}")

        self.combined_features_dim = feat_dim + 32 + 32
        self.gnn_feat_dim = self.combined_features_dim  # before only combined_features

        self.input_channels = input_channels

        self.freeze_backbone = freeze_backbone

        if architecture == "transformer":
            self.gnn_backbone = Transformer_GNN(
                self.gnn_feat_dim,
                n_layers=n_layers,
                hidden_dim=32 * 8,
                heads=8,
                output_size=self.gnn_feat_dim
            )
        elif architecture == "gcn":
            self.gnn_backbone = GCN(
                self.gnn_feat_dim,
                hidden_dim=32 * 8,
                output_size=self.gnn_feat_dim,
            )
        elif architecture == "exophormer":
            self.gnn_backbone = Exophormer_GNN(
                self.gnn_feat_dim,
                n_layers=n_layers,
                hidden_dim=32 * 8,
                heads=8,
                output_size=self.gnn_feat_dim,
                virt_nodes=virt_nodes
            )


        self.time_emb = nn.Embedding(steps, 32)
        self.pos_mlp = nn.Sequential(
            nn.Linear(input_channels, 16), nn.GELU(), nn.Linear(16, 32)
        )
        # self.GN = GraphNorm(self.combined_features_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.combined_features_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, self.gnn_feat_dim),
            nn.LeakyReLU(0.2),
        )
        self.mlp_t = nn.Sequential(
            nn.Linear(self.gnn_feat_dim, 256),
            nn.GELU(),
            nn.Linear(256, t_channels),
        )

        self.mlp_r = nn.Sequential(
            nn.Linear(self.gnn_feat_dim, 256),
            nn.GELU(),
            nn.Linear(256, r_channels),
        )
        # self.mlp_t = nn.Sequential(
        #    nn.Linear(self.gnn_feat_dim, t_channels),
        # )

        # self.mlp_r = nn.Sequential(
        #    nn.Linear(self.gnn_feat_dim, r_channels),
        # )
        # self.final_mlp = nn.Sequential(
        #    nn.Linear(self.gnn_feat_dim, 32),
        # nn.GELU(),
        #    nn.Linear(32, t_channels + r_channels),
        # )

    def forward(self, xy_pos, time, pcd, edge_index, batch):
        pcd_feats = self.pcd_features(pcd)
        final_feats = self.forward_with_feats(
            xy_pos, time, edge_index, pcd_feats=pcd_feats, batch=batch
        )
        return final_feats

    def forward_with_feats(
        self: nn.Module,
        xy_pos: Tensor,
        time: Tensor,
        edge_index: Tensor,
        pcd_feats: Tensor,
        batch,
    ):
        time_feats = self.time_emb(time)  # embedding, int -> 32
        pos_feats = self.pos_mlp(xy_pos)  # MLP, (x, y) -> 32

        # COMBINE  and transform with MLP
        combined_feats = torch.cat([pcd_feats, pos_feats, time_feats], -1)
        combined_feats = self.mlp(combined_feats)
        if (
            type(self.pcd_backbone).__name__ == "VN_DGCNN"
            and self.use_vn_dgcnn_equiv_inv_mp
        ):
            combined_feats_equivariant = torch.clone(combined_feats)
            combined_feats_equivariant[:, 768:1024] = 0
            combined_feats_invariant = torch.clone(combined_feats)
            combined_feats_equivariant[:, :768] = 0
            combined_feats_2 = torch.cat(
                [combined_feats_equivariant, combined_feats_invariant], 0
            )
            edge_index_2 = torch.clone(edge_index)
            edge_index_2[0, :] = edge_index[0, :] + pcd_feats.shape[0]
            feats, attentions = self.gnn_backbone(
                x=combined_feats_2, edge_index=edge_index_2, batch=batch
            )
            feats = feats[: pcd_feats.shape[0], :]
        else:
            # GNN
            feats, attentions = self.gnn_backbone(
                x=combined_feats, edge_index=edge_index, batch=batch
            )

        # Residual + final transform
        t_pred = self.mlp_t(feats + combined_feats)  # combined -> (err_x, err_y)

        r_pred = self.mlp_r(feats + combined_feats)  # combined -> (err_x, err_y)
        # r_pred = F.normalize(r_pred, p=2, dim=-1)
        # final_feats = self.final_mlp(feats + combined_feats)

        r_pred = matrix_to_quaternion(skew_to_rmat(r_pred))
        r_pred = F.normalize(r_pred, p=2, dim=-1)
        # t_pred = final_feats[:,3:]
        return torch.hstack((r_pred, t_pred)), attentions
        # return r_pred, attentions

        # TODO Add rot normalizer
        # TODO Add instance and label embeddings per node

        #
        # final_feats[:, :4] = rot

        # return final_feats, attentions

    def pcd_features(self, pcd):
        if self.freeze_backbone:
            with torch.no_grad():
                feats = self.pcd_backbone(pcd)
        else:
            feats = self.pcd_backbone.forward(pcd)
        return feats
