import torch
import torch.nn as nn
import pytorch_lightning as pl
from .vn_layers import *

class VN_DGCNN(pl.LightningModule):
    def __init__(self, feat_dim, inv = False):
        super(VN_DGCNN, self).__init__()
        self.n_knn = 20
        self.inv = inv
        # num_part = feat_dim  # 原版是做partseg,所以num_part=feat_dim
        pooling = 'mean'
        self.conv1 = VNLinearLeakyReLU(2, 64 // 3)
        self.conv2 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv3 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)
        self.conv4 = VNLinearLeakyReLU(64 // 3, 64 // 3)
        self.conv5 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)
        self.VnInv = VNStdFeature(2 * feat_dim, dim=3, normalize_frame=False)

        if pooling == 'max':
            self.pool1 = VNMaxPool(64 // 3)  # max_pooling
            self.pool2 = VNMaxPool(64 // 3)
            self.pool3 = VNMaxPool(64 // 3)
            self.pool4 = VNMaxPool(2 * feat_dim)
        elif pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
            self.pool4 = mean_pool

        self.conv6 = VNLinearLeakyReLU(64 // 3 * 3, feat_dim, dim=4, share_nonlinearity=True)
        self.linear0 = nn.Linear(3, 2 * feat_dim) # 3

    def forward(self, x):

        # x: (batch_size, 3, num_points)
        # l: (batch_size, 1, 16)
        x = x.transpose(-2, -1)
        batch_size = x.size(0)
        num_points = x.size(2)
        #l = x[:, 0, 0:16].reshape(batch_size, 1, 16)

        x = x.unsqueeze(1) # (32, 1, 3, 1024)

        x = get_graph_feature(x, k=self.n_knn) # (32, 2, 3, 1024, 20)

        x = self.conv1(x) # (32, 21, 3, 1024, 20)
        x = self.conv2(x) # (32, 21, 3, 1024, 20)
        x1 = self.pool1(x) # (32, 21, 3, 1024)

        x = get_graph_feature(x1, k=self.n_knn)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = self.pool2(x)

        x = get_graph_feature(x2, k=self.n_knn)
        x = self.conv5(x)
        x3 = self.pool3(x)

        x123 = torch.cat((x1, x2, x3), dim=1)

        x = self.conv6(x123)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x = self.pool4(x)  # [batch, feature_dim, 3]
        x1, z0 = self.VnInv(x)
        x1 = self.linear0(x)
        x1 = torch.mean(x1, dim=1) # [batch, feature_dim]
        # reshape of the equivariant
        x = x.view(batch_size, -1)
        if not self.inv:
            return x
        else:
            return x1
        #GF = torch.bmm(
        #        x1, x
        #    ).reshape(
        #        batch_size, -1
        #    ) 
        #return GF # [batch, feature_dim * 3]
        #return torch.cat([x, x1], axis = 1)  # [batch, feature_dim * 3], [batch, feature_dim]
        # transpose后x.shape: (batch_size, num_points, feat_dim())


def get_graph_feature(x, k=20, idx=None, x_coord=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if x_coord is None:  # dynamic knn graph
            idx = knn(x, k=k)
        else:  # fixed knn graph with input point coordinates
            idx = knn(x_coord, k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3)
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 4, 1, 2).contiguous()

    return feature

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx