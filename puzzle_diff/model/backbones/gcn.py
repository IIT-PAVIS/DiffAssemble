from torch import nn
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size) -> None:
        super().__init__()

        self.module_list = nn.ModuleList(
            [
                GCNConv(input_size, out_channels=hidden_dim),
                GCNConv(hidden_dim, out_channels=output_size),
            ]
        )

    def forward(self, x, edge_index, batch, *args):
        x = self.module_list[0](x=x, edge_index=edge_index)
        x = nn.functional.gelu(x)
        x = self.module_list[1](x=x, edge_index=edge_index)
        x = nn.functional.gelu(x)

        return x, None
