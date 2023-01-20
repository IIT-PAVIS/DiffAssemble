import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.graphgym.register import register_layer
from torch_geometric.nn.conv.transformer_conv import TransformerConv
from torch_scatter import scatter

def get_activation(activation):
    if activation == "relu":
        return 2, nn.ReLU()
    elif activation == "gelu":
        return 2, nn.GELU()
    elif activation == "silu":
        return 2, nn.SiLU()
    elif activation == "glu":
        return 1, nn.GLU()
    else:
        raise ValueError(f"activation function {activation} is not valid!")


class ExphormerFullLayer(nn.Module):
    """Exphormer attention + FFN"""

    def __init__(
        self,
        in_dim,
        out_dim,
        num_heads,
        dropout=0.0,
        dim_edge=None,
        layer_norm=False,
        batch_norm=True,
        activation="gelu",
        residual=False,
        use_bias=False,
        use_virt_nodes=False,
    ):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.attention = ExphormerAttention(
            in_dim,
            out_dim,
            num_heads,
            use_bias=use_bias,
            dim_edge=dim_edge,
            use_virt_nodes=use_virt_nodes,
        )

        self.O_h = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)

        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        factor, self.activation_fn = get_activation(activation=activation)
        self.FFN_h_layer2 = nn.Linear(out_dim * factor, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)

    def forward(
        self, x, batch, edge_index, edge_attr=None, return_attention_weights=False
    ):
        h = x
        h_in1 = h  # for first residual connection
        # multi-head attention out

        h_attn_out = self.attention(
            x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch
        )

        # Concat multi-head outputs
        h = h_attn_out.view(-1, self.out_channels)

        h = F.dropout(h, self.dropout, training=self.training)

        # h = self.O_h(h)

        if self.residual:
            h = h_in1 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm1_h(h)

        if self.batch_norm:
            h = self.batch_norm1_h(h)

        h_in2 = h  # for second residual connection

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = self.activation_fn(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            h = h_in2 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm2_h(h)

        if self.batch_norm:
            h = self.batch_norm2_h(h)

        return h, h_attn_out

    def __repr__(self):
        return "{}(in_channels={}, out_channels={}, heads={}, residual={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.num_heads,
            self.residual,
        )


class Exophormer_GNN(nn.Module):
    def __init__(
        self, input_size, hidden_dim, heads, output_size, n_layers=4, virt_nodes=4
    ) -> None:
        super().__init__()

        self.module_list = nn.ModuleList(
            [TransformerConv(input_size, out_channels=hidden_dim // heads, heads=heads)]
            + [
                TransformerConv(
                    hidden_dim, out_channels=hidden_dim // heads, heads=heads
                )
                for _ in range(n_layers - 2)
            ]
            + [
                TransformerConv(
                    hidden_dim,
                    out_channels=output_size // heads,
                    heads=heads,
                    concat=True,
                ),
            ]
        )

        self.virt_nodes = virt_nodes
        if self.virt_nodes > 0:
            self.virt_node_embedding = nn.Embedding(virt_nodes, input_size)
        self.n_layers = n_layers

    def forward(self, x, edge_index, move_to_cpu=False, batch=None, mean_value=False, *args):
        attentions = []

        n_graphs = batch.max() + 1
        num_real_nodes = len(batch)
        device = batch.device
        if self.virt_nodes > 0:
            # adding virtual nodes
            virtual_nodes = torch.arange(self.virt_nodes).repeat(n_graphs).to(device)
            #breakpoint()
            if mean_value:
                virt_nodes_h = x.mean(dim=0).unsqueeze_(0).repeat(self.virt_nodes*n_graphs, 1)
                #print(virt_nodes_h.size())
            else:
                virt_nodes_h = self.virt_node_embedding(virtual_nodes)
                #print(virt_nodes_h.size())
            #print(x.size())
            x = torch.cat((x, virt_nodes_h))
           # print(x.size())
            batch = torch.cat(
                (batch, torch.arange(n_graphs).repeat(self.virt_nodes).to(device)) # type: ignore
            )
            virt_edges = []

            for i in batch.unique():
                num_nodes = len(batch[batch == i])
                virt_edge = (
                    torch.arange(
                        num_real_nodes + i * self.virt_nodes,
                        num_real_nodes + (i + 1) * self.virt_nodes,
                    )
                    .repeat(num_nodes)
                    .to(device)
            )
                virt_edges.append(virt_edge)

            virt_edges = torch.cat(virt_edges)
            src_edges = torch.cat([torch.arange(num_real_nodes).to(device), virt_edges])
            dst_edges = torch.cat([virt_edges, torch.arange(num_real_nodes).to(device)])
            edge_index = torch.hstack((edge_index, torch.stack((src_edges, dst_edges))))

        for i in range(self.n_layers - 1):
            x = self.module_list[i](x=x, edge_index=edge_index)

        x, atts = self.module_list[-1](
            x=x, edge_index=edge_index, return_attention_weights=True
        )
        if self.virt_nodes > 0:
            x = x[:num_real_nodes]  # remove virtual nodes
        attentions.append(atts)

        if move_to_cpu:
            attentions = [(a[0].cpu().numpy(), a[1].cpu().numpy()) for a in attentions]
            x = x.cpu()
        return x, attentions
