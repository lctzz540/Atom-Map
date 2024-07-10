import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class DoubleGCNConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleGCNConv, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        self.conv2 = GCNConv(out_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return x


class GCNGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(GCNGenerator, self).__init__()

        self.enc_convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.dec_convs = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        in_channels = input_dim
        for hidden_dim in hidden_dims:
            self.enc_convs.append(DoubleGCNConv(in_channels, hidden_dim))
            self.pools.append(nn.Linear(hidden_dim, hidden_dim // 2))
            in_channels = hidden_dim // 2

        self.bottleneck_conv = DoubleGCNConv(hidden_dims[-1] // 2, hidden_dims[-1])

        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            self.upconvs.append(nn.Linear(hidden_dims[i], hidden_dims[i] // 2))
            self.dec_convs.append(DoubleGCNConv(hidden_dims[i], hidden_dims[i] // 2))

        self.output = nn.Linear(hidden_dims[-1] // 2, output_dim)

    def forward(self, x, edge_index):
        enc_outs = []

        for conv, pool in zip(self.enc_convs, self.pools):
            x = conv(x, edge_index)
            enc_outs.append(x)
            x = F.relu(pool(x))

        x = self.bottleneck_conv(x, edge_index)

        for upconv, conv in zip(self.upconvs, self.dec_convs):
            x = F.relu(upconv(x))
            x = torch.cat([x, enc_outs.pop()], dim=1)
            x = conv(x, edge_index)

        output = self.output(x)
        return torch.sigmoid(output)
