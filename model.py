import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv


class DoubleGINConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleGINConv, self).__init__()
        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU()
        ))
        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU()
        ))

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return x


class GINGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(GINGenerator, self).__init__()

        self.enc_convs = nn.ModuleList()
        self.dec_convs = nn.ModuleList()

        in_channels = input_dim
        for hidden_dim in hidden_dims:
            self.enc_convs.append(DoubleGINConv(in_channels, hidden_dim))
            in_channels = hidden_dim

        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            self.dec_convs.append(DoubleGINConv(hidden_dims[i], hidden_dims[i + 1]))

        self.output = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x, edge_index):
        enc_outs = []

        for conv in self.enc_convs:
            x = conv(x, edge_index)
            enc_outs.append(x)

        for conv in self.dec_convs:
            x = conv(x, edge_index)

        output = self.output(x)
        return torch.sigmoid(output)
