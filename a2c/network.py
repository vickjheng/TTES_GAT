import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeBatchNorm(nn.Module):
    def __init__(self,
                 dim,
                 device):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(dim)
        self.to(device)

    def forward(self, x):
        out = x.transpose(1, 2).contiguous()
        out = self.batch_norm(out).transpose(1, 2).contiguous()

        return out


class MLP(nn.Module):
    def __init__(self,
                 layer_dim,
                 device):
        super().__init__()
        self.hid_layer_1 = nn.Linear(layer_dim[0], layer_dim[1])
        self.hid_layer_2 = nn.Linear(layer_dim[1], layer_dim[2])
        self.hid_layer_3 = nn.Linear(layer_dim[2], layer_dim[3])
        self.to(device)

    def forward(self, state):
        # out = F.relu(self.hid_layer_1(state))
        # out = F.relu(self.hid_layer_2(out))
        out = torch.tanh(self.hid_layer_1(state))
        out = torch.tanh(self.hid_layer_2(out))
        out = self.hid_layer_3(out).squeeze()

        return out