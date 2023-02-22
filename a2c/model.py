import torch
import torch.nn as nn
import torch.optim as optim
from network import EdgeBatchNorm, MLP


class Actor(nn.Module):
    def __init__(self,
                 lr,
                 device):
        super().__init__()
        self.edge_bn = EdgeBatchNorm(dim=6,
                                     device=device)
        self.mlp = MLP(layer_dim=[6, 64, 8, 1],
                              device=device)
        self.optimizer = optim.SGD(self.parameters(), lr=lr, weight_decay=5e-4)

    def forward(self, state):
        out = self.edge_bn(state)
        out = self.mlp(out)

        return out

    def valid_choice(self, state, mask):
        out = self.forward(state)
        valid_choice = torch.take(out.cpu(), torch.LongTensor(mask))

        return valid_choice


class Critic(nn.Module):
    def __init__(self,
                 lr,
                 device):
        super().__init__()
        layer_dim = [120, 64, 4, 1]
        self.edge_bn = EdgeBatchNorm(dim=6,
                                     device=device)
        self.hid_layer_1 = nn.Linear(layer_dim[0], layer_dim[1])
        self.hid_layer_2 = nn.Linear(layer_dim[1], layer_dim[2])
        self.hid_layer_3 = nn.Linear(layer_dim[2], layer_dim[3])
        self.optimizer = optim.SGD(self.parameters(), lr=lr, weight_decay=5e-4)
        self.to(device)

    def forward(self, state):
        out = self.edge_bn(state).reshape(-1).detach()
        out = torch.tanh(self.hid_layer_1(out))
        out = torch.tanh(self.hid_layer_2(out))
        out = self.hid_layer_3(out)

        return out