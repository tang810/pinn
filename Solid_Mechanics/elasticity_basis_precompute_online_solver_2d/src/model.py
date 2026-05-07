import torch
import torch.nn as nn


class ElasticPINN(nn.Module):
    def __init__(self, in_dim=3, hidden=64, depth=3, out_dim=2):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)