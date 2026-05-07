import torch
import torch.nn as nn


class FNN(nn.Module):
    def __init__(self, num_layers=5, num_nodes=50):
        super().__init__()
        layers = []
        input_dim = 2
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, num_nodes))
            layers.append(nn.Tanh())
            input_dim = num_nodes
        layers.append(nn.Linear(num_nodes, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
