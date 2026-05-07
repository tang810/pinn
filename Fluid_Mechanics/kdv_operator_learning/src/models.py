import torch
import torch.nn as nn

class KdV_PINN(nn.Module):
    def __init__(self, n_input=1, n_output=1, n_hidden=50, n_layers=3):
        super().__init__()
        layers = [nn.Linear(n_input, n_hidden), nn.Tanh()]
        for _ in range(n_layers-1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.Tanh()]
        layers += [nn.Linear(n_hidden, n_output)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
