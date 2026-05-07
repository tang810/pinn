# src/network.py
import torch.nn as nn

class HighPrecisionPINN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=256, output_dim=6, n_layers=5):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]

        for i in range(n_layers - 2):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU() if i % 2 == 0 else nn.Tanh()
            ]

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)