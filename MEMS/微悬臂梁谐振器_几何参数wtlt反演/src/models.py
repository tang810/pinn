

# ================================
# File: src/models.py
# ================================
import torch
import torch.nn as nn

class PINNInverseMLP(nn.Module):
    def __init__(self, input_dim=8, output_dim=2, hidden_dim=128, hidden_layers=4, dropout=0.0):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())  # 强制输出在 0 ~ 1
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)