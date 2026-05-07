# src/pinn_model.py
import torch
import torch.nn as nn
from typing import Tuple, Dict


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HeatPINN(nn.Module):
    def __init__(self, in_dim: int = 3, hidden_dim: int = 64, n_layers: int = 5):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, xyt_hat: torch.Tensor) -> torch.Tensor:
        return self.net(xyt_hat)


def build_model(cfg: Dict) -> Tuple[HeatPINN, torch.optim.Optimizer, torch.device]:
    device = get_device()
    model = HeatPINN().to(device)
    lr = cfg["training"]["lr"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimizer, device
