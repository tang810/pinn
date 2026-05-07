
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class SnowfallPinn(nn.Module):
    """
    PINN for Snowfall Prediction.
    Input: (x, y, t)
    Output: Snow Depth S >= 0
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 60),
            nn.Tanh(),
            nn.Linear(60, 60),
            nn.Tanh(),
            nn.Linear(60, 60),
            nn.Tanh(),
            nn.Linear(60, 60),
            nn.Tanh(),
            nn.Linear(60, 1) # S >= 0
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.zeros_(m.bias)

    def forward(self, x):
        return F.softplus(self.net(x))

def build_model(device):
    """Constructs the Snowfall PINN model."""
    model = SnowfallPinn().to(device)
    return model
