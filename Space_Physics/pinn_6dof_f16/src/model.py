import torch
import torch.nn as nn

class AeroNet(nn.Module):
    """输入 9 维解释变量，输出 6 个气动系数。"""
    def __init__(self, in_dim: int = 9, hidden: int = 64, layers: int = 4):
        super().__init__()
        seq = [nn.Linear(in_dim, hidden), nn.Softplus()]
        for _ in range(layers - 1):
            seq += [nn.Linear(hidden, hidden), nn.Softplus()]
        seq += [nn.Linear(hidden, 6)]
        self.net = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
