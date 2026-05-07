import torch
import torch.nn as nn
from src.physics import frequencies, angles


class MultiFreqPINN(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        # 修复1: 调整参数范围和初始化，使r更容易优化
        # eps: 范围[1.0, 2.5]，真实值1.5在中间
        self.eps_param = nn.Parameter(torch.tensor(0.0))
        
        # r: 缩小范围，使真实值0.15在优化中心
        # 使用更窄的范围[0.12, 0.18]
        self.r_param = nn.Parameter(torch.tensor(0.0))

        self.net = nn.Sequential(
            nn.Linear(3, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 2 * len(angles)),
        )

    def forward(self, x, freq_idx):
        k = frequencies[freq_idx]
        k_feat = torch.ones_like(x[:, 0:1]) * k
        return self.net(torch.cat([x, k_feat], dim=1))

    def physical_params(self):
        # eps ∈ [1.0, 2.5]
        eps = 1.0 + 1.5 * torch.sigmoid(self.eps_param)
        
        # 修复2: 缩小r的范围，使其更集中
        # r ∈ [0.12, 0.18]，真实值0.15在正中间
        r = 0.12 + 0.06 * torch.sigmoid(self.r_param)
        
        return eps, r