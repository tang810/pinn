# -*- coding: utf-8 -*-
"""
模型（MLP：命名线性层，避免与 fmodule 冲突）
"""
import torch
import torch.nn as nn
from config import device, DTYPE, layer_sizes, activation

class Net3D(nn.Module):
    def __init__(self, layer_sizes, activation=torch.tanh):
        super().__init__()
        d0, d1, d2, d3, d4 = layer_sizes
        self.l1 = nn.Linear(d0, d1)
        self.l2 = nn.Linear(d1, d2)
        self.l3 = nn.Linear(d2, d3)
        self.l4 = nn.Linear(d3, d4)
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.l1(x))
        x = self.activation(self.l2(x))
        x = self.activation(self.l3(x))
        x = self.l4(x)
        return x

def build_nets():
    net_u = Net3D(layer_sizes, activation).to(device).to(DTYPE)
    nets  = [net_u]
    return nets
