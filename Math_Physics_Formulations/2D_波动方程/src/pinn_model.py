import torch
import torch.nn as nn
import numpy as np
from functools import reduce
from pyDOE import lhs

# 自动微分包装器
def deriv(u, x):
    return torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), create_graph=True, allow_unused=True
    )[0]

# PINN 主网络
class MLP(nn.Module):
    def __init__(self, in_features, out_features, num_layers=5, num_neurons=100, activation=torch.tanh):
        super(MLP, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.act_func = activation

        self.layer_input = nn.Linear(self.in_features, self.num_neurons)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(self.num_neurons, self.num_neurons) for _ in range(self.num_layers - 1)
        ])
        self.layer_output = nn.Linear(self.num_neurons, self.out_features)

    def forward(self, x):
        x_temp = self.act_func(self.layer_input(x))
        for layer in self.hidden_layers:
            x_temp = self.act_func(layer(x_temp))
        return self.layer_output(x_temp)

    def count_params(self):
        return sum(reduce(lambda x, y: x * y, p.size()) for p in self.parameters())

# Collocation Sampling Function (Latin Hypercube)
def lhs_sampling(N, lb, ub):
    return lb + (ub - lb) * lhs(len(lb), N)

# Loss Functions
class PINNLosses:
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device

    def pde_loss(self, X):
        x = X[:, 0:1]
        y = X[:, 1:2]
        t = X[:, 2:3]
        u = self.model(torch.cat([x, y, t], dim=1))

        u_x = deriv(u, x)
        u_xx = deriv(u_x, x)
        u_y = deriv(u, y)
        u_yy = deriv(u_y, y)
        u_t = deriv(u, t)
        u_tt = deriv(u_t, t)

        residual = u_tt - (u_xx + u_yy)
        return torch.mean(residual.pow(2))

    def boundary_loss(self, X_b):
        u = self.model(X_b)
        return torch.mean(u.pow(2))

    def initial_velocity_loss(self, X_i):
        x = X_i[:, 0:1]
        y = X_i[:, 1:2]
        t = X_i[:, 2:3]
        u = self.model(torch.cat([x, y, t], dim=1))
        u_t = deriv(u, t)
        return torch.mean(u_t.pow(2))

    def reconstruction_loss(self, X, Y):
        u = self.model(X)
        return torch.mean((u - Y) ** 2)

# Utility tensor converter with grad
def torch_tensor_grad(x, device="cpu"):
    tensor = torch.tensor(x, dtype=torch.float32, device=device, requires_grad=True)
    return tensor

def torch_tensor_nograd(x, device="cpu"):
    tensor = torch.tensor(x, dtype=torch.float32, device=device, requires_grad=False)
    return tensor
