import torch
from src.physics import pde_residual


def pde_loss(net_phi, net_logrho, x, A, B):
    res = pde_residual(net_phi, net_logrho, x, A, B)
    return torch.mean(res ** 2)


def data_loss(net_phi, M, N, d_obs):
    return (net_phi(M.unsqueeze(0)) -
            net_phi(N.unsqueeze(0)) - d_obs) ** 2
