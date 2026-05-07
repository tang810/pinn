# -*- coding: utf-8 -*-
"""
真解 & 右端（用于合成数据/验证）以及基本算子
"""
import torch
from config import device, DTYPE, alpha_true

def u_true_xyz(xyz: torch.Tensor) -> torch.Tensor:
    """ u(x,y,z) = [sin(6x) sin(6y) sin(6z)]^3 """
    x = xyz[:, 0:1]
    y = xyz[:, 1:2]
    z = xyz[:, 2:3]
    s = torch.sin(6*x) * torch.sin(6*y) * torch.sin(6*z)
    return s**3

def laplacian(u: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
    """ 计算 Δu（3D） """
    grad_u = torch.autograd.grad(
        outputs=u, inputs=xyz,
        grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0]  # (N,3)
    ux, uy, uz = grad_u[:, 0:1], grad_u[:, 1:2], grad_u[:, 2:3]

    uxx = torch.autograd.grad(ux, xyz, torch.ones_like(ux), create_graph=True, retain_graph=True)[0][:, 0:1]
    uyy = torch.autograd.grad(uy, xyz, torch.ones_like(uy), create_graph=True, retain_graph=True)[0][:, 1:2]
    uzz = torch.autograd.grad(uz, xyz, torch.ones_like(uz), create_graph=True, retain_graph=True)[0][:, 2:3]
    return uxx + uyy + uzz

def f_true_xyz(xyz: torch.Tensor) -> torch.Tensor:
    """ f(x,y,z) = 0.01 * Δu + α * tanh(u) """
    xyz = xyz.clone().detach().to(device).requires_grad_(True)
    u   = u_true_xyz(xyz)
    lap = laplacian(u, xyz)
    f   = 0.01 * lap + alpha_true * torch.tanh(u)
    return f.detach()

def rand_points(N: int, lb: float, ub: float) -> torch.Tensor:
    # 统一使用 device+DTYPE，避免设备/dtype 冲突
    return lb + (ub - lb) * torch.rand(N, 3, dtype=DTYPE, device=device)
