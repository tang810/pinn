# -*- coding: utf-8 -*-
"""
数据构建：训练集（带噪 u/f）与验证网格（真值）
"""
import torch
from config import device, DTYPE, like_std_u, like_std_f
from truth import u_true_xyz, laplacian, rand_points

def build_train_data(N_tr_u: int, N_tr_f: int, lb: float, ub: float):
    # 训练数据：u 点
    with torch.no_grad():
        x_u = rand_points(N_tr_u, lb, ub)
        y_u = u_true_xyz(x_u) + like_std_u * torch.randn(N_tr_u, 1, dtype=DTYPE, device=device)

    # 训练数据：f 点（需要用 autograd 得到 Δu）
    x_f = rand_points(N_tr_f, lb, ub).requires_grad_(True)
    with torch.enable_grad():
        u_clean = u_true_xyz(x_f)
        lap     = laplacian(u_clean, x_f)
        f_clean = 0.01 * lap + torch.tanh(u_clean)  # α_true=1.0 在 config 中定义；合成数据中已包含

    y_f = (f_clean.detach() + like_std_f * torch.randn(N_tr_f, 1, dtype=DTYPE, device=device))

    data = {'x_u': x_u, 'y_u': y_u, 'x_f': x_f.detach(), 'y_f': y_f}
    return data

def build_val_data(N_val_1d: int, lb: float, ub: float):
    from truth import u_true_xyz  # 局部导入，避免循环依赖
    from config import alpha_true

    grid_1d = torch.linspace(lb, ub, N_val_1d, dtype=DTYPE, device=device)
    X, Y, Z = torch.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij')
    xyz_val = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=1)

    with torch.no_grad():
        u_val = u_true_xyz(xyz_val)
    with torch.enable_grad():
        from truth import laplacian  # 避免顶层开销
        xyz_val_req = xyz_val.clone().detach().requires_grad_(True)
        u_tmp       = u_true_xyz(xyz_val_req)
        lap_val     = laplacian(u_tmp, xyz_val_req)
        f_val       = (0.01 * lap_val + alpha_true * torch.tanh(u_tmp)).detach()

    data_val = {'x_u': xyz_val, 'y_u': u_val, 'x_f': xyz_val, 'y_f': f_val}
    return data_val
