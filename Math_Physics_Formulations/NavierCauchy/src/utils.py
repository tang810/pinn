import os

import numpy as np
import torch
from scipy.stats import qmc

from .physics import dirichlet_bc, neumann_bc


d = 2


def latin_hypercube_sampling(n: int, dim: int, bounds=None):
    if bounds is None:
        bounds = np.array([[0, 1]] * dim)
    sampler = qmc.LatinHypercube(d=dim)
    sample = sampler.random(n)
    lb = bounds[:, 0]
    ub = bounds[:, 1]
    sample = lb + (ub - lb) * sample
    return torch.tensor(sample, dtype=torch.float32)


def generate_boundary_points(n: int, d_dim: int = 2):
    n_per_side = max(1, n // 4)
    x_left = torch.zeros(n_per_side, d_dim)
    x_left[:, 1] = torch.linspace(0, 1, n_per_side)

    x_right = torch.ones(n_per_side, d_dim)
    x_right[:, 1] = torch.linspace(0, 1, n_per_side)

    y_bottom = torch.zeros(n_per_side, d_dim)
    y_bottom[:, 0] = torch.linspace(0, 1, n_per_side)

    y_top = torch.zeros(n_per_side, d_dim)
    y_top[:, 0] = torch.linspace(0, 1, n_per_side)
    y_top[:, 1] = 1.0

    x_boundary = torch.cat([x_left, x_right, y_bottom, y_top], dim=0)

    n_boundary = torch.zeros_like(x_boundary)
    n_boundary[:n_per_side, 0] = -1.0
    n_boundary[n_per_side : 2 * n_per_side, 0] = 1.0
    n_boundary[2 * n_per_side : 3 * n_per_side, 1] = -1.0
    n_boundary[3 * n_per_side :, 1] = 1.0

    return x_boundary, n_boundary


def generate_and_save_data(x_pde_n=200, x_boundary_n=240, data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)

    x_pde = latin_hypercube_sampling(x_pde_n, d)
    np.save(os.path.join(data_dir, "x_pde.npy"), x_pde.numpy())

    x_dir, _ = generate_boundary_points(x_boundary_n, d)
    np.save(os.path.join(data_dir, "x_dir.npy"), x_dir.numpy())
    u_dir = dirichlet_bc(x_dir)
    np.save(os.path.join(data_dir, "u_dir.npy"), u_dir.numpy())

    x_neu, n_neu = generate_boundary_points(x_boundary_n, d)
    np.save(os.path.join(data_dir, "x_neu.npy"), x_neu.numpy())
    np.save(os.path.join(data_dir, "n_neu.npy"), n_neu.numpy())
    t_neu = neumann_bc(x_neu, n_neu)
    np.save(os.path.join(data_dir, "t_neu.npy"), t_neu.numpy())

    print("[data] Generated and saved data files")
    return x_pde, x_dir, u_dir, x_neu, n_neu, t_neu


def load_data(data_dir="data"):
    try:
        x_pde = torch.tensor(np.load(os.path.join(data_dir, "x_pde.npy")), dtype=torch.float32)
        x_dir = torch.tensor(np.load(os.path.join(data_dir, "x_dir.npy")), dtype=torch.float32)
        u_dir = torch.tensor(np.load(os.path.join(data_dir, "u_dir.npy")), dtype=torch.float32)
        x_neu = torch.tensor(np.load(os.path.join(data_dir, "x_neu.npy")), dtype=torch.float32)
        n_neu = torch.tensor(np.load(os.path.join(data_dir, "n_neu.npy")), dtype=torch.float32)
        t_neu = torch.tensor(np.load(os.path.join(data_dir, "t_neu.npy")), dtype=torch.float32)
        return x_pde, x_dir, u_dir, x_neu, n_neu, t_neu
    except Exception:
        print("[data] Files not exist, regenerating...")
        return generate_and_save_data(data_dir=data_dir)