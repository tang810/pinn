import torch
import numpy as np


# ---------------- Source term ----------------
def gaussian_source(x, src, eps=0.5):
    """
    Smooth approximation of delta source
    """
    return torch.exp(-((x - src) ** 2).sum(dim=1) / eps).unsqueeze(1)


# ---------------- PDE residual ----------------
def pde_residual(net_phi, net_logrho, x, A, B):
    """
    ∇·(σ∇φ) + s = 0
    Works for both 2D and 3D
    """
    phi = net_phi(x)
    rho = torch.exp(net_logrho(x))
    sigma = 1.0 / rho

    grad_phi = torch.autograd.grad(
        phi, x, torch.ones_like(phi), create_graph=True
    )[0]

    dim = x.shape[1]
    div = 0.0
    for i in range(dim):
        div += torch.autograd.grad(
            sigma * grad_phi[:, i:i+1],
            x,
            torch.ones_like(phi),
            create_graph=True
        )[0][:, i:i+1]

    src = gaussian_source(x, A) - gaussian_source(x, B)
    return div + src


# ---------------- Toy forward model (2D) ----------------
def phi_forward_toy_2d(x):
    """
    Artificial potential field (2D toy model)
    φ(x,z) = sin(pi x / Lx) * exp(-z / H)
    """
    return (
        torch.sin(np.pi * x[:, 0] / 50.0)
        * torch.exp(-x[:, 1] / 15.0)
    ).unsqueeze(1)


# ---------------- Toy forward model (3D) ----------------
def phi_forward_toy_3d(x):
    """
    Artificial potential field (3D toy model)
    φ(x,y,z) = sin(pi x / Lx) * sin(pi y / Ly) * exp(-z / H)
    """
    return (
        torch.sin(np.pi * x[:, 0] / 20.0)
        * torch.sin(np.pi * x[:, 1] / 20.0)
        * torch.exp(-x[:, 2] / 5.0)
    ).unsqueeze(1)
