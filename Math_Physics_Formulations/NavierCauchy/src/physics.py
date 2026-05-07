import torch

# -------------------- 物理项 --------------------
def body_force(x: torch.Tensor) -> torch.Tensor:
    b = torch.zeros_like(x)
    b[:, 1] = -0.1
    return b

def dirichlet_bc(x: torch.Tensor) -> torch.Tensor:
    u_bar = torch.zeros_like(x)
    mask_left  = (x[:, 0] < 1e-6)
    mask_right = (x[:, 0] > 1 - 1e-6)
    u_bar[mask_left, :] = 0.0
    u_bar[mask_right, 0] = 0.1
    return u_bar

def neumann_bc(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    t_bar = torch.zeros_like(x)
    mask_top = (x[:, 1] > 1 - 1e-6)
    t_bar[mask_top, 1] = 0.5
    return t_bar
