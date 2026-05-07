# src/pinn_model.py
import torch
import torch.nn as nn


class PINN2D(nn.Module):
    """
    (x, y, t) -> (u_x, u_y)
    内部对 (x, y, t) 做 [-1, 1] 归一化。
    """

    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 2,
        width: int = 64,
        depth: int = 5,
        use_symmetry: bool = False,
        Lx: float = 0.07,
        Ly: float = 0.14,
        T_final: float = 0.02,
    ):
        super().__init__()
        self.use_symmetry = use_symmetry
        self.Lx = Lx
        self.Ly = Ly
        self.T_final = T_final

        layers = []
        layers.append(nn.Linear(in_dim, width))
        layers.append(nn.Tanh())
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, y, t):
        # x, y, t: (N, 1)
        if self.use_symmetry:
            x_feat = x ** 2
            x_n = x_feat / ((self.Lx / 2.0) ** 2)
        else:
            x_n = x / (self.Lx / 2.0)

        y_n = (y - self.Ly / 2.0) / (self.Ly / 2.0)
        t_n = (t - self.T_final / 2.0) / (self.T_final / 2.0)

        X = torch.cat([x_n, y_n, t_n], dim=1)
        return self.net(X)


def build_model(cfg: dict, device: torch.device) -> PINN2D:
    m_cfg = cfg.get("model", {})
    p_cfg = cfg.get("physics", {})

    model = PINN2D(
        in_dim=m_cfg.get("in_dim", 3),
        out_dim=m_cfg.get("out_dim", 2),
        width=m_cfg.get("width", 64),
        depth=m_cfg.get("depth", 5),
        use_symmetry=m_cfg.get("use_symmetry", False),
        Lx=p_cfg.get("Lx", 0.07),
        Ly=p_cfg.get("Ly", 0.14),
        T_final=p_cfg.get("T_final", 0.02),
    )
    return model.to(device)


def grads(y, x):
    """
    dy/dx, 其中 y, x 维度均为 (N, 1)
    """
    return torch.autograd.grad(
        y,
        x,
        torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]


def strain_stress(u, x, y, lam_base: float, mu_base: float, grads_fn=grads):
    """
    线弹性平面问题：
    输入位移 u = (u_x, u_y)，输出应力张量分量和 von Mises 相关量。
    """
    ux = u[:, 0:1]
    uy = u[:, 1:2]

    ux_x = grads_fn(ux, x)
    ux_y = grads_fn(ux, y)
    uy_x = grads_fn(uy, x)
    uy_y = grads_fn(uy, y)

    eps_xx = ux_x
    eps_yy = uy_y
    eps_xy = 0.5 * (ux_y + uy_x)

    eps_vol = eps_xx + eps_yy

    sig_xx = lam_base * eps_vol + 2 * mu_base * eps_xx
    sig_yy = lam_base * eps_vol + 2 * mu_base * eps_yy
    sig_xy = 2 * mu_base * eps_xy

    return sig_xx, sig_yy, sig_xy
