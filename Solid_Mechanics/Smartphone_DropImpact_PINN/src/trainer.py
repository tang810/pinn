# src/trainer.py
from pathlib import Path
from typing import Tuple

import torch

from .pinn_model import grads, strain_stress
from .data_utils import (
    sample_pde,
    sample_ic,
    sample_bc_bottom,
    material_factor,
    body_force,
)


def build_lame_parameters(cfg: dict):
    mats = cfg.get("materials", {})
    base = mats.get("base", {})
    E_star = base.get("E_star", 1.0)
    nu = cfg.get("physics", {}).get("nu", 0.30)

    lam_base = E_star * nu / ((1 + nu) * (1 - 2 * nu))
    mu_base = E_star / (2 * (1 + nu))
    return lam_base, mu_base


def compute_loss(
    model: torch.nn.Module,
    cfg: dict,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    p_cfg = cfg["physics"]
    t_cfg = cfg["training"]
    impact_cfg = cfg["impact"]

    Lx = p_cfg["Lx"]
    Ly = p_cfg["Ly"]
    T_final = p_cfg["T_final"]

    N_pde = t_cfg["N_pde"]
    N_ic = t_cfg["N_ic"]
    N_bc = t_cfg["N_bc"]

    bottom_focus_ratio = impact_cfg.get("bottom_focus_ratio", 0.5)
    bottom_focus_y = impact_cfg.get("bottom_focus_y", 0.01)

    v0_y = cfg.get("initial_condition", {}).get("v0_y", -2.0)
    IC_WEIGHT = t_cfg.get("ic_weight", 1.0)
    BC_WEIGHT = t_cfg.get("bc_weight", 1.0)

    lam_base, mu_base = build_lame_parameters(cfg)

    # PDE 点
    x_p, y_p, t_p = sample_pde(
        N_pde, Lx, Ly, T_final, bottom_focus_ratio, bottom_focus_y, device
    )
    x_p.requires_grad_(True)
    y_p.requires_grad_(True)
    t_p.requires_grad_(True)

    u_p = model(x_p, y_p, t_p)
    ux_p = u_p[:, 0:1]
    uy_p = u_p[:, 1:2]

    ux_t = grads(ux_p, t_p)
    uy_t = grads(uy_p, t_p)
    ux_tt = grads(ux_t, t_p)
    uy_tt = grads(uy_t, t_p)

    E_factor_p, rho_factor_p = material_factor(x_p, y_p, cfg)

    sig_xx, sig_yy, sig_xy = strain_stress(u_p, x_p, y_p, lam_base, mu_base)
    sig_xx = sig_xx * E_factor_p
    sig_yy = sig_yy * E_factor_p
    sig_xy = sig_xy * E_factor_p

    div_x = grads(sig_xx, x_p) + grads(sig_xy, y_p)
    div_y = grads(sig_xy, x_p) + grads(sig_yy, y_p)

    fx, fy = body_force(x_p, y_p, t_p, rho_factor_p, cfg)

    res_x = rho_factor_p * ux_tt - div_x - fx
    res_y = rho_factor_p * uy_tt - div_y - fy

    loss_pde = torch.mean(res_x**2 + res_y**2)

    # 初始条件
    x_i, y_i, t_i = sample_ic(N_ic, Lx, Ly, device)
    x_i.requires_grad_(True)
    y_i.requires_grad_(True)
    t_i.requires_grad_(True)

    u_i = model(x_i, y_i, t_i)
    ux_i = u_i[:, 0:1]
    uy_i = u_i[:, 1:2]
    ux_t_i = grads(ux_i, t_i)
    uy_t_i = grads(uy_i, t_i)

    loss_ic_u = torch.mean(u_i**2)
    loss_ic_v = torch.mean(ux_t_i**2 + (uy_t_i - v0_y) ** 2)
    loss_ic = loss_ic_u + loss_ic_v

    # 底边位移 0
    x_b, y_b, t_b = sample_bc_bottom(N_bc, Lx, T_final, device)
    x_b.requires_grad_(True)
    y_b.requires_grad_(True)
    t_b.requires_grad_(True)

    u_b = model(x_b, y_b, t_b)
    loss_bc_bottom = torch.mean(u_b**2)
    loss_bc = loss_bc_bottom

    loss = loss_pde + IC_WEIGHT * loss_ic + BC_WEIGHT * loss_bc
    return loss, loss_pde.detach(), loss_ic.detach(), loss_bc.detach()


def train_model(
    model: torch.nn.Module,
    cfg: dict,
    device: torch.device,
    mode: str,
    output_dir: Path,
    model_dir: Path,
):
    t_cfg = cfg["training"]

    if mode == "quick":
        epochs = t_cfg.get("epochs_quick", 2000)
    else:
        epochs = t_cfg.get("epochs_full", 7600)

    lr = t_cfg.get("lr", 1e-3)
    step_size = t_cfg.get("scheduler_step", 2000)
    gamma = t_cfg.get("scheduler_gamma", 0.5)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        loss, lp, li, lb = compute_loss(model, cfg, device)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 200 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch}/{epochs} "
                f"L={loss.item():.3e} "
                f"PDE={lp.item():.3e}, IC={li.item():.3e}, BC={lb.item():.3e} "
                f"LR={current_lr:.1e}"
            )

    ckpt_path = model_dir / "pinn_phone_drop.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"[INFO] model saved to {ckpt_path}")

    return model
