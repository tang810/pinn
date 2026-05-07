# src/trainer.py
import torch
from pathlib import Path
from typing import Dict, Tuple

from .data_utils import sample_points, q_source_theta
from .pinn_model import build_model


def compute_loss(model: torch.nn.Module,
                 device: torch.device,
                 batch,
                 cfg: Dict,
                 params: Dict):
    """
    对应你原来的 compute_loss，只是参数拆成 cfg / params / model / device
    """
    (
        x_r, y_r, t_r,
        x_ic, y_ic, t_ic,
        x_ic2, y_ic2, t_ic2,
        x_left, y_left, t_left,
        x_right, y_right, t_right,
        x_bottom, y_bottom, t_bottom,
        x_top, y_top, t_top
    ) = batch

    # 移到 device
    x_r = x_r.to(device);   y_r = y_r.to(device);   t_r = t_r.to(device)
    x_ic = x_ic.to(device); y_ic = y_ic.to(device); t_ic = t_ic.to(device)
    x_ic2 = x_ic2.to(device); y_ic2 = y_ic2.to(device); t_ic2 = t_ic2.to(device)
    x_left = x_left.to(device);   y_left = y_left.to(device);   t_left = t_left.to(device)
    x_right = x_right.to(device); y_right = y_right.to(device); t_right = t_right.to(device)
    x_bottom = x_bottom.to(device); y_bottom = y_bottom.to(device); t_bottom = t_bottom.to(device)
    x_top = x_top.to(device);     y_top = y_top.to(device);     t_top = t_top.to(device)

    Lx_scale = params["Lx_scale"]
    Ly_scale = params["Ly_scale"]
    t_scale = params["t_scale"]
    theta_scale = params["theta_scale"]
    alpha_x_hat = params["alpha_x_hat"]
    alpha_y_hat = params["alpha_y_hat"]
    Bi_x = params["Bi_x"]
    Bi_y = params["Bi_y"]

    # ========= PDE，无量纲 =========
    xh_r = (x_r / Lx_scale).clone().detach().requires_grad_(True)
    yh_r = (y_r / Ly_scale).clone().detach().requires_grad_(True)
    th_r = (t_r / t_scale).clone().detach().requires_grad_(True)

    theta_hat_r = model(torch.cat([xh_r, yh_r, th_r], dim=1))

    dtheta_hat_dt_hat = torch.autograd.grad(
        theta_hat_r, th_r,
        grad_outputs=torch.ones_like(theta_hat_r),
        retain_graph=True,
        create_graph=True
    )[0]
    dtheta_hat_dx_hat = torch.autograd.grad(
        theta_hat_r, xh_r,
        grad_outputs=torch.ones_like(theta_hat_r),
        retain_graph=True,
        create_graph=True
    )[0]
    dtheta_hat_dy_hat = torch.autograd.grad(
        theta_hat_r, yh_r,
        grad_outputs=torch.ones_like(theta_hat_r),
        retain_graph=True,
        create_graph=True
    )[0]

    d2theta_hat_dx2_hat = torch.autograd.grad(
        dtheta_hat_dx_hat, xh_r,
        grad_outputs=torch.ones_like(dtheta_hat_dx_hat),
        retain_graph=True,
        create_graph=True
    )[0]
    d2theta_hat_dy2_hat = torch.autograd.grad(
        dtheta_hat_dy_hat, yh_r,
        grad_outputs=torch.ones_like(dtheta_hat_dy_hat),
        retain_graph=True,
        create_graph=True
    )[0]

    # q(x,y,t)
    q_tilde = q_source_theta(x_r, y_r, t_r, cfg, params)          # K/s
    q_hat = q_tilde * (t_scale / theta_scale)                     # 无量纲

    res_pde = dtheta_hat_dt_hat - alpha_x_hat * d2theta_hat_dx2_hat \
              - alpha_y_hat * d2theta_hat_dy2_hat - q_hat
    loss_pde = torch.mean(res_pde ** 2)

    # ========= IC: t=0 =========
    xh_ic = x_ic / Lx_scale
    yh_ic = y_ic / Ly_scale
    th_ic = t_ic / t_scale

    theta_hat_ic = model(torch.cat([xh_ic, yh_ic, th_ic], dim=1))
    loss_ic = torch.mean(theta_hat_ic ** 2)

    # ========= IC2: t≈0，可选 =========
    if x_ic2.numel() > 0:
        xh_ic2 = x_ic2 / Lx_scale
        yh_ic2 = y_ic2 / Ly_scale
        th_ic2 = t_ic2 / t_scale
        theta_hat_ic2 = model(torch.cat([xh_ic2, yh_ic2, th_ic2], dim=1))
        loss_ic2 = torch.mean(theta_hat_ic2 ** 2)
    else:
        loss_ic2 = torch.tensor(0.0, device=device)

    # ========= BC: 对流 =========
    # x = 0
    xh_left = (x_left / Lx_scale).clone().detach().requires_grad_(True)
    yh_left = (y_left / Ly_scale)
    th_left = (t_left / t_scale)
    theta_hat_left = model(torch.cat([xh_left, yh_left, th_left], dim=1))
    dtheta_hat_dx_hat_left = torch.autograd.grad(
        theta_hat_left, xh_left,
        grad_outputs=torch.ones_like(theta_hat_left),
        retain_graph=True,
        create_graph=True
    )[0]
    res_left = dtheta_hat_dx_hat_left - Bi_x * theta_hat_left
    loss_left = torch.mean(res_left ** 2)

    # x = Lx
    xh_right = (x_right / Lx_scale).clone().detach().requires_grad_(True)
    yh_right = (y_right / Ly_scale)
    th_right = (t_right / t_scale)
    theta_hat_right = model(torch.cat([xh_right, yh_right, th_right], dim=1))
    dtheta_hat_dx_hat_right = torch.autograd.grad(
        theta_hat_right, xh_right,
        grad_outputs=torch.ones_like(theta_hat_right),
        retain_graph=True,
        create_graph=True
    )[0]
    res_right = -dtheta_hat_dx_hat_right - Bi_x * theta_hat_right
    loss_right = torch.mean(res_right ** 2)

    # y = 0
    xh_bottom = (x_bottom / Lx_scale)
    yh_bottom = (y_bottom / Ly_scale).clone().detach().requires_grad_(True)
    th_bottom = (t_bottom / t_scale)
    theta_hat_bottom = model(torch.cat([xh_bottom, yh_bottom, th_bottom], dim=1))
    dtheta_hat_dy_hat_bottom = torch.autograd.grad(
        theta_hat_bottom, yh_bottom,
        grad_outputs=torch.ones_like(theta_hat_bottom),
        retain_graph=True,
        create_graph=True
    )[0]
    res_bottom = dtheta_hat_dy_hat_bottom - Bi_y * theta_hat_bottom
    loss_bottom = torch.mean(res_bottom ** 2)

    # y = Ly
    xh_top = (x_top / Lx_scale)
    yh_top = (y_top / Ly_scale).clone().detach().requires_grad_(True)
    th_top = (t_top / t_scale)
    theta_hat_top = model(torch.cat([xh_top, yh_top, th_top], dim=1))
    dtheta_hat_dy_hat_top = torch.autograd.grad(
        theta_hat_top, yh_top,
        grad_outputs=torch.ones_like(theta_hat_top),
        retain_graph=True,
        create_graph=True
    )[0]
    res_top = -dtheta_hat_dy_hat_top - Bi_y * theta_hat_top
    loss_top = torch.mean(res_top ** 2)

    loss_bc_all = loss_left + loss_right + loss_bottom + loss_top

    # ========= 总 loss =========
    tr_cfg = cfg["training"]
    w_pde = tr_cfg["w_pde"]
    w_ic = tr_cfg["w_ic"]
    w_ic2 = tr_cfg["w_ic2"]
    w_bc = tr_cfg["w_bc"]

    loss_total = (
        w_pde * loss_pde +
        w_ic * loss_ic +
        w_ic2 * loss_ic2 +
        w_bc * loss_bc_all
    )

    loss_dict = {
        "pde": loss_pde.item(),
        "ic": loss_ic.item(),
        "ic2": loss_ic2.item(),
        "bc": loss_bc_all.item()
    }
    return loss_total, loss_dict


def train(cfg: Dict,
          params: Dict,
          model_dir: str = "model",
          n_iters_override: int = None,
          print_every_override: int = None) -> Tuple[torch.nn.Module, torch.device]:
    """训练主循环，返回 model 和 device"""
    model, optimizer, device = build_model(cfg)
    batch = sample_points(cfg, params)

    n_iters = n_iters_override or cfg["training"]["n_iters"]
    print_every = print_every_override or cfg["training"]["print_every"]

    for it in range(1, n_iters + 1):
        optimizer.zero_grad()
        loss, loss_dict = compute_loss(model, device, batch, cfg, params)
        loss.backward()
        optimizer.step()

        if it % print_every == 0 or it == 1:
            print(
                f"Iter {it}/{n_iters} | "
                f"loss={loss.item():.3e} | "
                f"pde={loss_dict['pde']:.3e} | "
                f"ic={loss_dict['ic']:.3e} | "
                f"ic2={loss_dict['ic2']:.3e} | "
                f"bc={loss_dict['bc']:.3e}"
            )

    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)
    save_path = model_dir_path / "pinn_phone_thermal.pt"
    torch.save(model.state_dict(), save_path)
    print("Training done, model saved to:", save_path)

    return model, device
