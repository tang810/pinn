# src/data_utils.py
import random
from typing import Tuple

import numpy as np
import torch


def set_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str = "cuda") -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ==============================
# 采样函数
# ==============================

def sample_pde(
    N: int,
    Lx: float,
    Ly: float,
    T_final: float,
    bottom_focus_ratio: float,
    bottom_focus_y: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    一部分全域均匀，一部分集中在底部 bottom_focus_y 内。
    """
    N_bottom = int(N * bottom_focus_ratio)
    N_uniform = N - N_bottom

    # 全域均匀
    x_u = (torch.rand(N_uniform, 1, device=device) - 0.5) * Lx
    y_u = torch.rand(N_uniform, 1, device=device) * Ly
    t_u = torch.rand(N_uniform, 1, device=device) * T_final

    # 底部集中
    x_b = (torch.rand(N_bottom, 1, device=device) - 0.5) * Lx
    y_b = torch.rand(N_bottom, 1, device=device) * bottom_focus_y
    t_b = torch.rand(N_bottom, 1, device=device) * T_final

    x = torch.cat([x_u, x_b], dim=0)
    y = torch.cat([y_u, y_b], dim=0)
    t = torch.cat([t_u, t_b], dim=0)
    return x, y, t


def sample_ic(
    N: int,
    Lx: float,
    Ly: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = (torch.rand(N, 1, device=device) - 0.5) * Lx
    y = torch.rand(N, 1, device=device) * Ly
    t = torch.zeros_like(x)
    return x, y, t


def sample_bc_bottom(
    N: int,
    Lx: float,
    T_final: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_btm = (torch.rand(N, 1, device=device) - 0.5) * Lx
    y_btm = torch.zeros_like(x_btm)
    t_btm = torch.rand(N, 1, device=device) * T_final
    return x_btm, y_btm, t_btm


# ==============================
# 多材料：局部 E*, ρ* 分布
# ==============================

def material_factor(x: torch.Tensor, y: torch.Tensor, cfg: dict):
    """
    cfg['materials'] 内定义各模块 E_factor / rho_factor 及坐标范围（mm）。
    """
    mats = cfg.get("materials", {})
    camera = mats.get("camera", {})
    battery = mats.get("battery", {})
    side = mats.get("side_frame", {})

    x_mm = x * 1e3
    y_mm = y * 1e3

    E_factor = torch.ones_like(x)
    rho_factor = torch.ones_like(x)

    # 摄像头
    cam_region = camera.get("region_mm", {})
    if cam_region:
        cam_mask = (
            (x_mm >= cam_region.get("x0", -1e9))
            & (x_mm <= cam_region.get("x1", 1e9))
            & (y_mm >= cam_region.get("y0", -1e9))
            & (y_mm <= cam_region.get("y1", 1e9))
        )
        E_factor[cam_mask] = camera.get("E_factor", 1.0)
        rho_factor[cam_mask] = camera.get("rho_factor", 1.0)

    # 电池
    bat_region = battery.get("region_mm", {})
    if bat_region:
        bat_mask = (
            (x_mm >= bat_region.get("x0", -1e9))
            & (x_mm <= bat_region.get("x1", 1e9))
            & (y_mm >= bat_region.get("y0", -1e9))
            & (y_mm <= bat_region.get("y1", 1e9))
        )
        E_factor[bat_mask] = battery.get("E_factor", 1.0)
        rho_factor[bat_mask] = battery.get("rho_factor", 1.0)

    # 侧边框
    if side:
        y0 = side.get("y0_mm", 0.0)
        y1 = side.get("y1_mm", 140.0)
        left_x0 = side.get("left_x0_mm", -35.0)
        left_x1 = side.get("left_x1_mm", -30.0)
        right_x0 = side.get("right_x0_mm", 30.0)
        right_x1 = side.get("right_x1_mm", 35.0)

        frame_mask = (
            (
                (x_mm >= left_x0)
                & (x_mm <= left_x1)
                & (y_mm >= y0)
                & (y_mm <= y1)
            )
            | (
                (x_mm >= right_x0)
                & (x_mm <= right_x1)
                & (y_mm >= y0)
                & (y_mm <= y1)
            )
        )

        E_factor[frame_mask] = side.get("E_factor", 1.0)
        rho_factor[frame_mask] = side.get("rho_factor", 1.0)

    return E_factor, rho_factor


# ==============================
# 时间包络 & 等效体力
# ==============================

def impact_envelope(t: torch.Tensor, cfg: dict) -> torch.Tensor:
    impact_cfg = cfg.get("impact", {})
    use_env = impact_cfg.get("use_time_envelope", True)
    if not use_env:
        return torch.ones_like(t)

    t0 = impact_cfg.get("t0_env", 0.005)
    tau = impact_cfg.get("tau_env", 0.003)

    t0_t = torch.tensor(t0, dtype=t.dtype, device=t.device)
    env = torch.where(
        t <= t0_t,
        torch.sqrt(t / (t0_t + 1e-12)),
        torch.exp(-(t - t0_t) / (tau + 1e-12)),
    )
    return env


def body_force(
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    rho_local: torch.Tensor,
    cfg: dict,
):
    impact_cfg = cfg.get("impact", {})
    use_body_force = impact_cfg.get("use_body_force", True)
    peak = impact_cfg.get("body_force_peak", 40.0)
    if (not use_body_force) or peak == 0.0:
        fx = torch.zeros_like(x)
        fy = torch.zeros_like(y)
        return fx, fy

    t0 = torch.tensor(impact_cfg.get("pulse_t0", 0.0015), dtype=t.dtype, device=t.device)
    tau = torch.tensor(impact_cfg.get("pulse_tau", 0.0005), dtype=t.dtype, device=t.device)

    env_t = torch.exp(-((t - t0) / (tau + 1e-12)) ** 2)

    bottom_y = torch.tensor(
        impact_cfg.get("bottom_focus_y", 0.01), dtype=y.dtype, device=y.device
    )
    mask_bottom = (y <= bottom_y).float()
    spatial_decay = torch.exp(- (y / (bottom_y + 1e-12)) ** 2)

    a_y = peak * env_t * mask_bottom * spatial_decay
    fy = rho_local * a_y
    fx = torch.zeros_like(fy)
    return fx, fy
