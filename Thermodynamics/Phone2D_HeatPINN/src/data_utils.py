# src/data_utils.py
import yaml
import torch
from pathlib import Path
from typing import Dict, Tuple

import torch
import numpy as np
import random

def set_random_seed(seed: int = 42):
    """统一设置随机种子，确保可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def load_case_config(path: str = "data/phone_thermal_case.yaml") -> Dict:
    """读取 YAML 配置"""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_physical_params(cfg: Dict) -> Dict:
    """根据 cfg 计算无量纲参数、Bi 数等"""
    rho = cfg["material"]["rho"]
    cp = cfg["material"]["cp"]
    k = cfg["material"]["k"]

    Lx = cfg["geometry"]["Lx"]
    Ly = cfg["geometry"]["Ly"]
    t_max = cfg["geometry"]["t_max"]

    T_inf = cfg["environment"]["T_inf"]
    h_env = cfg["environment"]["h_env"]

    theta_scale = cfg["nondim"]["theta_scale"]

    alpha = k / (rho * cp)

    Lx_scale = Lx
    Ly_scale = Ly
    t_scale = t_max

    alpha_x_hat = alpha * t_scale / (Lx_scale ** 2)
    alpha_y_hat = alpha * t_scale / (Ly_scale ** 2)

    Bi_x = h_env * Lx_scale / k
    Bi_y = h_env * Ly_scale / k

    q_to_theta = 1.0 / (rho * cp)

    return dict(
        rho=rho,
        cp=cp,
        k=k,
        alpha=alpha,
        Lx=Lx,
        Ly=Ly,
        t_max=t_max,
        T_inf=T_inf,
        h_env=h_env,
        theta_scale=theta_scale,
        Lx_scale=Lx_scale,
        Ly_scale=Ly_scale,
        t_scale=t_scale,
        alpha_x_hat=alpha_x_hat,
        alpha_y_hat=alpha_y_hat,
        Bi_x=Bi_x,
        Bi_y=Bi_y,
        q_to_theta=q_to_theta,
    )


def q_source_theta(x: torch.Tensor,
                   y: torch.Tensor,
                   t: torch.Tensor,
                   cfg: Dict,
                   params: Dict) -> torch.Tensor:
    """
    q̃(x,y,t) = q(x,y,t)/(ρcp)  [K/s]
    对应你原来的 q_source_theta，只是参数从 cfg / params 里取
    """
    soc = cfg["soc"]
    x_c = soc["x_center"]
    y_c = soc["y_center"]
    w_chip_x = soc["width"]
    w_chip_y = soc["height"]
    q0 = soc["q0"]
    tau_q = soc["tau_q"]
    use_slow = soc.get("use_slow_source", True)

    q_to_theta = params["q_to_theta"]

    cond_x = (torch.abs(x - x_c) <= w_chip_x / 2.0)
    cond_y = (torch.abs(y - y_c) <= w_chip_y / 2.0)
    mask = (cond_x & cond_y).float()

    if use_slow:
        factor_t = (1.0 - torch.exp(-t / tau_q))
    else:
        factor_t = torch.ones_like(t)

    q = q0 * mask * factor_t
    return q * q_to_theta  # [K/s]


def sample_pde_points(N_r: int, cfg: Dict, params: Dict) -> Tuple[torch.Tensor, ...]:
    """
    PDE collocation：
    60% 点在前 30% 时间，40% 在全时间；
    再对一半点强制落在 SoC 附近。
    """
    Lx = params["Lx"]
    Ly = params["Ly"]
    t_max = params["t_max"]

    soc = cfg["soc"]
    x_c = soc["x_center"]
    y_c = soc["y_center"]
    w_chip_x = soc["width"]
    w_chip_y = soc["height"]

    N_early = int(0.6 * N_r)
    N_full = N_r - N_early

    # early time [0, 0.3 t_max]
    x_e = torch.rand(N_early, 1) * Lx
    y_e = torch.rand(N_early, 1) * Ly
    t_e = torch.rand(N_early, 1) * (0.3 * t_max)

    # full [0, t_max]
    x_f = torch.rand(N_full, 1) * Lx
    y_f = torch.rand(N_full, 1) * Ly
    t_f = torch.rand(N_full, 1) * t_max

    x_r = torch.cat([x_e, x_f], dim=0)
    y_r = torch.cat([y_e, y_f], dim=0)
    t_r = torch.cat([t_e, t_f], dim=0)

    # 强化 SoC 区域
    N_half = N_r // 2
    idx = torch.randperm(N_r)[:N_half]

    margin_x = w_chip_x * 0.5
    margin_y = w_chip_y * 0.5
    x_soc = x_c + (torch.rand(N_half, 1) - 0.5) * (w_chip_x + 2 * margin_x)
    y_soc = y_c + (torch.rand(N_half, 1) - 0.5) * (w_chip_y + 2 * margin_y)
    x_soc = torch.clamp(x_soc, 0.0, Lx)
    y_soc = torch.clamp(y_soc, 0.0, Ly)

    x_r[idx, :] = x_soc
    y_r[idx, :] = y_soc

    return x_r, y_r, t_r


def sample_points(cfg: Dict, params: Dict):
    """
    返回：
    - PDE collocation 点
    - IC: t=0
    - IC2: t≈0 (可选)
    - 四条对流边界
    """
    sampling = cfg["sampling"]
    Lx = params["Lx"]
    Ly = params["Ly"]
    t_max = params["t_max"]

    N_r = sampling["N_r"]
    N_ic = sampling["N_ic"]
    N_ic2 = sampling["N_ic2"]
    N_bc_edge = sampling["N_bc_edge"]
    use_extra_t0 = sampling.get("use_extra_t0_samples", False)

    x_r, y_r, t_r = sample_pde_points(N_r, cfg, params)

    # IC: t=0
    x_ic = torch.rand(N_ic, 1) * Lx
    y_ic = torch.rand(N_ic, 1) * Ly
    t_ic = torch.zeros(N_ic, 1)

    # IC2: t ∈ [0, 0.02 t_max]（可选）
    if use_extra_t0 and N_ic2 > 0:
        x_ic2 = torch.rand(N_ic2, 1) * Lx
        y_ic2 = torch.rand(N_ic2, 1) * Ly
        t_ic2 = torch.rand(N_ic2, 1) * (0.02 * t_max)
    else:
        x_ic2 = torch.zeros(0, 1)
        y_ic2 = torch.zeros(0, 1)
        t_ic2 = torch.zeros(0, 1)

    # BC
    x_left = torch.zeros(N_bc_edge, 1)
    y_left = torch.rand(N_bc_edge, 1) * Ly
    t_left = torch.rand(N_bc_edge, 1) * t_max

    x_right = torch.full((N_bc_edge, 1), Lx)
    y_right = torch.rand(N_bc_edge, 1) * Ly
    t_right = torch.rand(N_bc_edge, 1) * t_max

    x_bottom = torch.rand(N_bc_edge, 1) * Lx
    y_bottom = torch.zeros(N_bc_edge, 1)
    t_bottom = torch.rand(N_bc_edge, 1) * t_max

    x_top = torch.rand(N_bc_edge, 1) * Lx
    y_top = torch.full((N_bc_edge, 1), Ly)
    t_top = torch.rand(N_bc_edge, 1) * t_max

    return (
        x_r, y_r, t_r,
        x_ic, y_ic, t_ic,
        x_ic2, y_ic2, t_ic2,
        x_left, y_left, t_left,
        x_right, y_right, t_right,
        x_bottom, y_bottom, t_bottom,
        x_top, y_top, t_top
    )
