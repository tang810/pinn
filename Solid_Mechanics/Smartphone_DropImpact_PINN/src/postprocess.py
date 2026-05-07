# src/postprocess.py
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, BoxStyle
import numpy as np
import torch

from .pinn_model import strain_stress, grads
from .data_utils import material_factor, impact_envelope


def evaluate_stress_field(
    model: torch.nn.Module,
    cfg: dict,
    device: torch.device,
    t_val: float,
    nx: int = 120,
    ny: int = 240,):
    p_cfg = cfg["physics"]
    Lx = p_cfg["Lx"]
    Ly = p_cfg["Ly"]

    x = torch.linspace(-Lx / 2, Lx / 2, nx, device=device)
    y = torch.linspace(0.0, Ly, ny, device=device)
    X, Y = torch.meshgrid(x, y, indexing="xy")

    x_in = X.reshape(-1, 1)
    y_in = Y.reshape(-1, 1)
    t_in = torch.ones_like(x_in) * t_val

    x_in.requires_grad_(True)
    y_in.requires_grad_(True)
    t_in.requires_grad_(True)

    mats = cfg.get("materials", {})
    base = mats.get("base", {})
    E_star = base.get("E_star", 1.0)
    nu = p_cfg.get("nu", 0.30)
    lam_base = E_star * nu / ((1 + nu) * (1 - 2 * nu))
    mu_base = E_star / (2 * (1 + nu))

    with torch.no_grad():
        u = model(x_in, y_in, t_in)

    # 对应 material_factor / strain_stress 需要梯度，重新开一遍带 grad 的版本
    x_in.requires_grad_(True)
    y_in.requires_grad_(True)
    t_in.requires_grad_(True)
    u = model(x_in, y_in, t_in)

    E_factor, rho_factor = material_factor(x_in, y_in, cfg)

    sig_xx, sig_yy, sig_xy = strain_stress(u, x_in, y_in, lam_base, mu_base)
    sig_xx = sig_xx * E_factor
    sig_yy = sig_yy * E_factor
    sig_xy = sig_xy * E_factor

    Svm_star = torch.sqrt(
        sig_xx**2 + sig_yy**2 - sig_xx * sig_yy + 3 * sig_xy**2 + 1e-12
    )
    Svm_star = Svm_star.reshape(X.shape)

    # 时间包络整体缩放
    t_tensor = torch.tensor([t_val], dtype=torch.float32, device=device).view(-1, 1)
    env_t = impact_envelope(t_tensor, cfg)[0, 0].item()
    Svm_star_t = Svm_star * env_t

    return (
        X.detach().cpu().numpy(),
        Y.detach().cpu().numpy(),
        Svm_star_t.detach().cpu().numpy(),
    )


def to_physical_stress(cfg: dict, Svm_star: np.ndarray) -> np.ndarray:
    sigma_ref = cfg.get("physics", {}).get("E_phys0", 7.0e10)
    Svm_phys = Svm_star * sigma_ref
    Svm_MPa = Svm_phys / 1e6
    return Svm_MPa


def plot_stress_phone(
    cfg: dict,
    X: np.ndarray,
    Y: np.ndarray,
    S_norm: np.ndarray,
    t_val: float,
    savepath: Path = None,
    show: bool = False,):
    X_mm = X * 1000.0
    Y_mm = Y * 1000.0

    fig, ax = plt.subplots(figsize=(5.5, 10.0), dpi=220)
    ax.set_facecolor("white")

    x_min, x_max = X_mm.min(), X_mm.max()
    y_min, y_max = Y_mm.min(), Y_mm.max()
    phone_width = x_max - x_min
    phone_height = y_max - y_min
    corner_radius = 12.0

    im = ax.pcolormesh(
        X_mm,
        Y_mm,
        S_norm,
        shading="auto",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        zorder=5,
    )
    ax.set_aspect("equal")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    for spine in ax.spines.values():
        spine.set_visible(False)

    clip_pad = 1.0
    clip_box = FancyBboxPatch(
        (x_min - clip_pad, y_min - clip_pad),
        phone_width + 2 * clip_pad,
        phone_height + 2 * clip_pad,
        boxstyle=BoxStyle("Round", pad=0.0, rounding_size=corner_radius + 1.0),
        linewidth=0.0,
        edgecolor="none",
        facecolor="none",
        zorder=4,
    )
    ax.add_patch(clip_box)
    im.set_clip_path(clip_box)

    # 外框 & 屏幕框
    PHONE_W_MM = 70.0
    PHONE_H_MM = 140.0
    PHONE_OUTER_X0 = -PHONE_W_MM / 2.0
    PHONE_OUTER_Y0 = 0.0

    SCREEN_MARGIN_SIDE = 4.0
    SCREEN_MARGIN_TOP = 5.0
    SCREEN_MARGIN_BOTTOM = 5.0
    SCREEN_X0 = PHONE_OUTER_X0 + SCREEN_MARGIN_SIDE
    SCREEN_X1 = PHONE_OUTER_X0 + PHONE_W_MM - SCREEN_MARGIN_SIDE
    SCREEN_Y0 = PHONE_OUTER_Y0 + SCREEN_MARGIN_BOTTOM
    SCREEN_Y1 = PHONE_OUTER_Y0 + PHONE_H_MM - SCREEN_MARGIN_TOP

    outer_box = FancyBboxPatch(
        (PHONE_OUTER_X0, PHONE_OUTER_Y0),
        PHONE_W_MM,
        PHONE_H_MM,
        boxstyle=BoxStyle("Round", pad=0.0, rounding_size=corner_radius),
        linewidth=3.0,
        edgecolor="white",
        facecolor="none",
        zorder=10,
    )
    ax.add_patch(outer_box)

    inner_box = FancyBboxPatch(
        (SCREEN_X0, SCREEN_Y0),
        SCREEN_X1 - SCREEN_X0,
        SCREEN_Y1 - SCREEN_Y0,
        boxstyle=BoxStyle("Round", pad=0.0, rounding_size=corner_radius - 2.0),
        linewidth=2.0,
        edgecolor="white",
        facecolor="none",
        zorder=11,
    )
    ax.add_patch(inner_box)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("normalized von Mises stress", fontsize=11)

    ax.set_xlabel("x (mm)", fontsize=12)
    ax.set_ylabel("y (mm)", fontsize=12)

    title = (
        "Smartphone Drop-Impact Stress Field\n"
        f"(von Mises • Front View • t={t_val*1e3:.2f} ms)"
    )
    ax.set_title(title, fontsize=16, pad=24)

    # 组件标注（照你原来的 Camera / Battery / Side frame）
    CAM_W_MM = 15.0
    CAM_H_MM = 25.0
    CAM_X0_MM = -CAM_W_MM / 2
    CAM_X1_MM = CAM_W_MM / 2
    CAM_Y1_MM = PHONE_H_MM - 5.0
    CAM_Y0_MM = CAM_Y1_MM - CAM_H_MM

    BAT_W_MM = 30.0
    BAT_H_MM = 40.0
    BAT_X0_MM = -BAT_W_MM / 2
    BAT_X1_MM = BAT_W_MM / 2
    BAT_Y0_MM = 60.0
    BAT_Y1_MM = BAT_Y0_MM + BAT_H_MM

    FRAME_W_MM = 5.0
    LEFT_FRAME_X0_MM = -35.0
    LEFT_FRAME_X1_MM = LEFT_FRAME_X0_MM + FRAME_W_MM
    RIGHT_FRAME_X1_MM = 35.0
    RIGHT_FRAME_X0_MM = RIGHT_FRAME_X1_MM - FRAME_W_MM
    FRAME_Y0_MM = 0.0
    FRAME_Y1_MM = PHONE_H_MM

    def draw_rect(x0, x1, y0, y1, label=None, text_y=None):
        xs = [x0, x1, x1, x0, x0]
        ys = [y0, y0, y1, y1, y0]
        ax.plot(xs, ys, linestyle="--", color="cyan", linewidth=1.3, zorder=12)
        if label is not None:
            if text_y is None:
                text_y = y1 + 4.0
            ax.text(
                (x0 + x1) / 2,
                text_y,
                label,
                color="#00B7FF",
                ha="center",
                va="center",
                fontsize=10,
                zorder=13,
            )

    draw_rect(
        CAM_X0_MM,
        CAM_X1_MM,
        CAM_Y0_MM,
        CAM_Y1_MM,
        "Camera\nModule",
        text_y=CAM_Y1_MM - 5.0,
    )
    draw_rect(
        BAT_X0_MM, BAT_X1_MM, BAT_Y0_MM, BAT_Y1_MM, "Battery", text_y=BAT_Y0_MM - 6.0
    )

    mid_y = (FRAME_Y0_MM + FRAME_Y1_MM) * 0.5
    draw_rect(
        LEFT_FRAME_X0_MM,
        LEFT_FRAME_X1_MM,
        FRAME_Y0_MM,
        FRAME_Y1_MM,
        label=None,
        text_y=mid_y,
    )
    ax.text(
        LEFT_FRAME_X1_MM + 6,
        mid_y,
        "Side\nFrame",
        color="#00B7FF",
        ha="center",
        va="center",
        fontsize=10,
        zorder=13,
    )
    draw_rect(
        RIGHT_FRAME_X0_MM,
        RIGHT_FRAME_X1_MM,
        FRAME_Y0_MM,
        FRAME_Y1_MM,
        label=None,
        text_y=mid_y,
    )
    ax.text(
        RIGHT_FRAME_X0_MM - 6,
        mid_y,
        "Side\nFrame",
        color="#00B7FF",
        ha="center",
        va="center",
        fontsize=10,
        zorder=13,
    )

    fig.subplots_adjust(left=0.18, right=0.88, bottom=0.10, top=0.88)

    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
        print(f"[INFO] saved figure to {savepath}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def find_best_time_for_bottom_contrast(
    model: torch.nn.Module,
    cfg: dict,
    device: torch.device,
    num_t: int = 80,
    bottom_ratio: float = 0.2,
    top_ratio: float = 0.2,
    t_min: float = 0.0005,) -> Tuple[float, float]:
    p_cfg = cfg["physics"]
    T_final = p_cfg["T_final"]

    t_list = torch.linspace(0.0, T_final, num_t).cpu().numpy()
    best_t = None
    best_score = -1.0

    vis_cfg = cfg.get("visualization", {})
    nx = vis_cfg.get("grid_nx", 120)
    ny = vis_cfg.get("grid_ny", 240)

    for t in t_list:
        if float(t) < t_min:
            continue

        _, Y, Svm_star_t = evaluate_stress_field(model, cfg, device, float(t), nx, ny)
        y_vals = Y[0, :]
        Ly_val = y_vals.max()
        y_bottom_max = Ly_val * bottom_ratio
        y_top_min = Ly_val * (1.0 - top_ratio)

        bottom_mask = y_vals <= y_bottom_max
        top_mask = y_vals >= y_top_min

        S_bottom = Svm_star_t[:, bottom_mask].mean()
        S_top = Svm_star_t[:, top_mask].mean()

        score = float(S_bottom / (S_top + 1e-12))

        if score > best_score:
            best_score = score
            best_t = float(t)

    return best_t, best_score


def compute_max_stress_time_curve(
    model: torch.nn.Module,
    cfg: dict,
    device: torch.device,
    num_t: int = 150,):
    p_cfg = cfg["physics"]
    T_final = p_cfg["T_final"]

    vis_cfg = cfg.get("visualization", {})
    nx = vis_cfg.get("grid_nx", 120)
    ny = vis_cfg.get("grid_ny", 240)

    t_list = torch.linspace(0.0, T_final, num_t).cpu().numpy()
    max_vals = []

    for t_val in t_list:
        _, _, Svm_star_t = evaluate_stress_field(
            model, cfg, device, float(t_val), nx, ny
        )
        max_vals.append(Svm_star_t.max())

    max_vals = np.array(max_vals, dtype=np.float32)
    max_vals_norm = (max_vals - max_vals.min()) / (max_vals.max() - max_vals.min() + 1e-12)
    return t_list, max_vals_norm


def plot_max_stress_curve(
    t_samples: np.ndarray,
    max_stress_norm: np.ndarray,
    savepath: Path,):
    plt.figure(figsize=(6, 4))
    plt.plot(t_samples * 1e3, max_stress_norm, linewidth=3.0)
    plt.xlabel("Time (ms)", fontsize=12)
    plt.ylabel("Relative max von Mises (normalized)", fontsize=12)
    plt.title("Maximum von Mises Stress vs Time", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    savepath = Path(savepath)
    savepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savepath, dpi=300)
    plt.close()
    print(f"[INFO] saved max-stress curve to {savepath}")


def precompute_global_max(
    model: torch.nn.Module,
    cfg: dict,
    device: torch.device,
    t_list_ms: List[float],):
    global_max = 0.0
    vis_cfg = cfg.get("visualization", {})
    nx = vis_cfg.get("grid_nx", 120)
    ny = vis_cfg.get("grid_ny", 240)

    for t_ms in t_list_ms:
        t_val = t_ms * 1e-3
        _, _, S_star_t = evaluate_stress_field(model, cfg, device, t_val, nx, ny)
        global_max = max(global_max, float(S_star_t.max()))
    return global_max


def plot_multiple_frames(
    model: torch.nn.Module,
    cfg: dict,
    device: torch.device,
    t_list_ms: List[float],
    prefix: Path,):
    prefix = Path(prefix)
    global_max = precompute_global_max(model, cfg, device, t_list_ms)

    vis_cfg = cfg.get("visualization", {})
    nx = vis_cfg.get("grid_nx", 120)
    ny = vis_cfg.get("grid_ny", 240)

    for t_ms in t_list_ms:
        t_val = t_ms * 1e-3
        X, Y, S_star_t = evaluate_stress_field(model, cfg, device, t_val, nx, ny)
        Svm_norm = S_star_t / (global_max + 1e-12)

        fname = prefix.parent / f"{prefix.name}{t_ms:.2f}ms.png"
        plot_stress_phone(cfg, X, Y, Svm_norm, t_val, savepath=fname, show=False)
        print(f"[INFO] saved {fname}")
