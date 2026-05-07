# src/postprocess.py
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as mticker
from pathlib import Path
from typing import Dict

from .data_utils import load_case_config, build_physical_params

CMAP_VMIN = 300.0
CMAP_VMAX = 325.0


def predict_grid_at_time(model,
                         device,
                         cfg: Dict,
                         params: Dict,
                         t_val: float,
                         nx: int = 60,
                         ny: int = 60):
    """对应你原来的 predict_grid_at_time"""
    Lx = params["Lx"]
    Ly = params["Ly"]

    Lx_scale = params["Lx_scale"]
    Ly_scale = params["Ly_scale"]
    t_scale = params["t_scale"]
    theta_scale = params["theta_scale"]
    T_inf = params["T_inf"]

    x_lin = np.linspace(0.0, Lx, nx)
    y_lin = np.linspace(0.0, Ly, ny)
    X, Y = np.meshgrid(x_lin, y_lin, indexing="xy")

    x_flat = X.reshape(-1, 1)
    y_flat = Y.reshape(-1, 1)
    t_flat = np.full_like(x_flat, t_val)

    x_hat = x_flat / Lx_scale
    y_hat = y_flat / Ly_scale
    t_hat = t_flat / t_scale

    xyt_hat = torch.tensor(
        np.concatenate([x_hat, y_hat, t_hat], axis=1),
        dtype=torch.float32, device=device
    )

    model.eval()
    with torch.no_grad():
        theta_hat_pred = model(xyt_hat).cpu().numpy().reshape(ny, nx)

    theta_pred = theta_scale * theta_hat_pred
    T_pred = T_inf + theta_pred
    return X, Y, T_pred


def plot_temperature_phone(X, Y, T_grid, t_val: float, savepath=None):
    """完全照你原来的手机样式热力图函数写"""
    X_mm = X * 1000.0
    Y_mm = Y * 1000.0

    fig, ax = plt.subplots(figsize=(5.5, 10.0), dpi=220)
    ax.set_facecolor("white")

    x_min, x_max = X_mm.min(), X_mm.max()
    y_min, y_max = Y_mm.min(), Y_mm.max()
    phone_width = x_max - x_min
    phone_height = y_max - y_min

    corner_radius = 12.0

    # 热力图
    im = ax.pcolormesh(
        X_mm, Y_mm, T_grid,
        shading="auto",
        cmap="hot",
        vmin=CMAP_VMIN,
        vmax=CMAP_VMAX,
        zorder=5
    )

    ax.set_aspect("equal")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # clip 轮廓（略大一点）
    clip_pad = 1.0
    clip_box = patches.FancyBboxPatch(
        (x_min - clip_pad, y_min - clip_pad),
        phone_width + 2 * clip_pad,
        phone_height + 2 * clip_pad,
        boxstyle=patches.BoxStyle("Round", pad=0.0, rounding_size=corner_radius + 1.0),
        linewidth=0.0,
        edgecolor="none",
        facecolor="none",
        zorder=4
    )
    ax.add_patch(clip_box)
    im.set_clip_path(clip_box)

    # 外白框（机身）
    outer_box = patches.FancyBboxPatch(
        (x_min, y_min),
        phone_width, phone_height,
        boxstyle=patches.BoxStyle("Round", pad=0.0, rounding_size=corner_radius),
        linewidth=3.0,
        edgecolor="white",
        facecolor="none",
        zorder=10
    )
    ax.add_patch(outer_box)

    # 内白框（屏幕）
    bezel = 4.0
    inner_box = patches.FancyBboxPatch(
        (x_min + bezel, y_min + bezel),
        phone_width - 2 * bezel,
        phone_height - 2 * bezel,
        boxstyle=patches.BoxStyle("Round", pad=0.0, rounding_size=corner_radius - 2.0),
        linewidth=2.0,
        edgecolor="white",
        facecolor="none",
        zorder=11
    )
    ax.add_patch(inner_box)

    # SoC & Camera 区域
    line_color = "#0088ff"

    # SoC
    soc_w, soc_h = 32.0, 40.0
    soc_x = x_min + phone_width / 2.0 - soc_w / 2.0
    soc_y = y_min + phone_height / 2.0 - soc_h / 2.0 + 2.0
    soc_box = patches.Rectangle(
        (soc_x, soc_y), soc_w, soc_h,
        linewidth=1.6,
        edgecolor=line_color,
        facecolor="none",
        linestyle="--",
        zorder=12
    )
    ax.add_patch(soc_box)

    # Camera
    cam_w, cam_h = 20.0, 12.0
    cam_x = x_min + phone_width / 2.0 - cam_w / 2.0
    cam_y = y_max - bezel - cam_h - 10.0
    cam_box = patches.Rectangle(
        (cam_x, cam_y), cam_w, cam_h,
        linewidth=1.6,
        edgecolor=line_color,
        facecolor="none",
        linestyle="--",
        zorder=12
    )
    ax.add_patch(cam_box)

    # Side frame
    side_w = 6.0
    side_margin = 2.0
    side_h = phone_height - 2 * side_margin
    side_left = patches.Rectangle(
        (x_min - side_w, y_min + side_margin),
        side_w, side_h,
        linewidth=1.8,
        edgecolor=line_color,
        facecolor="none",
        linestyle="--",
        zorder=9
    )
    side_right = patches.Rectangle(
        (x_max, y_min + side_margin),
        side_w, side_h,
        linewidth=1.8,
        edgecolor=line_color,
        facecolor="none",
        linestyle="--",
        zorder=9
    )
    ax.add_patch(side_left)
    ax.add_patch(side_right)

    # 文本（axes 坐标，避免被裁掉）
    text_color = line_color
    ax.text(
        0.5, 0.93, "Camera Module",
        color=text_color, ha="center", va="bottom",
        fontsize=11, fontweight="bold",
        transform=ax.transAxes,
        zorder=20
    )
    ax.text(
        0.5, 0.58, "SoC",
        color=text_color, ha="center", va="bottom",
        fontsize=12, fontweight="bold",
        transform=ax.transAxes,
        zorder=20
    )
    ax.text(
        0.06, 0.5, "Side\nFrame",
        color=text_color, ha="left", va="center",
        fontsize=10, fontweight="bold",
        transform=ax.transAxes,
        zorder=20
    )
    ax.text(
        0.94, 0.5, "Side\nFrame",
        color=text_color, ha="right", va="center",
        fontsize=10, fontweight="bold",
        transform=ax.transAxes,
        zorder=20
    )

    ax.set_title(
        f"Smartphone Thermal Field (SoC Heating • Front View • t = {int(t_val)} s)",
        fontsize=16,
        pad=20
    )
    ax.set_xlabel("x (mm)", fontsize=12)
    ax.set_ylabel("y (mm)", fontsize=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03, format="%.1f")
    cbar.set_label("Temperature (K)", fontsize=12)
    formatter = mticker.ScalarFormatter(useMathText=False)
    formatter.set_useOffset(False)
    cbar.formatter = formatter
    cbar.update_ticks()

    ax.tick_params(labelsize=10)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)

    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def save_snapshots(model,
                   device,
                   cfg: Dict,
                   params: Dict,
                   results_dir: str = "results"):
    """保存 300/900/1800 s 三张手机热力图"""
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    for t_val in [300.0, 900.0, 1800.0]:
        X, Y, T_grid = predict_grid_at_time(model, device, cfg, params, t_val, nx=60, ny=60)
        out_path = results_path / f"phone_heat_t{int(t_val)}s.png"
        plot_temperature_phone(X, Y, T_grid, t_val, savepath=out_path)
        print("Saved:", out_path)


def save_center_temperature_curve(model,
                                  device,
                                  cfg: Dict,
                                  params: Dict,
                                  results_dir: str = "results"):
    """SoC 中心点温度随时间变化曲线"""
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    t_max = params["t_max"]
    t_line = np.linspace(0.0, t_max, 200)

    soc = cfg["soc"]
    x_c = soc["x_center"]
    y_c = soc["y_center"]

    Lx_scale = params["Lx_scale"]
    Ly_scale = params["Ly_scale"]
    t_scale = params["t_scale"]
    theta_scale = params["theta_scale"]
    T_inf = params["T_inf"]

    x_c_arr = np.full_like(t_line, x_c)
    y_c_arr = np.full_like(t_line, y_c)

    x_hat_line = x_c_arr / Lx_scale
    y_hat_line = y_c_arr / Ly_scale
    t_hat_line = t_line / t_scale

    xyt_hat_line = torch.tensor(
        np.stack([x_hat_line, y_hat_line, t_hat_line], axis=1),
        dtype=torch.float32,
        device=device
    )

    model.eval()
    with torch.no_grad():
        theta_hat_center = model(xyt_hat_line).cpu().numpy().reshape(-1)

    theta_center = theta_scale * theta_hat_center
    T_center = T_inf + theta_center

    plt.figure(figsize=(7.5, 4.5), dpi=220)
    plt.plot(t_line, T_center, label="PINN prediction", lw=2)
    plt.axhline(T_inf, ls="--", c="gray", label=f"T_inf = {T_inf:.0f} K")
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (K)")
    plt.title(
        f"Temperature evolution at SoC center\n"
        f"(x={x_c * 1000:.1f} mm, y={y_c * 1000:.1f} mm)"
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    curve_path = results_path / "center_T_curve.png"
    plt.savefig(curve_path, dpi=300)
    plt.close()
    print("Saved:", curve_path)
