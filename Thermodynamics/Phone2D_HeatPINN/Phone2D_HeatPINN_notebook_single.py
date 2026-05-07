"""
Notebook-friendly single-file version of Smartphone 2D Thermal PINN.

This file does not need any external dataset or project-local src modules.
In a notebook, run:

    %run Thermodynamics/Phone2D_HeatPINN/Phone2D_HeatPINN_notebook_single.py

Or import and call:

    from Phone2D_HeatPINN_notebook_single import run_demo
    model, history = run_demo(n_iters=1000)
"""

from pathlib import Path
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as mticker
import numpy as np
import torch
import torch.nn as nn


CONFIG = {
    "problem_name": "Smartphone 2D Thermal PINN",
    "geometry": {"Lx": 0.07, "Ly": 0.15, "t_max": 1800.0},
    "material": {"rho": 2500.0, "cp": 900.0, "k": 5.0},
    "environment": {"T_inf": 298.0, "h_env": 5.0},
    "soc": {
        "x_center": 0.035,
        "y_center": 0.075,
        "width": 0.01,
        "height": 0.01,
        "q0": 3_000_000.0,
        "tau_q": 100.0,
        "use_slow_source": True,
    },
    "nondim": {"theta_scale": 10.0},
    "sampling": {
        # The original project uses 15000/6000/6000/2000 and 30000 iters.
        # These smaller defaults make a notebook smoke run practical.
        "N_r": 2000,
        "N_ic": 800,
        "N_ic2": 0,
        "N_bc_edge": 300,
        "use_extra_t0_samples": False,
    },
    "training": {
        "n_iters": 1000,
        "print_every": 100,
        "lr": 1e-3,
        "w_pde": 1.0,
        "w_ic": 20.0,
        "w_ic2": 10.0,
        "w_bc": 1.0,
    },
}


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_physical_params(cfg):
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
    return {
        "rho": rho,
        "cp": cp,
        "k": k,
        "alpha": alpha,
        "Lx": Lx,
        "Ly": Ly,
        "t_max": t_max,
        "T_inf": T_inf,
        "h_env": h_env,
        "theta_scale": theta_scale,
        "Lx_scale": Lx,
        "Ly_scale": Ly,
        "t_scale": t_max,
        "alpha_x_hat": alpha * t_max / (Lx**2),
        "alpha_y_hat": alpha * t_max / (Ly**2),
        "Bi_x": h_env * Lx / k,
        "Bi_y": h_env * Ly / k,
        "q_to_theta": 1.0 / (rho * cp),
    }


def q_source_theta(x, y, t, cfg, params):
    soc = cfg["soc"]
    cond_x = torch.abs(x - soc["x_center"]) <= soc["width"] / 2.0
    cond_y = torch.abs(y - soc["y_center"]) <= soc["height"] / 2.0
    mask = (cond_x & cond_y).float()
    if soc.get("use_slow_source", True):
        factor_t = 1.0 - torch.exp(-t / soc["tau_q"])
    else:
        factor_t = torch.ones_like(t)
    return soc["q0"] * mask * factor_t * params["q_to_theta"]


def sample_pde_points(N_r, cfg, params):
    Lx, Ly, t_max = params["Lx"], params["Ly"], params["t_max"]
    soc = cfg["soc"]
    N_early = int(0.6 * N_r)
    N_full = N_r - N_early

    x_e = torch.rand(N_early, 1) * Lx
    y_e = torch.rand(N_early, 1) * Ly
    t_e = torch.rand(N_early, 1) * (0.3 * t_max)

    x_f = torch.rand(N_full, 1) * Lx
    y_f = torch.rand(N_full, 1) * Ly
    t_f = torch.rand(N_full, 1) * t_max

    x_r = torch.cat([x_e, x_f], dim=0)
    y_r = torch.cat([y_e, y_f], dim=0)
    t_r = torch.cat([t_e, t_f], dim=0)

    N_half = N_r // 2
    idx = torch.randperm(N_r)[:N_half]
    x_soc = soc["x_center"] + (torch.rand(N_half, 1) - 0.5) * (2.0 * soc["width"])
    y_soc = soc["y_center"] + (torch.rand(N_half, 1) - 0.5) * (2.0 * soc["height"])
    x_r[idx, :] = torch.clamp(x_soc, 0.0, Lx)
    y_r[idx, :] = torch.clamp(y_soc, 0.0, Ly)
    return x_r, y_r, t_r


def sample_points(cfg, params):
    sampling = cfg["sampling"]
    Lx, Ly, t_max = params["Lx"], params["Ly"], params["t_max"]
    N_r = sampling["N_r"]
    N_ic = sampling["N_ic"]
    N_ic2 = sampling["N_ic2"]
    N_bc_edge = sampling["N_bc_edge"]

    x_r, y_r, t_r = sample_pde_points(N_r, cfg, params)
    x_ic = torch.rand(N_ic, 1) * Lx
    y_ic = torch.rand(N_ic, 1) * Ly
    t_ic = torch.zeros(N_ic, 1)

    if sampling.get("use_extra_t0_samples", False) and N_ic2 > 0:
        x_ic2 = torch.rand(N_ic2, 1) * Lx
        y_ic2 = torch.rand(N_ic2, 1) * Ly
        t_ic2 = torch.rand(N_ic2, 1) * (0.02 * t_max)
    else:
        x_ic2 = torch.zeros(0, 1)
        y_ic2 = torch.zeros(0, 1)
        t_ic2 = torch.zeros(0, 1)

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
        x_top, y_top, t_top,
    )


class HeatPINN(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=64, n_layers=5):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, xyt_hat):
        return self.net(xyt_hat)


def compute_loss(model, device, batch, cfg, params):
    (
        x_r, y_r, t_r,
        x_ic, y_ic, t_ic,
        x_ic2, y_ic2, t_ic2,
        x_left, y_left, t_left,
        x_right, y_right, t_right,
        x_bottom, y_bottom, t_bottom,
        x_top, y_top, t_top,
    ) = [item.to(device) for item in batch]

    Lx_scale = params["Lx_scale"]
    Ly_scale = params["Ly_scale"]
    t_scale = params["t_scale"]
    theta_scale = params["theta_scale"]
    alpha_x_hat = params["alpha_x_hat"]
    alpha_y_hat = params["alpha_y_hat"]
    Bi_x = params["Bi_x"]
    Bi_y = params["Bi_y"]

    xh_r = (x_r / Lx_scale).clone().detach().requires_grad_(True)
    yh_r = (y_r / Ly_scale).clone().detach().requires_grad_(True)
    th_r = (t_r / t_scale).clone().detach().requires_grad_(True)
    theta_hat_r = model(torch.cat([xh_r, yh_r, th_r], dim=1))

    dtheta_dt = torch.autograd.grad(
        theta_hat_r, th_r, torch.ones_like(theta_hat_r), retain_graph=True, create_graph=True
    )[0]
    dtheta_dx = torch.autograd.grad(
        theta_hat_r, xh_r, torch.ones_like(theta_hat_r), retain_graph=True, create_graph=True
    )[0]
    dtheta_dy = torch.autograd.grad(
        theta_hat_r, yh_r, torch.ones_like(theta_hat_r), retain_graph=True, create_graph=True
    )[0]
    d2theta_dx2 = torch.autograd.grad(
        dtheta_dx, xh_r, torch.ones_like(dtheta_dx), retain_graph=True, create_graph=True
    )[0]
    d2theta_dy2 = torch.autograd.grad(
        dtheta_dy, yh_r, torch.ones_like(dtheta_dy), retain_graph=True, create_graph=True
    )[0]

    q_hat = q_source_theta(x_r, y_r, t_r, cfg, params) * (t_scale / theta_scale)
    res_pde = dtheta_dt - alpha_x_hat * d2theta_dx2 - alpha_y_hat * d2theta_dy2 - q_hat
    loss_pde = torch.mean(res_pde**2)

    theta_hat_ic = model(torch.cat([x_ic / Lx_scale, y_ic / Ly_scale, t_ic / t_scale], dim=1))
    loss_ic = torch.mean(theta_hat_ic**2)

    if x_ic2.numel() > 0:
        theta_hat_ic2 = model(torch.cat([x_ic2 / Lx_scale, y_ic2 / Ly_scale, t_ic2 / t_scale], dim=1))
        loss_ic2 = torch.mean(theta_hat_ic2**2)
    else:
        loss_ic2 = torch.tensor(0.0, device=device)

    xh_left = (x_left / Lx_scale).clone().detach().requires_grad_(True)
    theta_left = model(torch.cat([xh_left, y_left / Ly_scale, t_left / t_scale], dim=1))
    dtheta_dx_left = torch.autograd.grad(
        theta_left, xh_left, torch.ones_like(theta_left), retain_graph=True, create_graph=True
    )[0]
    loss_left = torch.mean((dtheta_dx_left - Bi_x * theta_left) ** 2)

    xh_right = (x_right / Lx_scale).clone().detach().requires_grad_(True)
    theta_right = model(torch.cat([xh_right, y_right / Ly_scale, t_right / t_scale], dim=1))
    dtheta_dx_right = torch.autograd.grad(
        theta_right, xh_right, torch.ones_like(theta_right), retain_graph=True, create_graph=True
    )[0]
    loss_right = torch.mean((-dtheta_dx_right - Bi_x * theta_right) ** 2)

    yh_bottom = (y_bottom / Ly_scale).clone().detach().requires_grad_(True)
    theta_bottom = model(torch.cat([x_bottom / Lx_scale, yh_bottom, t_bottom / t_scale], dim=1))
    dtheta_dy_bottom = torch.autograd.grad(
        theta_bottom, yh_bottom, torch.ones_like(theta_bottom), retain_graph=True, create_graph=True
    )[0]
    loss_bottom = torch.mean((dtheta_dy_bottom - Bi_y * theta_bottom) ** 2)

    yh_top = (y_top / Ly_scale).clone().detach().requires_grad_(True)
    theta_top = model(torch.cat([x_top / Lx_scale, yh_top, t_top / t_scale], dim=1))
    dtheta_dy_top = torch.autograd.grad(
        theta_top, yh_top, torch.ones_like(theta_top), retain_graph=True, create_graph=True
    )[0]
    loss_top = torch.mean((-dtheta_dy_top - Bi_y * theta_top) ** 2)

    loss_bc = loss_left + loss_right + loss_bottom + loss_top
    tr = cfg["training"]
    loss_total = (
        tr["w_pde"] * loss_pde
        + tr["w_ic"] * loss_ic
        + tr["w_ic2"] * loss_ic2
        + tr["w_bc"] * loss_bc
    )
    return loss_total, {
        "total": float(loss_total.detach().cpu()),
        "pde": float(loss_pde.detach().cpu()),
        "ic": float(loss_ic.detach().cpu()),
        "ic2": float(loss_ic2.detach().cpu()),
        "bc": float(loss_bc.detach().cpu()),
    }


def train_model(cfg, params, n_iters=None, print_every=None, seed=42):
    set_random_seed(seed)
    device = get_device()
    model = HeatPINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    batch = sample_points(cfg, params)
    n_iters = n_iters or cfg["training"]["n_iters"]
    print_every = print_every or cfg["training"]["print_every"]
    history = []

    print(f"Using device: {device}")
    for it in range(1, n_iters + 1):
        optimizer.zero_grad()
        loss, loss_dict = compute_loss(model, device, batch, cfg, params)
        loss.backward()
        optimizer.step()
        if it == 1 or it % print_every == 0 or it == n_iters:
            history.append({"iter": it, **loss_dict})
            print(
                f"Iter {it:5d}/{n_iters} | loss={loss_dict['total']:.3e} | "
                f"pde={loss_dict['pde']:.3e} | ic={loss_dict['ic']:.3e} | "
                f"bc={loss_dict['bc']:.3e}"
            )
    return model, device, history


def predict_grid_at_time(model, device, cfg, params, t_val, nx=60, ny=60):
    Lx, Ly = params["Lx"], params["Ly"]
    x_lin = np.linspace(0.0, Lx, nx)
    y_lin = np.linspace(0.0, Ly, ny)
    X, Y = np.meshgrid(x_lin, y_lin, indexing="xy")
    x_flat = X.reshape(-1, 1)
    y_flat = Y.reshape(-1, 1)
    t_flat = np.full_like(x_flat, t_val)
    xyt_hat = torch.tensor(
        np.concatenate(
            [
                x_flat / params["Lx_scale"],
                y_flat / params["Ly_scale"],
                t_flat / params["t_scale"],
            ],
            axis=1,
        ),
        dtype=torch.float32,
        device=device,
    )
    model.eval()
    with torch.no_grad():
        theta_hat = model(xyt_hat).cpu().numpy().reshape(ny, nx)
    T_pred = params["T_inf"] + params["theta_scale"] * theta_hat
    return X, Y, T_pred


def plot_temperature_phone(X, Y, T_grid, t_val, savepath=None, show=False):
    X_mm = X * 1000.0
    Y_mm = Y * 1000.0
    fig, ax = plt.subplots(figsize=(5.5, 10.0), dpi=160)
    x_min, x_max = X_mm.min(), X_mm.max()
    y_min, y_max = Y_mm.min(), Y_mm.max()
    phone_width = x_max - x_min
    phone_height = y_max - y_min

    im = ax.pcolormesh(X_mm, Y_mm, T_grid, shading="auto", cmap="hot", vmin=298.0, vmax=325.0)
    ax.set_aspect("equal")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    outer_box = patches.FancyBboxPatch(
        (x_min, y_min),
        phone_width,
        phone_height,
        boxstyle=patches.BoxStyle("Round", pad=0.0, rounding_size=12.0),
        linewidth=3.0,
        edgecolor="white",
        facecolor="none",
    )
    ax.add_patch(outer_box)

    soc_w, soc_h = 10.0, 10.0
    soc_x = x_min + phone_width / 2.0 - soc_w / 2.0
    soc_y = y_min + phone_height / 2.0 - soc_h / 2.0
    ax.add_patch(
        patches.Rectangle(
            (soc_x, soc_y),
            soc_w,
            soc_h,
            linewidth=1.5,
            edgecolor="#0088ff",
            facecolor="none",
            linestyle="--",
        )
    )
    ax.text(0.5, 0.53, "SoC", color="#0088ff", ha="center", va="bottom", transform=ax.transAxes)
    ax.set_title(f"Smartphone Thermal Field, t = {int(t_val)} s")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03, format="%.1f")
    cbar.set_label("Temperature (K)")
    cbar.formatter = mticker.ScalarFormatter(useMathText=False)
    cbar.update_ticks()
    plt.tight_layout()

    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=120)
    plt.close(fig)
    if show and savepath is not None:
        display_saved_image(savepath)


def plot_center_temperature_curve(model, device, cfg, params, savepath=None, show=False):
    t_line = np.linspace(0.0, params["t_max"], 200)
    soc = cfg["soc"]
    x_hat = np.full_like(t_line, soc["x_center"] / params["Lx_scale"])
    y_hat = np.full_like(t_line, soc["y_center"] / params["Ly_scale"])
    t_hat = t_line / params["t_scale"]
    xyt_hat = torch.tensor(np.stack([x_hat, y_hat, t_hat], axis=1), dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        theta_hat = model(xyt_hat).cpu().numpy().reshape(-1)
    T_center = params["T_inf"] + params["theta_scale"] * theta_hat

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=160)
    ax.plot(t_line, T_center, lw=2, label="PINN prediction")
    ax.axhline(params["T_inf"], ls="--", c="gray", label=f"T_inf = {params['T_inf']:.0f} K")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (K)")
    ax.set_title("Temperature evolution at SoC center")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=120)
    plt.close(fig)
    if show and savepath is not None:
        display_saved_image(savepath)


def display_saved_image(path):
    try:
        from IPython.display import Image, display

        display(Image(filename=str(path)))
    except Exception:
        print(f"Saved image: {Path(path).resolve()}")


def save_figures(model, device, cfg, params, output_dir="results_notebook", show=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for t_val in [300.0, 900.0, 1800.0]:
        X, Y, T_grid = predict_grid_at_time(model, device, cfg, params, t_val)
        plot_temperature_phone(
            X,
            Y,
            T_grid,
            t_val,
            savepath=output_dir / f"phone_heat_t{int(t_val)}s.png",
            show=show,
        )
    plot_center_temperature_curve(
        model,
        device,
        cfg,
        params,
        savepath=output_dir / "center_T_curve.png",
        show=show,
    )
    print(f"Figures saved to: {output_dir.resolve()}")


def save_prediction_arrays(model, device, cfg, params, output_dir="results_notebook"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for t_val in [300.0, 900.0, 1800.0]:
        X, Y, T_grid = predict_grid_at_time(model, device, cfg, params, t_val)
        np.savez(output_dir / f"phone_heat_t{int(t_val)}s.npz", X=X, Y=Y, T=T_grid)

    t_line = np.linspace(0.0, params["t_max"], 200)
    soc = cfg["soc"]
    x_hat = np.full_like(t_line, soc["x_center"] / params["Lx_scale"])
    y_hat = np.full_like(t_line, soc["y_center"] / params["Ly_scale"])
    t_hat = t_line / params["t_scale"]
    xyt_hat = torch.tensor(np.stack([x_hat, y_hat, t_hat], axis=1), dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        theta_hat = model(xyt_hat).cpu().numpy().reshape(-1)
    T_center = params["T_inf"] + params["theta_scale"] * theta_hat
    np.savez(output_dir / "center_T_curve.npz", t=t_line, T=T_center)
    print(f"Prediction arrays saved to: {output_dir.resolve()}")


def run_demo(
    n_iters=1000,
    print_every=100,
    seed=42,
    output_dir="results_notebook",
    save_outputs=False,
    make_plots=False,
    show=False,
):
    cfg = {key: value.copy() if isinstance(value, dict) else value for key, value in CONFIG.items()}
    cfg["training"]["n_iters"] = n_iters
    cfg["training"]["print_every"] = print_every
    params = build_physical_params(cfg)
    model, device, history = train_model(cfg, params, n_iters=n_iters, print_every=print_every, seed=seed)
    if save_outputs:
        save_prediction_arrays(model, device, cfg, params, output_dir=output_dir)
    if make_plots:
        save_figures(model, device, cfg, params, output_dir=output_dir, show=show)
    return model, history


if __name__ == "__main__":
    run_demo(n_iters=1000, print_every=100, save_outputs=False, make_plots=False, show=False)
