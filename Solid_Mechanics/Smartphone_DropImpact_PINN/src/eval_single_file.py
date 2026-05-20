"""
Notebook-friendly eval file for Smartphone_DropImpact_PINN.

Online Jupyter usage:
1. Upload the trained .pt model as a public asset.
2. Upload phone_drop_case.yaml as a public asset (optional, default config is built-in).
3. Fill MODEL_HASH and DATA_HASH, then run this whole file/cell.

This file loads the model, evaluates the stress field, and shows figures
with plt.show(). It does not save images.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, BoxStyle

import torch
import torch.nn as nn


# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = "f809062431b143eca8600e4466942506"
MODEL_HASH = "2b050ff2f56c40459cc78ba640dad36f"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/trained_model_phone_drop.pt")


# =========================
# 2. Default config
# =========================

DEFAULT_CFG = {
    "physics": {"Lx": 0.070, "Ly": 0.140, "T_final": 0.020, "E_phys0": 7.0e10,
                "rho_phys0": 2700.0, "nu": 0.30},
    "materials": {
        "base": {"E_star": 1.0, "rho_star": 1.0},
        "camera": {"E_factor": 1.4, "rho_factor": 1.15, "region_mm": {"x0": -7.5, "x1": 7.5, "y0": 115.0, "y1": 140.0}},
        "battery": {"E_factor": 0.7, "rho_factor": 1.4, "region_mm": {"x0": -15.0, "x1": 15.0, "y0": 60.0, "y1": 100.0}},
        "side_frame": {"E_factor": 1.3, "rho_factor": 1.1, "left_x0_mm": -35.0, "left_x1_mm": -30.0,
                       "right_x0_mm": 30.0, "right_x1_mm": 35.0, "y0_mm": 0.0, "y1_mm": 140.0},
    },
    "impact": {"use_time_envelope": True, "use_body_force": True, "body_force_peak": 40.0,
               "pulse_t0": 0.0015, "pulse_tau": 0.0005, "bottom_focus_y": 0.01, "bottom_focus_ratio": 0.5},
}


def load_or_default_config(data_path):
    import yaml
    data_path = os.path.expanduser(data_path) if data_path else ""
    yaml_path = None
    if data_path and os.path.isfile(data_path) and data_path.endswith((".yaml", ".yml")):
        yaml_path = data_path
    elif data_path and os.path.isdir(data_path):
        for f in sorted(os.listdir(data_path)):
            if f.endswith((".yaml", ".yml")):
                yaml_path = os.path.join(data_path, f)
                break
    if yaml_path:
        with open(yaml_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    return DEFAULT_CFG


# =========================
# 3. Model definition
# =========================

class PINN2D(nn.Module):
    def __init__(self, in_dim=3, out_dim=2, width=64, depth=5,
                 use_symmetry=False, Lx=0.07, Ly=0.14, T_final=0.02):
        super().__init__()
        self.use_symmetry = use_symmetry
        self.Lx, self.Ly, self.T_final = Lx, Ly, T_final
        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, y, t):
        if self.use_symmetry:
            x_n = (x ** 2) / ((self.Lx / 2.0) ** 2)
        else:
            x_n = x / (self.Lx / 2.0)
        y_n = (y - self.Ly / 2.0) / (self.Ly / 2.0)
        t_n = (t - self.T_final / 2.0) / (self.T_final / 2.0)
        X = torch.cat([x_n, y_n, t_n], dim=1)
        return self.net(X)


def grads(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True, retain_graph=True)[0]


# =========================
# 4. Physics / material utilities (for evaluation)
# =========================

def material_factor(x, y, cfg):
    mats = cfg.get("materials", {})
    camera = mats.get("camera", {}); battery = mats.get("battery", {}); side = mats.get("side_frame", {})
    x_mm, y_mm = x * 1e3, y * 1e3
    E_factor = torch.ones_like(x); rho_factor = torch.ones_like(x)
    cam = camera.get("region_mm", {})
    if cam:
        m = (x_mm >= cam.get("x0", -1e9)) & (x_mm <= cam.get("x1", 1e9)) & (y_mm >= cam.get("y0", -1e9)) & (y_mm <= cam.get("y1", 1e9))
        E_factor[m] = camera.get("E_factor", 1.0); rho_factor[m] = camera.get("rho_factor", 1.0)
    bat = battery.get("region_mm", {})
    if bat:
        m = (x_mm >= bat.get("x0", -1e9)) & (x_mm <= bat.get("x1", 1e9)) & (y_mm >= bat.get("y0", -1e9)) & (y_mm <= bat.get("y1", 1e9))
        E_factor[m] = battery.get("E_factor", 1.0); rho_factor[m] = battery.get("rho_factor", 1.0)
    if side:
        y0, y1 = side.get("y0_mm", 0.0), side.get("y1_mm", 140.0)
        lm = ((x_mm >= side.get("left_x0_mm", -35)) & (x_mm <= side.get("left_x1_mm", -30)) & (y_mm >= y0) & (y_mm <= y1))
        rm = ((x_mm >= side.get("right_x0_mm", 30)) & (x_mm <= side.get("right_x1_mm", 35)) & (y_mm >= y0) & (y_mm <= y1))
        E_factor[lm | rm] = side.get("E_factor", 1.0); rho_factor[lm | rm] = side.get("rho_factor", 1.0)
    return E_factor, rho_factor


def strain_stress(u, x, y, lam_base, mu_base):
    ux, uy = u[:, 0:1], u[:, 1:2]
    ux_x, ux_y = grads(ux, x), grads(ux, y)
    uy_x, uy_y = grads(uy, x), grads(uy, y)
    eps_xx, eps_yy = ux_x, uy_y
    eps_xy = 0.5 * (ux_y + uy_x)
    eps_vol = eps_xx + eps_yy
    sig_xx = lam_base * eps_vol + 2 * mu_base * eps_xx
    sig_yy = lam_base * eps_vol + 2 * mu_base * eps_yy
    sig_xy = 2 * mu_base * eps_xy
    return sig_xx, sig_yy, sig_xy


def evaluate_stress_field(model, cfg, device, t_val, nx=120, ny=240):
    p_cfg = cfg["physics"]
    Lx, Ly = p_cfg["Lx"], p_cfg["Ly"]
    x = torch.linspace(-Lx/2, Lx/2, nx, device=device)
    y = torch.linspace(0.0, Ly, ny, device=device)
    X, Y = torch.meshgrid(x, y, indexing="xy")
    x_in = X.reshape(-1, 1); y_in = Y.reshape(-1, 1)
    t_in = torch.full_like(x_in, t_val)
    x_in.requires_grad_(True); y_in.requires_grad_(True); t_in.requires_grad_(True)

    nu = p_cfg.get("nu", 0.30); lam_base = nu / ((1+nu)*(1-2*nu)); mu_base = 1.0/(2*(1+nu))
    u = model(x_in, y_in, t_in)
    E_factor, _ = material_factor(x_in, y_in, cfg)
    sxx, syy, sxy = strain_stress(u, x_in, y_in, lam_base, mu_base)
    sxx, syy, sxy = sxx * E_factor, syy * E_factor, sxy * E_factor
    Svm = torch.sqrt(sxx**2 + syy**2 - sxx*syy + 3*sxy**2 + 1e-12)
    return X.detach().cpu().numpy(), Y.detach().cpu().numpy(), Svm.reshape(X.shape).detach().cpu().numpy()


def load_model_package(model_path, device):
    model_path = os.path.expanduser(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"MODEL_PATH does not exist: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    cfg_ckpt = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    model = PINN2D(
        in_dim=3, out_dim=2,
        width=cfg_ckpt.get("width", 64), depth=cfg_ckpt.get("depth", 5),
        use_symmetry=cfg_ckpt.get("use_symmetry", False),
        Lx=cfg_ckpt.get("Lx", 0.07), Ly=cfg_ckpt.get("Ly", 0.14),
        T_final=cfg_ckpt.get("T_final", 0.02),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint


# =========================
# 5. Visualization
# =========================

def show_phone_stress(cfg, X, Y, Svm_norm, t_val, title="Smartphone Drop-Impact Stress Field"):
    X_mm = X * 1000.0; Y_mm = Y * 1000.0
    fig, ax = plt.subplots(figsize=(5.5, 10.0), dpi=180)
    ax.set_facecolor("white")
    x_min, x_max = X_mm.min(), X_mm.max()
    y_min, y_max = Y_mm.min(), Y_mm.max()
    phone_w = x_max - x_min; phone_h = y_max - y_min
    corner_r = 12.0

    im = ax.pcolormesh(X_mm, Y_mm, Svm_norm, shading="auto", cmap="magma", vmin=0.0, vmax=1.0, zorder=5)
    ax.set_aspect("equal"); ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    for spine in ax.spines.values():
        spine.set_visible(False)

    pad = 1.0; cb = FancyBboxPatch((x_min-pad, y_min-pad), phone_w+2*pad, phone_h+2*pad,
                                    boxstyle=BoxStyle("Round", pad=0.0, rounding_size=corner_r+1.0),
                                    linewidth=0.0, edgecolor="none", facecolor="none", zorder=4)
    ax.add_patch(cb); im.set_clip_path(cb)

    PHONE_W, PHONE_H = 70.0, 140.0; PX0, PY0 = -PHONE_W/2.0, 0.0
    outer = FancyBboxPatch((PX0, PY0), PHONE_W, PHONE_H,
                           boxstyle=BoxStyle("Round", pad=0.0, rounding_size=corner_r),
                           linewidth=3.0, edgecolor="white", facecolor="none", zorder=10)
    ax.add_patch(outer)
    inner = FancyBboxPatch((PX0+4, PY0+5), PHONE_W-8, PHONE_H-10,
                           boxstyle=BoxStyle("Round", pad=0.0, rounding_size=corner_r-2),
                           linewidth=2.0, edgecolor="white", facecolor="none", zorder=11)
    ax.add_patch(inner)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("normalized von Mises stress", fontsize=11)
    ax.set_xlabel("x (mm)", fontsize=12); ax.set_ylabel("y (mm)", fontsize=12)
    ax.set_title(f"{title}\n(von Mises | Front View | t={t_val*1e3:.2f} ms)", fontsize=14, pad=20)
    fig.subplots_adjust(left=0.18, right=0.88, bottom=0.10, top=0.88)
    plt.show()


def show_eval_result(model, cfg, checkpoint, device):
    t_cfg = cfg.get("training", {})
    history = checkpoint.get("history", []) if isinstance(checkpoint, dict) else []
    if history:
        plt.figure(figsize=(8, 4))
        plt.plot(history, linewidth=1); plt.yscale("log")
        plt.xlabel("Epoch"); plt.ylabel("Total Loss")
        plt.title("Training Loss"); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.show()

    T_final = cfg["physics"]["T_final"]
    t_list = np.linspace(0.0, T_final, 80)
    best_t = T_final * 0.1
    best_score = -1.0
    Ly_val = cfg["physics"]["Ly"]
    for t in t_list:
        if float(t) < 0.0005:
            continue
        _, Y, Svm = evaluate_stress_field(model, cfg, device, float(t), nx=60, ny=120)
        yv = Y[:, 0]
        b_idx = int(len(yv) * 0.2); t_idx = int(len(yv) * 0.8)
        s_bottom = Svm[:b_idx, :].mean()
        s_top = Svm[t_idx:, :].mean()
        s = float(s_bottom / (s_top + 1e-12))
        if s > best_score:
            best_score, best_t = s, float(t)
    print(f"Best bottom-contrast time: t={best_t*1e3:.2f} ms (bottom/top={best_score:.2f})")

    t_samples = np.linspace(0.0, T_final, 80)
    max_vals = []
    for t_val in t_samples:
        _, _, Svm = evaluate_stress_field(model, cfg, device, float(t_val), nx=60, ny=120)
        max_vals.append(float(Svm.max()))
    max_vals = np.array(max_vals)
    max_norm = (max_vals - max_vals.min()) / (max_vals.max() - max_vals.min() + 1e-12)

    plt.figure(figsize=(7, 4))
    plt.plot(t_samples*1e3, max_norm, linewidth=3)
    plt.xlabel("Time (ms)"); plt.ylabel("Relative max von Mises (norm.)")
    plt.title("Maximum von Mises Stress vs Time"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.show()

    for t_ms in [0.25, 1.0, 5.0]:
        X, Y, Svm = evaluate_stress_field(model, cfg, device, t_ms*1e-3, nx=60, ny=120)
        S_norm = Svm / (Svm.max() + 1e-12)
        show_phone_stress(cfg, X, Y, S_norm, t_ms*1e-3)

    X, Y, Svm = evaluate_stress_field(model, cfg, device, best_t, nx=60, ny=120)
    S_norm = Svm / (Svm.max() + 1e-12)
    show_phone_stress(cfg, X, Y, S_norm, best_t, title=f"Best Bottom Contrast (t={best_t*1e3:.2f} ms)")


# =========================
# 6. Eval entry
# =========================

def evaluate(model_path=MODEL_PATH, data_path=DATA_PATH):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"device={device}")

    model, checkpoint = load_model_package(model_path, device)
    cfg = load_or_default_config(data_path)

    print("Eval result")
    print(f"model_path: {model_path}")
    if isinstance(checkpoint, dict):
        hist = checkpoint.get("history", [])
        if hist:
            print(f"final_loss: {hist[-1]:.6e}")

    show_eval_result(model, cfg, checkpoint, device)
    return {"model": model, "checkpoint": checkpoint, "cfg": cfg}


eval_result = evaluate()
