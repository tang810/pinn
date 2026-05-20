"""
Notebook-friendly train file for Smartphone_DropImpact_PINN.

Online Jupyter usage:
1. Upload phone_drop_case.yaml as a public asset (optional, default config is built-in).
2. Fill DATA_HASH and MODEL_HASH.
3. Run this whole file/cell. It saves trained_model_phone_drop.pt to
   ~/minio/Resource/{MODEL_HASH}/ for later public-asset upload.

Physics: 2D linear elasticity PINN for phone drop impact stress field.
"""

import os
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

def get_device():
    """Detect device in priority: GPU (cuda) -> NPU (npu/ascend) -> CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_npu
        if torch_npu.npu.is_available():
            return torch.device("npu")
    except (ImportError, AttributeError):
        pass
    return torch.device("cpu")


# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = "f809062431b143eca8600e4466942506"
MODEL_HASH = "f809062431b143eca8600e4466942506"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/minio/Resource/{MODEL_HASH}/trained_model_phone_drop.pt")


# =========================
# 2. Default config (can be overridden by YAML)
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
    "initial_condition": {"v0_y": -2.0},
    "training": {"epochs_quick": 2000, "epochs_full": 7600, "lr": 1.0e-3, "N_pde": 5000, "N_ic": 2000, "N_bc": 2000,
                 "ic_weight": 1.0, "bc_weight": 1.0, "scheduler_step": 2000, "scheduler_gamma": 0.5},
    "model": {"in_dim": 3, "out_dim": 2, "width": 64, "depth": 5, "use_symmetry": False},
}


def load_or_default_config(data_path):
    data_path = os.path.expanduser(data_path) if data_path else ""
    import yaml
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
            cfg = yaml.safe_load(fh)
        print(f"Loaded config from: {yaml_path}")
        return cfg
    print("Using built-in default config.")
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
# 4. Physics utilities
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


def body_force(x, y, t, rho_local, cfg):
    imp = cfg.get("impact", {})
    if (not imp.get("use_body_force", True)) or imp.get("body_force_peak", 0.0) == 0.0:
        return torch.zeros_like(x), torch.zeros_like(y)
    peak = imp.get("body_force_peak", 40.0)
    t0 = torch.tensor(imp.get("pulse_t0", 0.0015), dtype=t.dtype, device=t.device)
    tau = torch.tensor(imp.get("pulse_tau", 0.0005), dtype=t.dtype, device=t.device)
    env_t = torch.exp(-((t - t0) / (tau + 1e-12)) ** 2)
    by = torch.tensor(imp.get("bottom_focus_y", 0.01), dtype=y.dtype, device=y.device)
    spatial_decay = torch.exp(-(y / (by + 1e-12)) ** 2)
    fy = rho_local * peak * env_t * (y <= by).float() * spatial_decay
    return torch.zeros_like(fy), fy


def strain_stress(u, x, y, lam_base, mu_base):
    ux, uy = u[:, 0:1], u[:, 1:2]
    ux_x, ux_y, uy_x, uy_y = grads(ux, x), grads(ux, y), grads(uy, x), grads(uy, y)
    eps_xx, eps_yy, eps_xy = ux_x, uy_y, 0.5 * (ux_y + uy_x)
    eps_vol = eps_xx + eps_yy
    sig_xx = lam_base * eps_vol + 2 * mu_base * eps_xx
    sig_yy = lam_base * eps_vol + 2 * mu_base * eps_yy
    sig_xy = 2 * mu_base * eps_xy
    return sig_xx, sig_yy, sig_xy


def sample_pde(N, Lx, Ly, T_final, bottom_focus_ratio, bottom_focus_y, device):
    Nb = int(N * bottom_focus_ratio); Nu = N - Nb
    xu = (torch.rand(Nu, 1, device=device) - 0.5) * Lx
    yu = torch.rand(Nu, 1, device=device) * Ly
    tu = torch.rand(Nu, 1, device=device) * T_final
    xb = (torch.rand(Nb, 1, device=device) - 0.5) * Lx
    yb = torch.rand(Nb, 1, device=device) * bottom_focus_y
    tb = torch.rand(Nb, 1, device=device) * T_final
    return torch.cat([xu, xb], 0), torch.cat([yu, yb], 0), torch.cat([tu, tb], 0)


def sample_ic(N, Lx, Ly, device):
    x = (torch.rand(N, 1, device=device) - 0.5) * Lx
    y = torch.rand(N, 1, device=device) * Ly
    return x, y, torch.zeros_like(x)


def sample_bc_bottom(N, Lx, T_final, device):
    x = (torch.rand(N, 1, device=device) - 0.5) * Lx
    return x, torch.zeros_like(x), torch.rand(N, 1, device=device) * T_final


def compute_loss(model, cfg, device):
    p_cfg, t_cfg, imp_cfg = cfg["physics"], cfg["training"], cfg["impact"]
    Lx, Ly, T_final = p_cfg["Lx"], p_cfg["Ly"], p_cfg["T_final"]
    N_pde, N_ic, N_bc = t_cfg["N_pde"], t_cfg["N_ic"], t_cfg["N_bc"]
    bfr = imp_cfg.get("bottom_focus_ratio", 0.5); bfy = imp_cfg.get("bottom_focus_y", 0.01)
    v0_y = cfg.get("initial_condition", {}).get("v0_y", -2.0)

    nu = p_cfg.get("nu", 0.30); E_star = 1.0
    lam_base = E_star * nu / ((1 + nu) * (1 - 2 * nu))
    mu_base = E_star / (2 * (1 + nu))

    x_p, y_p, t_p = sample_pde(N_pde, Lx, Ly, T_final, bfr, bfy, device)
    x_p.requires_grad_(True); y_p.requires_grad_(True); t_p.requires_grad_(True)
    u_p = model(x_p, y_p, t_p)
    ux_tt = grads(grads(u_p[:, 0:1], t_p), t_p)
    uy_tt = grads(grads(u_p[:, 1:2], t_p), t_p)
    E_p, rho_p = material_factor(x_p, y_p, cfg)
    sxx, syy, sxy = strain_stress(u_p, x_p, y_p, lam_base, mu_base)
    sxx, syy, sxy = sxx * E_p, syy * E_p, sxy * E_p
    div_x = grads(sxx, x_p) + grads(sxy, y_p)
    div_y = grads(sxy, x_p) + grads(syy, y_p)
    fx, fy = body_force(x_p, y_p, t_p, rho_p, cfg)
    loss_pde = torch.mean((rho_p * ux_tt - div_x - fx) ** 2 + (rho_p * uy_tt - div_y - fy) ** 2)

    x_i, y_i, t_i = sample_ic(N_ic, Lx, Ly, device)
    x_i.requires_grad_(True); y_i.requires_grad_(True); t_i.requires_grad_(True)
    u_i = model(x_i, y_i, t_i)
    ut_i = grads(u_i[:, 0:1], t_i), grads(u_i[:, 1:2], t_i)
    loss_ic = torch.mean(u_i ** 2) + torch.mean(ut_i[0] ** 2 + (ut_i[1] - v0_y) ** 2)

    x_b, y_b, t_b = sample_bc_bottom(N_bc, Lx, T_final, device)
    x_b.requires_grad_(True); y_b.requires_grad_(True); t_b.requires_grad_(True)
    loss_bc = torch.mean(model(x_b, y_b, t_b) ** 2)

    IC_W, BC_W = t_cfg.get("ic_weight", 1.0), t_cfg.get("bc_weight", 1.0)
    return loss_pde + IC_W * loss_ic + BC_W * loss_bc, loss_pde.detach(), loss_ic.detach(), loss_bc.detach()


# =========================
# 5. Visualization (train)
# =========================

def show_train_result(history):
    if not history:
        return
    plt.figure(figsize=(8, 4))
    plt.plot(history, linewidth=1)
    plt.yscale("log")
    plt.xlabel("Epoch"); plt.ylabel("Total Loss")
    plt.title("Training Loss")
    plt.grid(alpha=0.3); plt.tight_layout(); plt.show()


# =========================
# 6. Train entry
# =========================

def train(data_path=DATA_PATH, model_path=MODEL_PATH, epochs=None, lr=None, device_str="cpu"):
    t0 = time.time()
    device = get_device()
    if not torch.cuda.is_available() and device_str.endswith("cuda"):
        pass  # get_device() already handles priority detection
    print(f"device={device}")

    cfg = load_or_default_config(data_path)
    t_cfg = cfg["training"]
    if epochs is None:
        epochs = t_cfg.get("epochs_quick", 2000)
    if lr is None:
        lr = t_cfg.get("lr", 1e-3)

    p_cfg = cfg["physics"]; m_cfg = cfg["model"]
    model = PINN2D(
        in_dim=m_cfg.get("in_dim", 3), out_dim=m_cfg.get("out_dim", 2),
        width=m_cfg.get("width", 64), depth=m_cfg.get("depth", 5),
        use_symmetry=m_cfg.get("use_symmetry", False),
        Lx=p_cfg["Lx"], Ly=p_cfg["Ly"], T_final=p_cfg["T_final"],
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    step_size = t_cfg.get("scheduler_step", 2000)
    gamma = t_cfg.get("scheduler_gamma", 0.5)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)
    history = []

    print(f"Training {epochs} epochs (Lx={p_cfg['Lx']}, Ly={p_cfg['Ly']}, T={p_cfg['T_final']})...")
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        loss, lp, li, lb = compute_loss(model, cfg, device)
        loss.backward(); opt.step(); sched.step()
        history.append(float(loss.detach().cpu()))
        if ep == 1 or ep % 200 == 0 or ep == epochs:
            print(f"Epoch {ep:5d}/{epochs} | total={loss.item():.3e} | PDE={lp.item():.3e} IC={li.item():.3e} BC={lb.item():.3e} | lr={sched.get_last_lr()[0]:.1e}")

    model_path = os.path.expanduser(model_path)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    checkpoint = {
        "project": "Smartphone_DropImpact_PINN",
        "state_dict": model.state_dict(),
        "config": {
            "Lx": p_cfg["Lx"], "Ly": p_cfg["Ly"], "T_final": p_cfg["T_final"],
            "E_phys0": p_cfg.get("E_phys0", 7.0e10), "nu": p_cfg.get("nu", 0.30),
            "width": m_cfg.get("width", 64), "depth": m_cfg.get("depth", 5),
            "use_symmetry": m_cfg.get("use_symmetry", False),
        },
        "history": history,
    }
    torch.save(checkpoint, model_path)

    print("Training finished")
    print(f"model saved to: {model_path}")
    print(f"elapsed: {time.time() - t0:.1f}s")
    print(f"final loss: {history[-1]:.6e}")

    show_train_result(history)
    return {"model": model, "model_path": model_path, "history": history}


train_result = train()
