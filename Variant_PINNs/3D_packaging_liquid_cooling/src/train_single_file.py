"""
Notebook-friendly train file for 3D_packaging_liquid_cooling.

Online Jupyter usage:
1. (Optional) Upload data_3d_liquid_cooling_simul.csv as a public asset.
2. Fill DATA_HASH and MODEL_HASH.
3. Run this whole file/cell. It saves trained_model_3d_liquid_cooling.pt to
   ~/minio/Resource/{MODEL_HASH}/ for later public-asset upload.

Physics: 3D PINN for solid-fluid liquid cooling packaging.
         Solid: K_s·∇²T + Q_chip = 0
         Fluid: ρCp·u·∂T/∂x = K_f·∇²T
         Dual-network with interface continuity constraints.

Self-contained — data is auto-generated if .csv not found.
"""

import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = "97efdb436dc14d8da69b83ed38938775"
MODEL_HASH = "97efdb436dc14d8da69b83ed38938775"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/minio/Resource/{MODEL_HASH}/trained_model_3d_liquid_cooling.pt")

K_SOLID = 150.0; K_FLUID = 0.61
RHO_FLUID = 1000.0; CP_FLUID = 4186.0
T_IN = 298.0; Q_CHIP = 1.0e6

W_PHY = 1.0; W_INT = 50.0; W_BC = 20.0; W_DATA = 10.0
TRAIN_EPOCHS = 100; LR = 1e-3; SEED = 42


# =========================
# 2. Device detection (GPU → NPU → CPU)
# =========================

def get_device():
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
# 3. Model definition
# =========================

class PINN3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )
    def forward(self, x, y, z):
        return self.net(torch.cat([x, y, z], dim=1))


def grad(u, x):
    return torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]


def laplace(u, x, y, z):
    ux = grad(u, x); uy = grad(u, y); uz = grad(u, z)
    return grad(ux, x) + grad(uy, y) + grad(uz, z)


# =========================
# 4. Data generation / loading
# =========================

def generate_simul_data(path=None, n=300):
    rng = np.random.default_rng(SEED)
    def sample(n_s, label, y_range):
        x = rng.random(n_s); y = rng.uniform(*y_range, n_s); z = rng.random(n_s)
        temp = T_IN + 20.0 * np.exp(-((y - 0.2) ** 2) * 10.0)
        return pd.DataFrame({"x": x, "y": y, "z": z, "T": temp, "label": label})
    df = pd.concat([
        sample(n, 0, (0.5, 1.0)), sample(n, 1, (0.0, 0.5)),
        sample(n // 5, 2, (0.45, 0.55)),
        sample(n // 5, 3, (0.0, 0.05)),
        sample(n // 5, 4, (0.9, 1.0)),
    ], ignore_index=True)
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
    return df


def load_data(data_path=None, device=None):
    data_path = os.path.expanduser(data_path) if data_path else ""
    if data_path and os.path.isfile(data_path):
        df = pd.read_csv(data_path)
    elif data_path and os.path.isdir(data_path):
        for f in sorted(os.listdir(data_path)):
            if f.endswith('.csv'):
                df = pd.read_csv(os.path.join(data_path, f)); break
        else:
            df = generate_simul_data()
    else:
        print("Generating simulated data...")
        df = generate_simul_data()
    data = {}
    for lbl in sorted(df["label"].unique()):
        sub = df[df["label"] == lbl]
        x = torch.tensor(sub["x"].values[:, None], dtype=torch.float32, device=device)
        y = torch.tensor(sub["y"].values[:, None], dtype=torch.float32, device=device)
        z = torch.tensor(sub["z"].values[:, None], dtype=torch.float32, device=device)
        x.requires_grad_(True); y.requires_grad_(True); z.requires_grad_(True)
        temp = torch.tensor(sub["T"].values[:, None], dtype=torch.float32, device=device)
        data[int(lbl)] = (x, y, z, temp)
    return data


# =========================
# 5. Physics losses
# =========================

def physics_solid(model_s, x, y, z):
    return K_SOLID * laplace(model_s(x, y, z), x, y, z) + Q_CHIP


def physics_fluid(model_f, x, y, z, u=1.0):
    T = model_f(x, y, z)
    return RHO_FLUID * CP_FLUID * (u * grad(T, x)) - K_FLUID * laplace(T, x, y, z)


def interface_loss(model_s, model_f, x, y, z):
    Ts = model_s(x, y, z); Tf = model_f(x, y, z)
    loss_T = torch.mean((Ts - Tf) ** 2)
    loss_flux = torch.mean((K_SOLID * grad(Ts, y) - K_FLUID * grad(Tf, y)) ** 2)
    return loss_T + loss_flux


def boundary_loss(model, x, y, z, bc_type):
    T = model(x, y, z)
    if bc_type == "inlet": return torch.mean((T - T_IN) ** 2)
    if bc_type == "heat": return torch.mean((-K_SOLID * grad(T, y) - Q_CHIP) ** 2)
    return torch.tensor(0.0, device=T.device)


# =========================
# 6. Visualization
# =========================

def show_result(model_s, data, history):
    if history:
        plt.figure(figsize=(7, 4)); plt.semilogy(history); plt.xlabel("Epoch")
        plt.ylabel("Loss"); plt.title("Training Loss"); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.show()

    model_s.eval()
    x, y, z, temp_true = data[0]
    with torch.no_grad():
        temp_pred = model_s(x, y, z)
    rmse = torch.sqrt(torch.mean((temp_pred - temp_true)**2)).item()

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x.detach().cpu().numpy().reshape(-1), y.detach().cpu().numpy().reshape(-1),
                    z.detach().cpu().numpy().reshape(-1), c=temp_pred.detach().cpu().numpy().reshape(-1),
                    cmap="jet", s=6)
    fig.colorbar(sc, label="T (K)");
    ax.set_title("3D Liquid Cooling Temperature Field"); plt.show()
    print(f"RMSE: {rmse:.4f} K")
    return rmse


# =========================
# 7. Train entry
# =========================

def train(data_path=DATA_PATH, model_path=MODEL_PATH, epochs=TRAIN_EPOCHS, lr=LR):
    t0 = time.time()
    device = get_device(); torch.manual_seed(SEED)
    print(f"device={device}")

    data = load_data(data_path, device=device)
    model_s = PINN3D().to(device); model_f = PINN3D().to(device)
    optim = torch.optim.Adam(list(model_s.parameters()) + list(model_f.parameters()), lr=lr)
    history = []

    print(f"Training {epochs} epochs...")
    for ep in range(1, epochs + 1):
        optim.zero_grad()
        xs, ys, zs, ts = data[0]; xf, yf, zf, _ = data[1]
        xi, yi, zi, _ = data[2]; xin, yin, zin, _ = data[3]
        xh, yh, zh, _ = data[4]

        loss_phy = torch.mean(physics_solid(model_s, xs, ys, zs)**2) \
                 + torch.mean(physics_fluid(model_f, xf, yf, zf)**2)
        loss_int = interface_loss(model_s, model_f, xi, yi, zi)
        loss_bc = boundary_loss(model_f, xin, yin, zin, "inlet") \
                + boundary_loss(model_s, xh, yh, zh, "heat")
        loss_data = nn.MSELoss()(model_s(xs, ys, zs), ts)
        loss = W_PHY * loss_phy + W_INT * loss_int + W_BC * loss_bc + W_DATA * loss_data

        loss.backward(); optim.step()
        history.append(float(loss.detach().cpu()))
        if ep == 1 or ep % 50 == 0 or ep == epochs:
            print(f"Epoch {ep:4d}/{epochs} | loss={loss.item():.3e}")

    model_path = os.path.expanduser(model_path)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save({"solid": model_s.state_dict(), "fluid": model_f.state_dict(), "history": history}, model_path)

    print("Training finished"); print(f"model saved to: {model_path}")
    print(f"elapsed: {time.time() - t0:.1f}s")
    show_result(model_s, data, history)
    return {"model_s": model_s, "model_f": model_f, "model_path": model_path, "history": history}


train_result = train()
