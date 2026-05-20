"""
Notebook-friendly train file for Embedded_liquid_cooling.

Online Jupyter usage:
1. Fill DATA_HASH (optional) and MODEL_HASH.
2. Run this whole file/cell. It saves trained_model_embedded_liquid_cooling.pt to
   ~/minio/Resource/{MODEL_HASH}/ for later public-asset upload.

Physics: 2D embedded liquid cooling PINN with solid-fluid conjugate heat transfer.
         Solid: k_s·∇²T + q_heat = 0
         Fluid: ρCp·(u·∂T/∂x) = k_f·∇²T
         Dual-network with interface continuity.

Self-contained — generates collocation points on the fly.
"""

import os
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = ""
MODEL_HASH = ""

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/minio/Resource/{MODEL_HASH}/trained_model_embedded_liquid_cooling.pt")
DATA_FILE_NAME = "data_embedded_liquid_cooling_simul.csv"

# Physics constants
k_s = 401.0; k_f = 0.613; rho_f = 1000.0; cp_f = 4186.0
T_in = 298.0; q_heat = 1e5
DEFAULT_EPOCHS = 300; DEFAULT_N_POINTS = 128; LR = 1e-4
HIDDEN_DIM = 32
GRID_RES = 60


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

class PINN_Net(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))


def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    elif order == 2:
        return gradients(gradients(u, x), x)
    return None


def solid_eq(model, x, y):
    T = model(x, y); Tx = gradients(T, x); Ty = gradients(T, y)
    return k_s * (gradients(Tx, x) + gradients(Ty, y)) + q_heat


def fluid_eq(model, x, y, u=0.5, v=0.0):
    T = model(x, y); Tx = gradients(T, x); Ty = gradients(T, y)
    conv = rho_f * cp_f * (u * Tx + v * Ty)
    diff = k_f * (gradients(Tx, x) + gradients(Ty, y))
    return conv - diff


def heat_source_loss(model, x, y):
    T = model(x, y); nT = gradients(T, y)
    return torch.mean((-k_s * nT - q_heat) ** 2)


def inlet_loss(model, x, y):
    return torch.mean((model(x, y) - T_in) ** 2)


def interface_loss(model_s, model_f, x, y):
    Ts = model_s(x, y); Tf = model_f(x, y)
    loss_T = torch.mean((Ts - Tf) ** 2)
    qs = k_s * gradients(Ts, y); qf = k_f * gradients(Tf, y)
    return loss_T + torch.mean((qs - qf) ** 2)


# =========================
# 4. Data sampling
# =========================

def sample_points(n, device):
    x = torch.rand(n, 1, device=device)
    y = torch.rand(n, 1, device=device)
    x.requires_grad_(True); y.requires_grad_(True)
    return x, y


# =========================
# 5. Visualization
# =========================

def show_result(model_s, model_f, history, device):
    if history:
        plt.figure(figsize=(7,4)); plt.semilogy(history)
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training Loss")
        plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

    x = np.linspace(0, 1, GRID_RES); y = np.linspace(0, 1, GRID_RES)
    X, Y = np.meshgrid(x, y)
    xt = torch.tensor(X.reshape(-1, 1), dtype=torch.float32, device=device)
    yt = torch.tensor(Y.reshape(-1, 1), dtype=torch.float32, device=device)
    model_s.eval(); model_f.eval()
    with torch.no_grad():
        Ts = model_s(xt, yt).cpu().numpy().reshape(GRID_RES, GRID_RES)
        Tf = model_f(xt, yt).cpu().numpy().reshape(GRID_RES, GRID_RES)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, val, title in [(axes[0], Ts, "Solid"), (axes[1], Tf, "Fluid")]:
        im = ax.contourf(X, Y, val, levels=30, cmap="jet")
        ax.set_title(f"{title} Thermal Field"); ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.colorbar(im, ax=ax)
    fig.suptitle("Embedded Liquid Cooling PINN", fontsize=13)
    plt.tight_layout(); plt.show()


# =========================
# 6. Train entry
# =========================

def train(data_path=DATA_PATH, model_path=MODEL_PATH, epochs=DEFAULT_EPOCHS,
          n_points=DEFAULT_N_POINTS, lr=LR):
    t0 = time.time()
    device = get_device(); torch.manual_seed(42)
    print(f"device={device}")

    model_s = PINN_Net(hidden_dim=HIDDEN_DIM).to(device)
    model_f = PINN_Net(hidden_dim=HIDDEN_DIM).to(device)
    opt = torch.optim.Adam(list(model_s.parameters()) + list(model_f.parameters()), lr=lr)
    history = []

    print(f"Training {epochs} epochs...")
    for ep in range(epochs):
        opt.zero_grad()
        xs, ys = sample_points(n_points, device)
        xf, yf = sample_points(n_points, device)
        xi, yi = sample_points(max(64, n_points // 3), device)

        loss_phys = torch.mean(solid_eq(model_s, xs, ys)**2) + torch.mean(fluid_eq(model_f, xf, yf)**2)
        loss_bc = heat_source_loss(model_s, xs, ys) + inlet_loss(model_f, xf, yf)
        loss_int = interface_loss(model_s, model_f, xi, yi)
        loss = loss_phys + 10 * loss_bc + 10 * loss_int
        loss.backward(); opt.step()
        history.append(float(loss.detach().cpu()))
        if ep == 0 or ep % 50 == 0 or ep == epochs - 1:
            print(f"Epoch {ep:4d}/{epochs} | loss={loss.item():.4e}")

    model_path = os.path.expanduser(model_path)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save({
        "model_s": model_s.state_dict(),
        "model_f": model_f.state_dict(),
        "history": history,
        "hidden_dim": HIDDEN_DIM,
        "epochs": epochs,
        "n_points": n_points,
    }, model_path)

    print("Training finished"); print(f"model saved to: {model_path}")
    print(f"elapsed: {time.time() - t0:.1f}s"); print(f"final loss: {history[-1]:.6e}")
    show_result(model_s, model_f, history, device)
    return {"model_s": model_s, "model_f": model_f, "model_path": model_path, "history": history}


train_result = train()
