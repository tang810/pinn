"""
Notebook-friendly train file for mixed_convection (MCF PINN).

Online Jupyter usage:
1. Fill DATA_HASH (optional) and MODEL_HASH.
2. Run this whole file/cell. It saves trained_model_mcf.pt to
   ~/minio/Resource/{MODEL_HASH}/ for later public-asset upload.

Physics: 2D mixed convection in a cavity (Re=2000, Pr=0.71, Ri=0.1).
         PINN learns (u,v,p,T) from collocation points with NS + energy PDEs.

Self-contained — generates collocation points on the fly.
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

DATA_HASH = "d0e5d8636a5d410faee8f982a4496740"
MODEL_HASH = "d0e5d8636a5d410faee8f982a4496740"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/minio/Resource/{MODEL_HASH}/trained_model_mcf.pt")

Re = 2000.0; Pr = 0.71; Ri = 0.1
HIDDEN = 80; LAYERS = 4
EPOCHS = 2000; LR = 1e-3; PRINT_EVERY = 200
N_DOMAIN = 800; N_BC = 200
LAM_BC = 10.0; LAM_EQ = 1.0


# =========================
# 2. Model (FCNet — 2 inputs x,y -> 4 outputs u,v,p,T)
# =========================

class FCNet(nn.Module):
    def __init__(self, in_dim=2, out_dim=4, hidden=80, layers=4):
        super().__init__()
        seq = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            seq += [nn.Linear(hidden, hidden), nn.Tanh()]
        seq += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*seq)

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))


def _grad(u, v):
    return torch.autograd.grad(u, v, torch.ones_like(u), create_graph=True, retain_graph=True)[0]


# =========================
# 3. Physics losses (simplified mixed convection)
# =========================

def compute_loss(model, x, y, x_bc, y_bc, u_bc, v_bc, T_bc_hot, T_bc_cold):
    x.requires_grad_(True); y.requires_grad_(True)
    out = model(x, y)
    u, v, p, T = out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4]

    # Gradients
    ux = _grad(u, x); uy = _grad(u, y)
    vx = _grad(v, x); vy = _grad(v, y)
    px = _grad(p, x); py = _grad(p, y)
    Tx = _grad(T, x); Ty = _grad(T, y)

    uxx = _grad(ux, x); uyy = _grad(uy, y)
    vxx = _grad(vx, x); vyy = _grad(vy, y)
    Txx = _grad(Tx, x); Tyy = _grad(Ty, y)

    # Momentum: u·∇u = -∇p + (1/Re)∇²u + Ri·T·e_y
    mom_x = u*ux + v*uy + px - (1.0/Re)*(uxx + uyy)
    mom_y = u*vx + v*vy + py - (1.0/Re)*(vxx + vyy) - Ri*T

    # Continuity
    cont = ux + vy

    # Energy: u·∇T = (1/(Re·Pr))∇²T
    energy = u*Tx + v*Ty - (1.0/(Re*Pr))*(Txx + Tyy)

    loss_pde = torch.mean(mom_x**2) + torch.mean(mom_y**2) + torch.mean(cont**2) + torch.mean(energy**2)

    # BC losses
    out_bc = model(x_bc, y_bc)
    u_b, v_b, p_b, T_b = out_bc[:, 0:1], out_bc[:, 1:2], out_bc[:, 2:3], out_bc[:, 3:4]

    loss_bc = torch.mean((u_b - u_bc)**2) + torch.mean((v_b - v_bc)**2)
    mask_top = (y_bc > 0.99).squeeze()
    mask_bot = (y_bc < 0.01).squeeze()
    if mask_top.any(): loss_bc += torch.mean((T_b[mask_top] - T_bc_cold)**2)
    if mask_bot.any(): loss_bc += torch.mean((T_b[mask_bot] - T_bc_hot)**2)

    return LAM_EQ * loss_pde + LAM_BC * loss_bc, loss_pde, loss_bc


# =========================
# 4. Data sampling
# =========================

def sample_points(device):
    # Interior domain [0,1]x[0,1]
    x = torch.rand(N_DOMAIN, 1, device=device)
    y = torch.rand(N_DOMAIN, 1, device=device)

    # Boundary: all 4 walls
    tb = N_BC // 4
    x_left = torch.zeros(tb, 1, device=device);     y_left = torch.rand(tb, 1, device=device)
    x_right = torch.ones(tb, 1, device=device);     y_right = torch.rand(tb, 1, device=device)
    x_bot = torch.rand(tb, 1, device=device);       y_bot = torch.zeros(tb, 1, device=device)
    x_top = torch.rand(tb, 1, device=device);       y_top = torch.ones(tb, 1, device=device)

    x_bc = torch.cat([x_left, x_right, x_bot, x_top], dim=0)
    y_bc = torch.cat([y_left, y_right, y_bot, y_top], dim=0)

    # BC values: no-slip walls, T=1 bottom, T=0 top
    u_bc = torch.zeros_like(x_bc)
    v_bc = torch.zeros_like(x_bc)
    T_bc_hot = torch.tensor(1.0, device=device)
    T_bc_cold = torch.tensor(0.0, device=device)

    return x, y, x_bc, y_bc, u_bc, v_bc, T_bc_hot, T_bc_cold


# =========================
# 5. Visualization
# =========================

def show_result(model, history, device):
    if history:
        plt.figure(figsize=(7,4)); plt.semilogy(history); plt.xlabel("Epoch")
        plt.ylabel("Total Loss"); plt.title("Training Loss"); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.show()

    x = torch.linspace(0, 1, 50, device=device)
    y = torch.linspace(0, 1, 50, device=device)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    xf = X.reshape(-1, 1); yf = Y.reshape(-1, 1)
    model.eval()
    with torch.no_grad():
        out = model(xf, yf).cpu().numpy()
    U = out[:, 0].reshape(50, 50)
    T = out[:, 3].reshape(50, 50)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    im1 = ax1.contourf(x.cpu(), y.cpu(), U.T, levels=30, cmap="RdBu_r")
    ax1.set_title("u-velocity"); ax1.set_xlabel("x"); ax1.set_ylabel("y")
    plt.colorbar(im1, ax=ax1)
    im2 = ax2.contourf(x.cpu(), y.cpu(), T.T, levels=30, cmap="hot")
    ax2.set_title("Temperature"); ax2.set_xlabel("x"); ax2.set_ylabel("y")
    plt.colorbar(im2, ax=ax2)
    plt.suptitle(f"Mixed Convection (Re={int(Re)}, Pr={Pr}, Ri={Ri})", fontsize=13)
    plt.tight_layout(); plt.show()


# =========================
# 6. Train entry
# =========================

def train(data_path=DATA_PATH, model_path=MODEL_PATH, re=Re, pr=Pr, ri=Ri,
          hidden=HIDDEN, layers=LAYERS, epochs=EPOCHS, lr=LR):
    t0 = time.time(); torch.manual_seed(42)
    device = get_device(); print(f"device={device}")

    model = FCNet(in_dim=2, out_dim=4, hidden=hidden, layers=layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    print(f"Training {epochs} epochs (Re={re}, Pr={pr}, Ri={ri})...")
    for ep in range(1, epochs + 1):
        x, y, xb, yb, ub, vb, Th, Tc = sample_points(device)
        loss, lp, lb = compute_loss(model, x, y, xb, yb, ub, vb, Th, Tc)
        opt.zero_grad(); loss.backward(); opt.step()
        history.append(float(loss.detach().cpu()))
        if ep == 1 or ep % PRINT_EVERY == 0 or ep == epochs:
            print(f"Epoch {ep:5d}/{epochs} | total={history[-1]:.3e} pde={lp.item():.3e} bc={lb.item():.3e}")

    model_path = os.path.expanduser(model_path)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "history": history,
                "hidden": hidden, "layers": layers, "re": re, "pr": pr, "ri": ri}, model_path)

    print("Training finished"); print(f"model saved to: {model_path}")
    print(f"elapsed: {time.time() - t0:.1f}s"); print(f"final loss: {history[-1]:.6e}")
    show_result(model, history, device)
    return {"model": model, "model_path": model_path, "history": history}


train_result = train()
