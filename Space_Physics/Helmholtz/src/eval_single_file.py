"""
Notebook-friendly eval file for Helmholtz PINN.

Online Jupyter usage:
1. Upload trained_model_helmholtz.pt as a public asset.
2. Fill MODEL_HASH, then run this whole file/cell.

This file loads the Helmholtz PINN model, predicts u(x) on [-1,1],
and shows figures with plt.show(). It does not save images.

Self-contained — generates evaluation grid on the fly.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


# =========================
# 1. Parameters to edit
# =========================

MODEL_HASH = "0d395a3f2e1e44a196fa9c27feb5d4e4"

MODEL_PATH = os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/trained_model_helmholtz.pt")


# =========================
# 2. Model
# =========================

class HelmholtzPINN(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=4, k=3.0):
        super().__init__()
        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, 1)]
        self.net = nn.Sequential(*layers)
        self.k = nn.Parameter(torch.tensor(float(k)))

    def forward(self, x):
        return self.net(x)


def load_model_package(model_path, device):
    model_path = os.path.expanduser(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict") or ckpt
    k_val = float(ckpt.get("k", 3.0))
    hidden_dim = int(ckpt.get("hidden_dim", 64))
    num_layers = int(ckpt.get("num_layers", 4))
    model = HelmholtzPINN(hidden_dim=hidden_dim, num_layers=num_layers, k=k_val).to(device)
    model.load_state_dict(state); model.eval()
    return model, k_val


# =========================
# 3. Visualization
# =========================

def show_eval_result(model, device):
    x = torch.linspace(-1, 1, 300, device=device).reshape(-1, 1)
    model.eval()
    with torch.no_grad():
        u = model(x).cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax1.plot(x.cpu().numpy(), u, 'b-', lw=2)
    ax1.scatter([-1, 1], [0, 1], c='r', s=60, zorder=5, label="BCs")
    ax1.set_xlabel("x"); ax1.set_ylabel("u(x)"); ax1.set_title(f"Helmholtz PINN (k={model.k.item():.4f})")
    ax1.legend(); ax1.grid(alpha=0.3)

    # Compute PDE residual along x
    xr = x.clone().detach().requires_grad_(True)
    u_r = model(xr)
    du = torch.autograd.grad(u_r, xr, torch.ones_like(u_r), create_graph=True, retain_graph=True)[0]
    d2u = torch.autograd.grad(du, xr, torch.ones_like(du), create_graph=True, retain_graph=True)[0]
    residual = (d2u + model.k**2 * u_r).detach().cpu().numpy()
    ax2.plot(x.cpu().numpy(), residual.reshape(-1), 'r-', lw=1.5)
    ax2.set_xlabel("x"); ax2.set_ylabel("PDE residual"); ax2.set_title("Helmholtz PDE Residual")
    ax2.grid(alpha=0.3); ax2.axhline(0, color='k', ls='--', lw=0.7)

    plt.suptitle(f"Helmholtz PINN Eval — Identified k={model.k.item():.6f}", fontsize=14)
    plt.tight_layout(); plt.show()

    print(f"Identified k = {model.k.item():.6f}")
    print(f"PDE residual: mean={np.mean(np.abs(residual)):.4e}, max={np.max(np.abs(residual)):.4e}")


# =========================
# 4. Eval entry
# =========================

def evaluate(model_path=MODEL_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    model, k_val = load_model_package(model_path, device)

    print("Eval result"); print(f"model_path: {model_path}")
    show_eval_result(model, device)
    return {"k": model.k.item()}


eval_result = evaluate()
