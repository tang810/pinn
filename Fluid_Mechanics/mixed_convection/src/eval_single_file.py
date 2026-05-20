"""
Notebook-friendly eval file for mixed_convection (MCF PINN).

Online Jupyter usage:
1. Upload trained_model_mcf.pt as a public asset.
2. Fill MODEL_HASH, then run this whole file/cell.

This file loads the MCF PINN, evaluates velocity/temperature fields,
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

MODEL_HASH = "a6b58d5849a546c987b2ade8499ae1a7"

MODEL_PATH = os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/trained_model_mcf.pt")


# =========================
# 2. Model
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


def load_model(model_path, device):
    model_path = os.path.expanduser(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict") or ckpt
    hidden = ckpt.get("hidden", 80) if isinstance(ckpt, dict) else 80
    layers = ckpt.get("layers", 4) if isinstance(ckpt, dict) else 4
    model = FCNet(in_dim=2, out_dim=4, hidden=hidden, layers=layers).to(device)
    model.load_state_dict(state); model.eval()
    return model, ckpt


# =========================
# 3. Visualization
# =========================

def show_eval_result(model, device, history):
    if history:
        plt.figure(figsize=(7,4)); plt.semilogy(history); plt.xlabel("Epoch")
        plt.ylabel("Total Loss"); plt.title("Training Loss"); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.show()

    x = torch.linspace(0, 1, 50, device=device)
    y = torch.linspace(0, 1, 50, device=device)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    xf, yf = X.reshape(-1, 1), Y.reshape(-1, 1)
    with torch.no_grad():
        out = model(xf, yf).cpu().numpy()
    U = out[:, 0].reshape(50, 50); V = out[:, 1].reshape(50, 50)
    P = out[:, 2].reshape(50, 50); T = out[:, 3].reshape(50, 50)
    speed = np.sqrt(U**2 + V**2)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    titles = ["Speed", "Pressure", "Temperature", "u-velocity"]
    fields = [speed, P, T, U]
    cmaps = ["viridis", "RdBu_r", "hot", "RdBu_r"]
    for ax, fld, t, cm in zip(axes, fields, titles, cmaps):
        im = ax.contourf(x.cpu(), y.cpu(), fld.T, levels=30, cmap=cm)
        ax.set_title(t); ax.set_xlabel("x"); ax.set_ylabel("y")
        plt.colorbar(im, ax=ax)
    plt.suptitle("Mixed Convection PINN Eval", fontsize=14)
    plt.tight_layout(); plt.show()


# =========================
# 4. Eval entry
# =========================

def evaluate(model_path=MODEL_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    model, ckpt = load_model(model_path, device)
    history = ckpt.get("history", []) if isinstance(ckpt, dict) else []

    print("Eval result"); print(f"model_path: {model_path}")
    show_eval_result(model, device, history)
    return {"ok": True}


eval_result = evaluate()
