"""
Notebook-friendly eval file for kdv_operator_learning.

Online Jupyter usage:
1. Upload trained_model_kdv_pinn.pt as a public asset.
2. Upload real_data.npz as a public asset (optional, virtual data used if missing).
3. Fill MODEL_HASH and DATA_HASH, then run this whole file/cell.

This file loads the KdV PINN model, evaluates on soliton profiles,
and shows figures with plt.show(). It does not save images.

Self-contained — only depends on .npz dataset and .pt model.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = "b653b27230f64aca84000bcd9290e824"
MODEL_HASH = "a0fdaff3b4524395945748b357e33494"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/trained_model_kdv_pinn.pt")


# =========================
# 2. KdV data generation
# =========================

def generate_virtual_data(n_samples=80, n_x=256):
    min_x, max_x = -40, 40
    x = np.linspace(min_x, max_x, n_x, dtype=np.float32).reshape(1, -1)
    x = np.repeat(x, n_samples, axis=0)
    rho = np.random.uniform(0.9, 0.99, size=(n_samples, 1)).astype(np.float32)
    h1 = np.random.uniform(4, 10, size=(n_samples, 1)).astype(np.float32)
    a = np.random.uniform(-0.4, -0.05, size=(n_samples, 1)).astype(np.float32)
    h2 = 1.0; g = 9.81; h = h2 / h1
    c0 = np.sqrt((g * h2 * (1 - rho)) / (h + rho))
    c1 = -1.5 * c0 * ((rho - h**2) / (rho * h2 + h * h2))
    c2 = (1.0/6.0) * c0 * ((rho * h2**2 * h1 + h1**2 * h2) / (rho * h1 + h2))
    lam = np.sqrt(12 * c2 / (a * c1))
    ref = a / (np.cosh(x / lam) ** 2)
    return x.astype(np.float32), ref.astype(np.float32)


def load_data(data_path, n_samples=80, n_x=256):
    data_path = os.path.expanduser(data_path) if data_path else ""
    if data_path and os.path.isfile(data_path):
        d = np.load(data_path)
        if 'x' in d and 'ref' in d:
            return d['x'].astype(np.float32), d['ref'].astype(np.float32)
    return generate_virtual_data(n_samples=n_samples, n_x=n_x)


# =========================
# 3. Model
# =========================

class KdV_PINN(nn.Module):
    def __init__(self, n_input=1, n_output=1, n_hidden=50, n_layers=3):
        super().__init__()
        layers = [nn.Linear(n_input, n_hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.Tanh()]
        layers += [nn.Linear(n_hidden, n_output)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


def load_model_package(model_path, device):
    model_path = os.path.expanduser(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict") or ckpt.get("model") or ckpt
    n_hidden = ckpt.get("n_hidden", 50) if isinstance(ckpt, dict) else 50
    n_layers = ckpt.get("n_layers", 3) if isinstance(ckpt, dict) else 3
    model = KdV_PINN(n_hidden=n_hidden, n_layers=n_layers).to(device)
    model.load_state_dict(state); model.eval()
    return model, ckpt.get("history", []) if isinstance(ckpt, dict) else []


# =========================
# 4. Visualization
# =========================

def show_eval_result(model, x_data, ref_data, history, device):
    if history:
        plt.figure(figsize=(7, 4)); plt.plot(history); plt.yscale("log")
        plt.xlabel("Epoch"); plt.ylabel("MSE Loss"); plt.title("Training Loss")
        plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

    # Show first 6 samples
    n_show = min(6, x_data.shape[0])
    n_cols = 3; n_rows = (n_show + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    with torch.no_grad():
        for i in range(n_show):
            x_i = torch.tensor(x_data[i:i+1].reshape(-1, 1), dtype=torch.float32, device=device)
            pred = model(x_i).cpu().numpy().reshape(-1)
            axes[i].plot(x_data[i], ref_data[i], 'k-', lw=2, label="Ref")
            axes[i].plot(x_data[i], pred, 'r--', lw=1.5, label="Pred")
            axes[i].set_xlabel("x"); axes[i].set_ylabel("u(x)")
            axes[i].set_title(f"Sample {i+1}"); axes[i].legend(fontsize=6); axes[i].grid(alpha=0.3)
    for j in range(n_show, len(axes)): axes[j].axis('off')
    plt.suptitle("KdV PINN Eval — Reference vs Prediction", fontsize=14)
    plt.tight_layout(); plt.show()

    x_f = x_data.reshape(-1, 1)
    with torch.no_grad():
        pred_all = model(torch.tensor(x_f, dtype=torch.float32, device=device)).cpu().numpy()
    mse = float(np.mean((ref_data.reshape(-1) - pred_all.reshape(-1))**2))
    l2 = np.linalg.norm(ref_data.reshape(-1) - pred_all.reshape(-1)) / (np.linalg.norm(ref_data.reshape(-1)) + 1e-12)
    print(f"MSE: {mse:.6e}, Relative L2: {l2:.4e}")
    return mse, l2


# =========================
# 5. Eval entry
# =========================

def evaluate(model_path=MODEL_PATH, data_path=DATA_PATH, n_samples=20, n_x=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    model, history = load_model_package(model_path, device)
    x_data, ref_data = load_data(data_path, n_samples=n_samples, n_x=n_x)

    print("Eval result"); print(f"model_path: {model_path}")
    mse, l2 = show_eval_result(model, x_data, ref_data, history, device)
    return {"mse": mse, "relative_l2": l2}


eval_result = evaluate()
