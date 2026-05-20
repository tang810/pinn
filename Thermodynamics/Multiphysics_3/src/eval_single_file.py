"""
Notebook-friendly eval file for Multiphysics_3 (PINNsformer inverse).

Online Jupyter usage:
1. Upload trained_model_multiphysics3.pt as a public asset.
2. Upload data_7246.txt as a public asset.
3. Fill MODEL_HASH and DATA_HASH, then run this whole file/cell.

This file loads the PINNsformer model, evaluates temperature + displacement
prediction, and shows figures with plt.show(). It does not save images.

Note: imports from src/ for complex model architecture and physics.
"""

import os

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.model import PINNsformer
from src import config


# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = ""
MODEL_HASH = ""

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/trained_model_multiphysics3.pt")


# =========================
# 2. Model loading
# =========================

def load_model_package(model_path, device):
    model_path = os.path.expanduser(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict") or ckpt.get("model_state_dict")
    if state is None: raise KeyError("Checkpoint must contain state_dict or model_state_dict.")
    model = PINNsformer(d_model=config.d_model, d_hidden=config.d_hidden,
                        N=config.n_layers, heads=config.n_heads, T0=config.t_0).to(device)
    model.load_state_dict(state); model.eval()
    return model


# =========================
# 3. Evaluation
# =========================

def evaluate(model_path=MODEL_PATH, data_path=DATA_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    data_path = os.path.expanduser(data_path) if data_path else config.data_path
    model = load_model_package(model_path, device)

    df = pd.read_csv(data_path, delimiter=' ', header=None).to_numpy()
    n_eval = min(5000, len(df))
    idx = np.linspace(0, len(df) - 1, n_eval).astype(int)
    x = torch.tensor(df[idx, 0:1], dtype=torch.float32, device=device)
    y = torch.tensor(df[idx, 1:2], dtype=torch.float32, device=device)
    z = torch.tensor(df[idx, 2:3], dtype=torch.float32, device=device)
    t = torch.tensor(df[idx, 3:4], dtype=torch.float32, device=device)
    T_true = df[idx, 4]

    with torch.no_grad():
        out = model(x, y, z, t).cpu().numpy()
    T_pred = out[:, 0]

    print("Eval result"); print(f"model_path: {model_path}")
    print(f"eval points: {n_eval}, T range: [{T_true.min():.1f}, {T_true.max():.1f}]")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    sc = axes[0].scatter(T_true, T_pred, c=t.cpu().numpy().reshape(-1), s=3, alpha=0.4, cmap="plasma")
    axes[0].plot([T_true.min(), T_true.max()], [T_true.min(), T_true.max()], "r--", lw=1)
    axes[0].set_xlabel("True T (K)"); axes[0].set_ylabel("Pred T (K)"); axes[0].set_title("True vs Pred")
    axes[0].grid(alpha=0.3); plt.colorbar(sc, ax=axes[0], label="t")
    err = T_pred - T_true
    axes[1].hist(err, bins=50); axes[1].set_xlabel("Error (K)"); axes[1].set_title("Error Distribution")
    axes[2].scatter(T_true, err, s=2, alpha=0.4); axes[2].axhline(0, color='r', ls='--')
    axes[2].set_xlabel("True T (K)"); axes[2].set_ylabel("Error (K)"); axes[2].set_title("Error vs True T")
    axes[2].grid(alpha=0.3)
    plt.suptitle("Multiphysics_3 — PINNsformer Inverse Eval", fontsize=14)
    plt.tight_layout(); plt.show()

    rl2 = np.linalg.norm(err) / (np.linalg.norm(T_true) + 1e-12)
    mae = float(np.mean(np.abs(err))); rmse = float(np.sqrt(np.mean(err**2)))
    print(f"Relative L2: {rl2:.4e}, MAE: {mae:.4f} K, RMSE: {rmse:.4f} K")
    return {"rl2": rl2, "mae": mae, "rmse": rmse}


eval_result = evaluate()
