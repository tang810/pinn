"""
Notebook-friendly eval file for Multiphysics (PINNsformer).

Online Jupyter usage:
1. Upload trained_model_multiphysics.pt as a public asset.
2. Upload data_7246.txt as a public asset.
3. Fill MODEL_HASH and DATA_HASH, then run this whole file/cell.

This file loads the PINNsformer model, evaluates temperature prediction,
and shows figures with plt.show(). It does not save images.

Note: imports from src/ for complex model architecture.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from src.model import PINNsformer


# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = ""
MODEL_HASH = ""

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/trained_model_multiphysics.pt")


# =========================
# 2. Model loading
# =========================

def load_model_package(model_path, device):
    model_path = os.path.expanduser(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict") if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        d_model = ckpt.get("d_model", 128); d_hidden = ckpt.get("d_hidden", 128)
        N = ckpt.get("N", 1); heads = ckpt.get("heads", 2)
    else:
        d_model = state["linear_emb.weight"].shape[0]
        d_hidden = state["linear_out.0.weight"].shape[0]
        N = 1; heads = 2
    history = ckpt.get("history", []) if isinstance(ckpt, dict) else []
    model = PINNsformer(d_model=d_model, d_hidden=d_hidden, N=N, heads=heads).to(device)
    model.load_state_dict(state); model.eval()
    return model, history


# =========================
# 3. Data loading
# =========================

def load_eval_data(data_path):
    data_path = os.path.expanduser(data_path) if data_path else "data/data_7246.txt"
    if not os.path.isfile(data_path):
        if os.path.isdir(data_path):
            data_path = os.path.join(data_path, "data_7246.txt")
    if not os.path.isfile(data_path):
        print("Generating small synthetic dataset...")
        rng = np.random.default_rng(42); n = 2000
        x = rng.uniform(0, 1, n); y = rng.uniform(0, 1, n)
        z = rng.uniform(0, 1, n); t = rng.uniform(0, 1, n)
        T = 273.15 + 30*np.exp(-((x-0.5)**2+(y-0.5)**2+(z-0.5)**2)/0.1)*(1+0.1*np.sin(2*np.pi*t))
        df = pd.DataFrame({"x": x, "y": y, "z": z, "t": t, "T": T})
        os.makedirs(os.path.dirname(data_path) if os.path.dirname(data_path) else ".", exist_ok=True)
        df.to_csv(data_path, sep=" ", index=False, header=False, float_format="%.6f")
        print(f"Generated: {len(df)} points, {os.path.getsize(data_path)/1024:.0f} KB")
    df = pd.read_csv(data_path, sep=r"\s+", header=None,
                     names=["x", "y", "z", "t", "T"], dtype=np.float32)
    x = df["x"].values.reshape(-1, 1); y = df["y"].values.reshape(-1, 1)
    z = df["z"].values.reshape(-1, 1); t = df["t"].values.reshape(-1, 1)
    T = df["T"].values.reshape(-1, 1)
    return x, y, z, t, T


# =========================
# 4. Visualization
# =========================

def show_eval_result(model, device, x, y, z, t, T_true, history):
    if history:
        plt.figure(figsize=(7,4)); plt.plot(history); plt.yscale("log")
        plt.xlabel("Epoch"); plt.ylabel("Total Loss"); plt.title("Training Loss")
        plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

    # Subsample for prediction
    n_eval = min(5000, len(x))
    idx = np.linspace(0, len(x) - 1, n_eval).astype(int)
    x_e, y_e, z_e, t_e, T_e = x[idx], y[idx], z[idx], t[idx], T_true[idx]

    with torch.no_grad():
        xt = torch.tensor(np.hstack([x_e, y_e, z_e, t_e]), dtype=torch.float32, device=device)
        T_pred = model(xt).cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    err = np.abs(T_e - T_pred)

    sc = axes[0].scatter(T_e, T_pred, c=t_e.reshape(-1), s=4, alpha=0.5, cmap="plasma")
    axes[0].plot([T_e.min(), T_e.max()], [T_e.min(), T_e.max()], "r--", lw=1)
    axes[0].set_xlabel("True T (K)"); axes[0].set_ylabel("Pred T (K)")
    axes[0].set_title("True vs Pred"); axes[0].grid(alpha=0.3)
    plt.colorbar(sc, ax=axes[0], label="t")

    axes[1].hist(err.reshape(-1), bins=50)
    axes[1].set_xlabel("Absolute Error (K)"); axes[1].set_title("Error Distribution")

    axes[2].scatter(x_e.reshape(-1), T_e.reshape(-1), s=2, alpha=0.4, label="True")
    axes[2].scatter(x_e.reshape(-1), T_pred.reshape(-1), s=2, alpha=0.4, label="Pred")
    axes[2].set_xlabel("x"); axes[2].set_ylabel("T (K)")
    axes[2].set_title("T vs x"); axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.suptitle("Multiphysics PINNsformer Eval", fontsize=14)
    plt.tight_layout(); plt.show()

    mae = float(np.mean(err)); rmse = float(np.sqrt(np.mean(err**2)))
    print(f"MAE: {mae:.4f} K, RMSE: {rmse:.4f} K")
    return mae, rmse


# =========================
# 5. Eval entry
# =========================

def evaluate(model_path=MODEL_PATH, data_path=DATA_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    model, history = load_model_package(model_path, device)
    x, y, z, t, T = load_eval_data(data_path)

    print("Eval result"); print(f"model_path: {model_path}")
    print(f"data points: {len(T)}, T range: [{T.min():.1f}, {T.max():.1f}]")

    mae, rmse = show_eval_result(model, device, x, y, z, t, T, history)
    return {"mae": mae, "rmse": rmse}


eval_result = evaluate()
