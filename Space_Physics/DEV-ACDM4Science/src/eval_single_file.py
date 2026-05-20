"""
Notebook-friendly eval file for DEV-ACDM4Science.

Online Jupyter usage:
1. Upload train_model_surrogate.pt as a public asset.
2. Upload simul_acdm.npz as a public asset (optional, used for comparison).
3. Fill MODEL_HASH and DATA_HASH, then run this whole file/cell.

This file loads the surrogate model, visualizes the predicted fields,
and shows figures with plt.show(). It does not save images.

Self-contained — depends on .pt model and .npz data.
"""

import os

import numpy as np
import torch
import matplotlib.pyplot as plt


# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = "46b920ed3e9f4e3ebc453a92f62932a9"
MODEL_HASH = "6b9e3fb21d964a1f89973362928890c0"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/train_model_surrogate.pt")


# =========================
# 2. Data loading
# =========================

def load_or_create_data(data_path, seed=42):
    data_path = os.path.expanduser(data_path) if data_path else ""
    if data_path and os.path.isfile(data_path):
        d = np.load(data_path)
        return d["x"], d["y"], d["u"], d["v"], d["p"]
    print("Generating simulated data...")
    rng = np.random.default_rng(seed); n = 220
    x = np.linspace(-3.0, 8.0, n); y = np.linspace(-2.0, 2.0, n // 2)
    xx, yy = np.meshgrid(x, y); phase = 0.6
    u = 1.0 - 0.55 * np.exp(-((xx - 0.8) ** 2 + (1.4 * yy) ** 2) / 2.2)
    v = 0.25 * np.exp(-((xx - 1.2) ** 2 + yy**2) / 4.0) * np.sin(yy * 3.0 + phase)
    p = 0.45 * np.exp(-((xx + 0.2) ** 2 + yy**2) / 0.7) - 0.03 * xx
    return xx, yy, u, v, p


def load_model(model_path):
    model_path = os.path.expanduser(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    d = torch.load(model_path, map_location="cpu", weights_only=False)
    return d["u"].numpy(), d["v"].numpy(), d["p"].numpy()


# =========================
# 3. Visualization
# =========================

def show_eval_result(xx, yy, u_ref, v_ref, p_ref, u_pred, v_pred, p_pred):
    speed_ref = np.sqrt(u_ref**2 + v_ref**2)
    speed_pred = np.sqrt(u_pred**2 + v_pred**2)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, val, title in [
        (axes[0, 0], speed_ref, "Speed (Reference)"),
        (axes[0, 1], p_ref, "Pressure (Reference)"),
        (axes[0, 2], np.abs(np.gradient(speed_ref, axis=1)), "Speed Grad |dx| (Ref)"),
        (axes[1, 0], speed_pred, "Speed (Model)"),
        (axes[1, 1], p_pred, "Pressure (Model)"),
        (axes[1, 2], np.abs(np.gradient(speed_pred, axis=1)), "Speed Grad |dx| (Model)"),
    ]:
        im = ax.imshow(val, cmap="jet", origin="lower",
                       extent=[xx.min(), xx.max(), yy.min(), yy.max()], aspect="auto")
        ax.set_title(title); ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle("DEV-ACDM4Science Eval — Reference vs Model", fontsize=13)
    plt.tight_layout(); plt.show()

    err_u = np.mean(np.abs(u_ref - u_pred))
    err_v = np.mean(np.abs(v_ref - v_pred))
    err_p = np.mean(np.abs(p_ref - p_pred))
    print(f"MAE: u={err_u:.4e} v={err_v:.4e} p={err_p:.4e}")
    return err_u, err_v, err_p


# =========================
# 4. Eval entry
# =========================

def evaluate(model_path=MODEL_PATH, data_path=DATA_PATH, seed=42):
    xx, yy, u_ref, v_ref, p_ref = load_or_create_data(data_path, seed=seed)
    u_pred, v_pred, p_pred = load_model(model_path)

    print("Eval result")
    print(f"model_path: {model_path}")
    print(f"grid shape: {xx.shape}")

    errs = show_eval_result(xx, yy, u_ref, v_ref, p_ref, u_pred, v_pred, p_pred)
    return {"errors": {"u": errs[0], "v": errs[1], "p": errs[2]}}


eval_result = evaluate()
