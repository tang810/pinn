"""
Notebook-friendly train file for DEV-ACDM4Science.

Online Jupyter usage:
1. (Optional) Upload simul_acdm.npz as a public asset.
2. Fill DATA_HASH and MODEL_HASH.
3. Run this whole file/cell. It saves train_model_surrogate.npz to
   ~/minio/Resource/{MODEL_HASH}/ for later public-asset upload.

Method: Lightweight ACDM surrogate — applies a smoothing factor to the
        synthetic flow-field data and saves the result as the model.

Self-contained — only depends on the .npz dataset.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt



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

DATA_HASH = "46b920ed3e9f4e3ebc453a92f62932a9"
MODEL_HASH = "46b920ed3e9f4e3ebc453a92f62932a9"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/minio/Resource/{MODEL_HASH}/train_model_surrogate.pt")


# =========================
# 2. Data generation
# =========================

def make_simul_data(path=None, seed=42, n=220):
    rng = np.random.default_rng(seed)
    x = np.linspace(-3.0, 8.0, n)
    y = np.linspace(-2.0, 2.0, n // 2)
    xx, yy = np.meshgrid(x, y)
    phase = 0.6
    u = 1.0 - 0.55 * np.exp(-((xx - 0.8) ** 2 + (1.4 * yy) ** 2) / 2.2)
    v = 0.25 * np.exp(-((xx - 1.2) ** 2 + yy**2) / 4.0) * np.sin(yy * 3.0 + phase)
    p = 0.45 * np.exp(-((xx + 0.2) ** 2 + yy**2) / 0.7) - 0.03 * xx
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, x=xx, y=yy, u=u, v=v, p=p)
    return xx, yy, u, v, p


def load_or_create_data(data_path, seed=42):
    data_path = os.path.expanduser(data_path) if data_path else ""
    if data_path and os.path.isfile(data_path):
        d = np.load(data_path)
        return d["x"], d["y"], d["u"], d["v"], d["p"]
    print("Generating simulated data...")
    return make_simul_data(seed=seed)


# =========================
# 3. Surrogate model (ACDM lightweight)
# =========================

def apply_surrogate(u, v, p, k=0.85):
    """Lightweight ACDM: blend field with its spatial mean."""
    u2 = k * u + (1 - k) * np.mean(u)
    v2 = k * v + (1 - k) * np.mean(v)
    p2 = k * p + (1 - k) * np.mean(p)
    return u2, v2, p2


# =========================
# 4. Visualization
# =========================

def show_result(xx, yy, u, v, p, title="DEV-ACDM4Science Result"):
    speed = np.sqrt(u**2 + v**2)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, val, t in [
        (axes[0], speed, "Speed"),
        (axes[1], p, "Pressure"),
        (axes[2], np.abs(np.gradient(speed, axis=1)), "Speed Grad |dx|"),
    ]:
        im = ax.imshow(val, cmap="jet", origin="lower",
                       extent=[xx.min(), xx.max(), yy.min(), yy.max()], aspect="auto")
        ax.set_title(t); ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle(title, fontsize=13); plt.tight_layout(); plt.show()


# =========================
# 5. Train entry
# =========================

def train(data_path=DATA_PATH, model_path=MODEL_PATH, k=0.85, seed=42):
    xx, yy, u, v, p = load_or_create_data(data_path, seed=seed)
    print(f"Input shape: {xx.shape}")

    u2, v2, p2 = apply_surrogate(u, v, p, k=k)

    model_path = os.path.expanduser(model_path)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save({"u": torch.tensor(u2), "v": torch.tensor(v2), "p": torch.tensor(p2)}, model_path)

    print("Training finished")
    print(f"model saved to: {model_path}")
    print(f"surrogate factor k={k:.2f}")

    show_result(xx, yy, u2, v2, p2, title="DEV-ACDM4Science Trained Surrogate")
    return {"model_path": model_path, "u": u2, "v": v2, "p": p2}


train_result = train()
