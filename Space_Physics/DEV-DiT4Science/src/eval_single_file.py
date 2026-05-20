"""
Notebook-friendly eval file for DEV-DiT4Science.

Online Jupyter usage:
1. Upload dev_dit4science_surrogate.pt as a public asset.
2. Upload simul_airfoil_flow.npz as a public asset (optional).
3. Fill MODEL_HASH and DATA_HASH, then run this whole file/cell.

This file loads the linear surrogate model, evaluates on test data (p,u,v),
and shows figures with plt.show(). It does not save images.

Self-contained — only depends on the .npz dataset and .pt model.
"""

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt


# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = "811530c549444a5f9c77de677bfe085c"
MODEL_HASH = "29f591b520504ab798fc8703207f55bd"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/dev_dit4science_surrogate.pt")


# =========================
# 2. Data generation / loading
# =========================

def make_synthetic_data(n=2048, seed=42):
    rng = np.random.default_rng(seed)
    alpha = rng.uniform(-5.0, 15.0, size=(n, 1))
    reynolds = rng.uniform(1e5, 5e6, size=(n, 1))
    x = rng.uniform(0.0, 1.0, size=(n, 1))
    y = rng.uniform(-0.3, 0.3, size=(n, 1))
    features = np.hstack([alpha / 15.0, np.log10(reynolds) / 7.0, x, y]).astype(np.float32)
    p = (1.2 - x) * (1.0 + 0.02 * alpha) - 0.8 * y**2
    u = (1.0 + 0.05 * alpha) * (1.0 - 0.15 * y**2)
    v = 0.35 * y * (x - 0.5) + 0.01 * alpha
    targets = np.hstack([p, u, v]).astype(np.float32)
    return features, targets


def load_data(data_path, seed=42):
    data_path = os.path.expanduser(data_path) if data_path else ""
    if data_path and os.path.isfile(data_path):
        d = np.load(data_path)
        return d["x"].astype(np.float32), d["y"].astype(np.float32)
    return make_synthetic_data(seed=seed)


def load_model(model_path):
    model_path = os.path.expanduser(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict(x, model):
    return x @ model["w"] + model["b"]


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0, keepdims=True))**2, axis=0) + 1e-12
    return 1.0 - ss_res / ss_tot


# =========================
# 3. Visualization
# =========================

def show_eval_result(y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    mse = float(np.mean((y_pred - y_test)**2))

    labels = ["p", "u", "v"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for i, (ax, name) in enumerate(zip(axes, labels)):
        ax.scatter(y_test[:, i], y_pred[:, i], s=6, alpha=0.5)
        lo = min(y_test[:, i].min(), y_pred[:, i].min())
        hi = max(y_test[:, i].max(), y_pred[:, i].max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_xlabel(f"True {name}"); ax.set_ylabel(f"Pred {name}")
        ax.set_title(f"{name} (R2={r2[i]:.4f})"); ax.grid(alpha=0.3)
    plt.suptitle(f"DEV-DiT4Science Eval — MSE={mse:.4e}, R2_macro={float(np.mean(r2)):.4f}", fontsize=13)
    plt.tight_layout(); plt.show()

    fig2, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, r2); ax.axhline(0, color="gray", ls="--")
    ax.set_title("R2 per Output"); ax.set_ylabel("R2"); ax.set_ylim(-1, 1)
    for i, v in enumerate(r2): ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout(); plt.show()
    return mse, r2


# =========================
# 4. Eval entry
# =========================

def evaluate(model_path=MODEL_PATH, data_path=DATA_PATH, seed=42):
    x, y = load_data(data_path, seed=seed)
    n_train = int(x.shape[0] * 0.8)
    x_test, y_test = x[n_train:], y[n_train:]

    model = load_model(model_path)
    y_pred = predict(x_test, model)

    print("Eval result"); print(f"model_path: {model_path}")
    mse, r2 = show_eval_result(y_test, y_pred)
    print(f"MSE: {mse:.6e}")
    print(f"R2 per output: p={r2[0]:.4f} u={r2[1]:.4f} v={r2[2]:.4f}")
    return {"mse": mse, "r2": r2}


eval_result = evaluate()
