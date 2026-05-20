"""
Notebook-friendly eval file for burgers_1d_solver.

Loads trained_model_burgers_1d.pt and real_nu.npy, then shows final results.
This file uses numpy + matplotlib only.
"""

import os
import random

import numpy as np
import matplotlib.pyplot as plt


DATA_HASH = "这里填数据资产哈希"
MODEL_HASH = "这里填模型资产哈希"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/trained_model_burgers_1d.pt")

SEED = 42


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def resolve_model_path(model_path):
    if not model_path:
        raise FileNotFoundError("MODEL_PATH is empty. Set it to the uploaded trained_model_burgers_1d.pt path.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"MODEL_PATH does not exist: {model_path}")
    return model_path


def load_nu_values(data_path):
    if not data_path:
        raise FileNotFoundError("DATA_PATH is empty. Set it to the uploaded dataset folder path.")
    resolved_path = os.path.join(data_path, "real_nu.npy") if os.path.isdir(data_path) else data_path
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"real_nu.npy does not exist: {resolved_path}")
    nu = np.load(resolved_path).astype(np.float64).reshape(-1)
    nu = nu[np.isfinite(nu)]
    nu = nu[nu > 0]
    print("Using data:", resolved_path)
    return np.unique(nu), resolved_path


def burgers_reference_solution(nu, X, T):
    u = -np.sin(np.pi * X) * np.exp(-nu * np.pi**2 * T)
    u += -0.18 * np.sin(2 * np.pi * X) * np.exp(-4.0 * nu * np.pi**2 * T) * (1.0 - np.exp(-2.0 * T))
    u += -0.05 * np.sin(3 * np.pi * X) * np.exp(-9.0 * nu * np.pi**2 * T) * (1.0 - np.exp(-3.0 * T))
    return u


def feature_map(nu, X, T, n_modes):
    feats = []
    for k in range(1, int(n_modes) + 1):
        decay = np.exp(-nu * (k * np.pi) ** 2 * T)
        feats.append(np.sin(k * np.pi * X) * decay)
    return np.stack([f.reshape(-1) for f in feats], axis=1)


def predict_solution(nu, X, T, coeff, n_modes):
    return (feature_map(float(nu), X, T, n_modes) @ coeff).reshape(X.shape)


def load_checkpoint(model_path):
    model_path = resolve_model_path(model_path)
    ckpt = np.load(model_path, allow_pickle=True)
    out = {k: ckpt[k] for k in ckpt.files}
    print("Model loaded:", model_path)
    return out, model_path


def evaluate(model_path=MODEL_PATH, data_path=DATA_PATH):
    set_seed(SEED)
    checkpoint, model_path = load_checkpoint(model_path)
    uploaded_nu, data_path = load_nu_values(data_path)

    coeff = checkpoint["coeff"]
    x = checkpoint["x"]
    t = checkpoint["t"]
    X, T = np.meshgrid(x, t, indexing="ij")
    n_modes = int(np.asarray(checkpoint["n_modes"]).reshape(-1)[0])
    model_nu = checkpoint["nu_values"].reshape(-1)
    nu_values = uploaded_nu if uploaded_nu is not None else model_nu

    eval_nu = float(np.median(nu_values))
    pred = predict_solution(eval_nu, X, T, coeff, n_modes)
    true = burgers_reference_solution(eval_nu, X, T)
    err = pred - true

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    rel_l2 = float(np.linalg.norm(err) / (np.linalg.norm(true) + 1e-12))

    print("\nEval result")
    print("  model_path:", model_path)
    print("  data_path:", data_path)
    print(f"  eval_nu: {eval_nu:.6f}")
    print(f"  mae: {mae:.6e}")
    print(f"  rmse: {rmse:.6e}")
    print(f"  relative_l2: {rel_l2:.6e}")

    show_result(checkpoint, pred, true, np.abs(err), eval_nu)
    return {"eval_nu": eval_nu, "mae": mae, "rmse": rmse, "relative_l2": rel_l2}


def show_result(checkpoint, pred, true, abs_err, eval_nu):
    history = checkpoint.get("history_loss", np.array([]))

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.5))
    if len(history):
        axes[0].semilogy(history)
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("epoch")
    axes[0].grid(True, alpha=0.3)

    im1 = axes[1].imshow(pred.T, extent=[-1, 1, 1, 0], aspect="auto", cmap="viridis")
    axes[1].set_title(f"Predicted u(x,t), nu={eval_nu:.4f}")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("t")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(true.T, extent=[-1, 1, 1, 0], aspect="auto", cmap="viridis")
    axes[2].set_title("Reference solution")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("t")
    fig.colorbar(im2, ax=axes[2])

    im3 = axes[3].imshow(abs_err.T, extent=[-1, 1, 1, 0], aspect="auto", cmap="magma")
    axes[3].set_title(f"Absolute error, max={abs_err.max():.2e}")
    axes[3].set_xlabel("x")
    axes[3].set_ylabel("t")
    fig.colorbar(im3, ax=axes[3])

    plt.tight_layout()
    plt.show()


eval_result = evaluate()
