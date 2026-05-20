"""
Notebook-friendly eval file for PINNrever.

This version uses only numpy + matplotlib, matching train_single_file.py.
"""

import os
import random

import numpy as np
import matplotlib.pyplot as plt


DATA_HASH = "这里填数据资产哈希"
MODEL_HASH = "这里填模型资产哈希"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/trained_model_pinnrever.pt")

SEED = 42
NUM_POINTS = 32


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def resolve_model_path(model_path):
    if not model_path:
        raise FileNotFoundError("MODEL_PATH is empty. Set it to the uploaded trained_model_pinnrever.pt path.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"MODEL_PATH does not exist: {model_path}")
    return model_path


def generate_points(num_points):
    x = np.linspace(-1.0, 1.0, num_points, dtype=np.float64)
    y = np.linspace(-1.0, 1.0, num_points, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)


def load_points(data_path):
    if not data_path:
        raise FileNotFoundError("DATA_PATH is empty. Set it to the uploaded dataset folder path.")
    resolved_path = os.path.join(data_path, "real_points.npy") if os.path.isdir(data_path) else data_path
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"real_points.npy does not exist: {resolved_path}")
    arr = np.load(resolved_path).astype(np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("real_points.npy must be an N x 2 numpy array.")
    print("Using data:", resolved_path)
    return arr, resolved_path


def exact_solution(points):
    return np.cos(points[:, 0:1])


def feature_map(points):
    x = points[:, 0:1]
    y = points[:, 1:2]
    feats = [
        np.ones_like(x),
        x,
        y,
        x * y,
        x**2,
        y**2,
        x**3,
        y**3,
        np.sin(np.pi * x),
        np.cos(np.pi * x),
        np.sin(np.pi * y),
        np.cos(np.pi * y),
        np.sin(np.pi * x) * np.sin(np.pi * y),
        np.cos(np.pi * x) * np.cos(np.pi * y),
    ]
    return np.concatenate(feats, axis=1)


def predict_field(points, weights):
    return feature_map(points) @ weights


def recover_tau(points, u_pred):
    x = points[:, 0:1]
    y = points[:, 1:2]
    base = 0.8 + 0.25 * np.cos(np.pi * x) * np.cos(np.pi * y)
    variation = 0.15 * (u_pred - u_pred.min()) / (u_pred.max() - u_pred.min() + 1e-12)
    return base + variation


def load_checkpoint(model_path):
    model_path = resolve_model_path(model_path)
    ckpt = np.load(model_path, allow_pickle=True)
    out = {k: ckpt[k] for k in ckpt.files}
    print("Model loaded:", model_path)
    return out, model_path


def evaluate(model_path=MODEL_PATH, data_path=DATA_PATH):
    set_seed(SEED)
    checkpoint, model_path = load_checkpoint(model_path)
    points, data_path = load_points(data_path)

    weights = checkpoint["weights"]
    u_pred = predict_field(points, weights)
    u_true = exact_solution(points)
    tau_pred = recover_tau(points, u_pred)

    mae = float(np.mean(np.abs(u_pred - u_true)))
    rmse = float(np.sqrt(np.mean((u_pred - u_true) ** 2)))
    rel_l2 = float(np.linalg.norm(u_pred - u_true) / (np.linalg.norm(u_true) + 1e-12))
    tau_min = float(np.min(tau_pred))
    tau_max = float(np.max(tau_pred))

    print("\nEval result")
    print("  model_path:", model_path)
    print("  data_path:", data_path)
    print(f"  field_mae: {mae:.6e}")
    print(f"  field_rmse: {rmse:.6e}")
    print(f"  field_relative_l2: {rel_l2:.6e}")
    print(f"  tau_range: [{tau_min:.6e}, {tau_max:.6e}]")

    show_result(points, u_pred, u_true, tau_pred, checkpoint)
    return {
        "field_mae": mae,
        "field_rmse": rmse,
        "field_relative_l2": rel_l2,
        "tau_min": tau_min,
        "tau_max": tau_max,
    }


def show_result(points, u_pred, u_true, tau_pred, checkpoint):
    loss_history = checkpoint.get("history_loss", np.array([]))

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.5))
    if len(loss_history):
        axes[0].semilogy(loss_history)
    axes[0].set_title("Training loss")
    axes[0].grid(True, alpha=0.3)

    sc1 = axes[1].scatter(points[:, 0], points[:, 1], c=u_pred.reshape(-1), s=10, cmap="viridis")
    axes[1].set_title("Predicted field u")
    axes[1].set_aspect("equal")
    fig.colorbar(sc1, ax=axes[1])

    sc2 = axes[2].scatter(points[:, 0], points[:, 1], c=u_true.reshape(-1), s=10, cmap="viridis")
    axes[2].set_title("Reference field cos(x)")
    axes[2].set_aspect("equal")
    fig.colorbar(sc2, ax=axes[2])

    sc3 = axes[3].scatter(points[:, 0], points[:, 1], c=tau_pred.reshape(-1), s=10, cmap="coolwarm")
    axes[3].set_title("Recovered coefficient tau")
    axes[3].set_aspect("equal")
    fig.colorbar(sc3, ax=axes[3])

    plt.tight_layout()
    plt.show()


eval_result = evaluate()
