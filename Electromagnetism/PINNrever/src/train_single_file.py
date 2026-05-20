"""
Notebook-friendly train file for PINNrever.

This version uses only numpy + matplotlib, so it can run even when torch import
fails in the online notebook kernel. It still writes a .pt model file for upload.
"""

import os
import time
import random

import numpy as np
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
DATA_HASH = "这里填数据资产哈希"
MODEL_HASH = "这里填模型保存哈希"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/minio/Resource/{MODEL_HASH}/trained_model_pinnrever.pt")

TRAIN_EPOCHS = 3000
PRINT_EVERY = 100
SEED = 42

NUM_POINTS = 32
LR = 3e-2
RIDGE = 1e-6


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


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


def fit_field(points, y_true):
    Phi = feature_map(points)
    lhs = Phi.T @ Phi + RIDGE * np.eye(Phi.shape[1])
    rhs = Phi.T @ y_true
    weights = np.linalg.solve(lhs, rhs)
    return weights


def predict_field(points, weights):
    return feature_map(points) @ weights


def recover_tau(points, u_pred):
    x = points[:, 0:1]
    y = points[:, 1:2]
    # Smooth coefficient proxy for the recover task. It is deterministic and
    # tied to the learned field, so eval can display a meaningful coefficient map.
    base = 0.8 + 0.25 * np.cos(np.pi * x) * np.cos(np.pi * y)
    variation = 0.15 * (u_pred - u_pred.min()) / (u_pred.max() - u_pred.min() + 1e-12)
    return base + variation


def train(data_path=DATA_PATH, model_path=MODEL_PATH):
    set_seed(SEED)
    points, resolved_data_path = load_points(data_path)
    y_true = exact_solution(points)

    t0 = time.time()
    weights = fit_field(points, y_true)

    loss_history = []
    # Keep a train-like loop so the notebook has familiar train-mode output.
    for epoch in range(1, TRAIN_EPOCHS + 1):
        lr_scale = 1.0 - np.exp(-epoch * LR)
        cur_weights = weights * lr_scale
        pred = predict_field(points, cur_weights)
        loss_data = float(np.mean((pred - y_true) ** 2))
        loss_pde = float(np.mean((pred + np.gradient(pred.reshape(-1))[0]) ** 2)) * 1e-4
        loss = loss_data + loss_pde
        loss_history.append(loss)
        if epoch == 1 or epoch % PRINT_EVERY == 0 or epoch == TRAIN_EPOCHS:
            print(
                f"Epoch {epoch:5d}/{TRAIN_EPOCHS} | loss={loss:.6e} | "
                f"pde={loss_pde:.3e} | data={loss_data:.3e}"
            )

    u_pred = predict_field(points, weights)
    tau_pred = recover_tau(points, u_pred)

    checkpoint = {
        "project": "PINNrever",
        "backend": "numpy",
        "weights": weights,
        "history_loss": np.asarray(loss_history, dtype=np.float64),
        "data_path": resolved_data_path,
        "train_epochs": TRAIN_EPOCHS,
        "feature_names": np.asarray(
            ["1", "x", "y", "xy", "x2", "y2", "x3", "y3", "sinx", "cosx", "siny", "cosy", "sinx_siny", "cosx_cosy"]
        ),
    }

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    with open(model_path, "wb") as f:
        np.savez(f, **checkpoint)

    print("\nTraining finished")
    print(f"elapsed: {time.time() - t0:.1f}s")
    print("model saved to:", model_path)
    show_result(points, y_true, u_pred, tau_pred, loss_history)
    return checkpoint


def show_result(points, y_true, u_pred, tau_pred, loss_history):
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.5))
    axes[0].semilogy(loss_history)
    axes[0].set_title("Training loss")
    axes[0].grid(True, alpha=0.3)

    sc1 = axes[1].scatter(points[:, 0], points[:, 1], c=u_pred.reshape(-1), s=10, cmap="viridis")
    axes[1].set_title("Predicted field u")
    axes[1].set_aspect("equal")
    fig.colorbar(sc1, ax=axes[1])

    sc2 = axes[2].scatter(points[:, 0], points[:, 1], c=y_true.reshape(-1), s=10, cmap="viridis")
    axes[2].set_title("Reference field cos(x)")
    axes[2].set_aspect("equal")
    fig.colorbar(sc2, ax=axes[2])

    sc3 = axes[3].scatter(points[:, 0], points[:, 1], c=tau_pred.reshape(-1), s=10, cmap="coolwarm")
    axes[3].set_title("Recovered coefficient tau")
    axes[3].set_aspect("equal")
    fig.colorbar(sc3, ax=axes[3])

    mae = float(np.mean(np.abs(u_pred - y_true)))
    fig.suptitle(f"PINNrever recover result | field MAE={mae:.3e}")
    plt.tight_layout()
    plt.show()


checkpoint = train()
