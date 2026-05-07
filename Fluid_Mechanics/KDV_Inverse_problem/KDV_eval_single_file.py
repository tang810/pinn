"""
Eval-only single file for Fluid_Mechanics/KDV_Inverse_problem.

It loads:
    - model/trained_model.pth
    - data/heat_eq_data.npz

and displays:
    - predicted u(x,t)
    - ground truth u(x,t)
    - absolute error
    - curve comparison at selected times

Notebook usage:
    import KDV_eval_single_file as ev
    result = ev.run_eval(
        model_path="/path/to/trained_model.pth",
        data_path="/path/to/heat_eq_data.npz",
    )

This file does not save images. It only calls plt.show().
"""

import os
import random
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def find_file(root="/mnt/minio/userdk8e2v7l", filename="trained_model.pth"):
    for current_root, _, files in os.walk(root):
        if filename in files:
            return os.path.join(current_root, filename)
    return None


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DeepXDEFNN(nn.Module):
    """Minimal PyTorch replica of dde.maps.FNN([2] + [20] * 3 + [1], "tanh", ...)."""

    def __init__(self, layer_sizes=(2, 20, 20, 20, 1)):
        super().__init__()
        self.linears = nn.ModuleList(
            [nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]
        )
        self.activation = torch.tanh

    def forward(self, x):
        for layer in self.linears[:-1]:
            x = self.activation(layer(x))
        return self.linears[-1](x)


def load_data(data_path):
    data = np.load(data_path)
    required = {"x", "t", "usol"}
    missing = required.difference(data.files)
    if missing:
        raise KeyError(f"Dataset missing keys: {sorted(missing)}. Found keys: {data.files}")

    x = data["x"].reshape(-1).astype(np.float32)
    t = data["t"].reshape(-1).astype(np.float32)
    u = data["usol"].astype(np.float32)
    if u.shape != (len(x), len(t)):
        raise ValueError(f"Expected usol shape {(len(x), len(t))}, got {u.shape}")
    return x, t, u


def load_model(model_path, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepXDEFNN().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device


def predict(model, device, x, t, batch_size=65536):
    X, T = np.meshgrid(x, t, indexing="ij")
    xt = np.stack([X.reshape(-1), T.reshape(-1)], axis=1).astype(np.float32)

    preds = []
    with torch.no_grad():
        for start in range(0, xt.shape[0], batch_size):
            batch = torch.tensor(xt[start : start + batch_size], dtype=torch.float32, device=device)
            pred = model(batch).detach().cpu().numpy()
            preds.append(pred)
    U_pred = np.vstack(preds).reshape(len(x), len(t))
    return X, T, U_pred


def compute_metrics(U_pred, U_true):
    err = U_pred - U_true
    rel_l2 = np.linalg.norm(err) / (np.linalg.norm(U_true) + 1e-12)
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))
    max_abs = np.max(np.abs(err))
    return {
        "relative_l2": float(rel_l2),
        "mae": float(mae),
        "rmse": float(rmse),
        "max_abs_error": float(max_abs),
    }


def show_field(T, X, U, title, cmap="viridis"):
    plt.figure(figsize=(8, 4.8))
    plt.pcolormesh(T, X, U, shading="auto", cmap=cmap)
    plt.colorbar(label="u(x,t)")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def show_curve_slices(x, t, U_true, U_pred, times=(0.5, 1.0, 2.0, 4.0)):
    plt.figure(figsize=(9, 5))
    for target_t in times:
        idx = int(np.argmin(np.abs(t - target_t)))
        plt.plot(x, U_true[:, idx], linewidth=2, label=f"true t={t[idx]:.2f}")
        plt.plot(x, U_pred[:, idx], "--", linewidth=2, label=f"pred t={t[idx]:.2f}")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title("Curve Comparisons at Selected Times")
    plt.grid(alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.show()


def run_eval(
    model_path: Optional[str] = None,
    data_path: Optional[str] = None,
    search_root="/mnt/minio/userdk8e2v7l",
    show: bool = True,
    slice_times: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
):
    set_seed(42)
    if model_path is None:
        model_path = find_file(search_root, "trained_model.pth")
    if data_path is None:
        data_path = find_file(search_root, "heat_eq_data.npz")

    if model_path is None or not os.path.exists(model_path):
        raise FileNotFoundError(f"Cannot find model file: {model_path}")
    if data_path is None or not os.path.exists(data_path):
        raise FileNotFoundError(f"Cannot find data file: {data_path}")

    print("Using model:", model_path)
    print("Using data :", data_path)

    x, t, U_true = load_data(data_path)
    model, device = load_model(model_path)
    print("Using device:", device)

    X, T, U_pred = predict(model, device, x, t)
    metrics = compute_metrics(U_pred, U_true)
    print("Metrics:", metrics)
    print(
        "Prediction summary:",
        f"min={U_pred.min():.6f}",
        f"max={U_pred.max():.6f}",
        f"mean={U_pred.mean():.6f}",
    )

    if show:
        show_field(T, X, U_pred, "KDV PINN Prediction", cmap="viridis")
        show_field(T, X, U_true, "Ground Truth", cmap="viridis")
        show_field(T, X, np.abs(U_pred - U_true), "Absolute Error", cmap="magma")
        show_curve_slices(x, t, U_true, U_pred, times=slice_times)

    return {
        "model_path": model_path,
        "data_path": data_path,
        "model": model,
        "device": device,
        "x": x,
        "t": t,
        "X": X,
        "T": T,
        "U_true": U_true,
        "U_pred": U_pred,
        "metrics": metrics,
    }


if __name__ == "__main__":
    run_eval()
