"""
Notebook-friendly eval file for KDV_Inverse_problem.

Online Jupyter usage:
1. Upload trained_model_kdv.pt as a public asset.
2. Upload heat_eq_data.npz as a public asset.
3. Fill MODEL_HASH and DATA_HASH, then run this whole file/cell.

This file loads the FNN model, evaluates on the full (x,t) grid, and shows
figures with plt.show(). It does not save images.

Self-contained PyTorch-only — no DeepXDE dependency.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = "1babec8f54bb4393b8d4e89cd52fbcce"
MODEL_HASH = "da411883a45f434191fa59bc8eca5965"
DATA_FILENAME = "heat_eq_data.npz"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/trained_model_kdv.pt")


# =========================
# 2. Model
# =========================

class DeepXDEFNN(nn.Module):
    def __init__(self, layer_sizes=(2, 20, 20, 20, 1)):
        super().__init__()
        self.linears = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)
        ])
        self.activation = torch.tanh
    def forward(self, x):
        for layer in self.linears[:-1]: x = self.activation(layer(x))
        return self.linears[-1](x)


# =========================
# 3. Data loading
# =========================

def load_data(data_path):
    data = np.load(data_path)
    x = data["x"].reshape(-1).astype(np.float32)
    t = data["t"].reshape(-1).astype(np.float32)
    u = data["usol"].astype(np.float32)
    return x, t, u


def load_model(model_path, device):
    model_path = os.path.expanduser(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = DeepXDEFNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict(model, device, x, t, batch_size=65536):
    X, T = np.meshgrid(x, t, indexing="ij")
    xt = np.stack([X.reshape(-1), T.reshape(-1)], axis=1).astype(np.float32)
    preds = []
    with torch.no_grad():
        for start in range(0, xt.shape[0], batch_size):
            batch = torch.tensor(xt[start:start+batch_size], dtype=torch.float32, device=device)
            preds.append(model(batch).cpu().numpy())
    return X, T, np.vstack(preds).reshape(len(x), len(t))


# =========================
# 4. Visualization
# =========================

def compute_metrics(U_pred, U_true):
    err = U_pred - U_true
    return {
        "relative_l2": float(np.linalg.norm(err) / (np.linalg.norm(U_true) + 1e-12)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "max_abs_error": float(np.max(np.abs(err))),
    }


def show_eval_result(X, T, U_pred, U_true):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, U, title, cmap in [
        (axes[0], U_pred, "PINN Prediction", "viridis"),
        (axes[1], U_true, "Ground Truth", "viridis"),
        (axes[2], np.abs(U_pred - U_true), "Absolute Error", "magma"),
    ]:
        im = ax.pcolormesh(T, X, U, shading="auto", cmap=cmap)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xlabel("t"); ax.set_ylabel("x"); ax.set_title(title)
    plt.suptitle("KDV Inverse Problem — PINN Evaluation", fontsize=14)
    plt.tight_layout(); plt.show()

    plt.figure(figsize=(10, 5))
    for target_t in [0.5, 1.0, 2.0, 4.0]:
        idx = int(np.argmin(np.abs(T[0, :] - target_t)))
        plt.plot(X[:, 0], U_true[:, idx], lw=2, label=f"true t={T[0,idx]:.2f}")
        plt.plot(X[:, 0], U_pred[:, idx], "--", lw=2)
    plt.xlabel("x"); plt.ylabel("u"); plt.title("Curve Comparison at Selected Times")
    plt.legend(ncol=2, fontsize=8); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

    metrics = compute_metrics(U_pred, U_true)
    print("Metrics:", {k: f"{v:.4e}" for k, v in metrics.items()})
    return metrics


# =========================
# 5. Eval entry
# =========================

def evaluate(model_path=MODEL_PATH, data_path=DATA_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    data_path = os.path.expanduser(data_path) if data_path else ""
    candidates = []
    if data_path:
        candidates.append(data_path)
        if os.path.isdir(data_path) or not os.path.splitext(data_path)[1]:
            candidates.append(os.path.join(data_path, DATA_FILENAME))
    candidates.extend([
        os.path.expanduser(f"~/public/Resource/{DATA_HASH}/{DATA_FILENAME}"),
        os.path.expanduser(f"~/Resource/{DATA_HASH}/{DATA_FILENAME}"),
        f"/Resource/{DATA_HASH}/{DATA_FILENAME}",
    ])
    for candidate in candidates:
        candidate = os.path.abspath(os.path.normpath(candidate))
        if os.path.isfile(candidate):
            data_path = candidate
            break
    else:
        raise FileNotFoundError("Data file not found in public assets. Expected one of: " + ", ".join(candidates))

    x, t, U_true = load_data(data_path)
    model = load_model(model_path, device)
    X, T, U_pred = predict(model, device, x, t)

    print("Eval result"); print(f"model_path: {model_path}")
    print(f"x={len(x)}, t={len(t)}, U range=[{U_pred.min():.4f}, {U_pred.max():.4f}]")

    metrics = show_eval_result(X, T, U_pred, U_true)
    return {"U_pred": U_pred, "U_true": U_true, "metrics": metrics}


eval_result = evaluate()
