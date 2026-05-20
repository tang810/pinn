"""
Notebook-friendly eval file for 2D wave equation.

Online Jupyter usage:
1. Upload the trained .pt model as a public asset.
2. Upload or keep data.csv as a public dataset asset.
3. Fill MODEL_HASH and DATA_HASH, then run this whole file/cell.

This file loads the model and data, prints metrics, and shows result figures.
It does not save images.
"""

import csv
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = "91e36f764219474d9a89fc69eddd39ec"
MODEL_HASH = "939f314c1b61492ab28c1b3295939d03"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/trained_model_2d_wave_equation.pt")

CSV_FILE_NAME = "data.csv"
MAX_EVAL_POINTS = 80000


# =========================
# 2. Utilities and dataset
# =========================

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_csv_path(data_path):
    if not data_path:
        raise FileNotFoundError("DATA_PATH is empty. Fill DATA_HASH first.")
    data_path = os.path.expanduser(data_path)

    if os.path.isfile(data_path):
        return data_path

    candidate = os.path.join(data_path, CSV_FILE_NAME)
    if os.path.isfile(candidate):
        return candidate

    raise FileNotFoundError(
        f"Could not find {CSV_FILE_NAME}. Upload it to the data asset folder. "
        f"Current DATA_PATH={data_path!r}"
    )


def load_wave_csv(csv_path):
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"x", "y", "t", "u"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{csv_path} must contain columns: x, y, t, u")
        for row in reader:
            rows.append([float(row["x"]), float(row["y"]), float(row["t"]), float(row["u"])])

    arr = np.asarray(rows, dtype=np.float32)
    X = arr[:, :3]
    y = arr[:, 3:4]
    return X, y


# =========================
# 3. Model
# =========================

class WaveMLP(nn.Module):
    def __init__(self, width=96, depth=4):
        super().__init__()
        layers = [nn.Linear(3, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_checkpoint(model_path, device):
    model_path = os.path.expanduser(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"MODEL_PATH does not exist: {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    model = WaveMLP(width=checkpoint.get("width", 96), depth=checkpoint.get("depth", 4)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return checkpoint, model


def predict(model, checkpoint, X, device):
    x_mean = checkpoint["x_mean"]
    x_std = checkpoint["x_std"]
    y_mean = checkpoint["y_mean"]
    y_std = checkpoint["y_std"]

    preds = []
    batch = 65536
    with torch.no_grad():
        for start in range(0, X.shape[0], batch):
            xb = (X[start : start + batch] - x_mean) / x_std
            xb = torch.tensor(xb, dtype=torch.float32, device=device)
            pred_n = model(xb).cpu().numpy()
            preds.append(pred_n * y_std + y_mean)
    return np.vstack(preds)


def make_frame(X, values, t_value):
    xs = np.unique(X[:, 0])
    ys = np.unique(X[:, 1])
    mask = np.isclose(X[:, 2], t_value)
    frame_x = X[mask]
    frame_v = values[mask, 0]

    grid = np.full((len(xs), len(ys)), np.nan, dtype=np.float32)
    x_map = {v: i for i, v in enumerate(xs)}
    y_map = {v: i for i, v in enumerate(ys)}
    for row, val in zip(frame_x, frame_v):
        grid[x_map[row[0]], y_map[row[1]]] = val
    return xs, ys, grid


# =========================
# 4. Eval and visualization
# =========================

def evaluate(model_path=MODEL_PATH, data_path=DATA_PATH):
    device = get_device()
    csv_path = resolve_csv_path(data_path)
    checkpoint, model = load_checkpoint(model_path, device)
    X, y_true = load_wave_csv(csv_path)

    if X.shape[0] > MAX_EVAL_POINTS:
        idx = np.linspace(0, X.shape[0] - 1, MAX_EVAL_POINTS).astype(int)
        X_eval = X[idx]
        y_eval = y_true[idx]
    else:
        X_eval = X
        y_eval = y_true

    y_pred = predict(model, checkpoint, X_eval, device)
    err = y_pred - y_eval
    metrics = {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "relative_l2": float(np.linalg.norm(err) / (np.linalg.norm(y_eval) + 1e-12)),
        "max_abs_error": float(np.max(np.abs(err))),
    }

    print("Eval result")
    print("device:", device)
    print("model_path:", model_path)
    print("data_path:", csv_path)
    print("eval points:", X_eval.shape[0])
    for key, value in metrics.items():
        print(f"{key}: {value:.6e}")

    show_eval_result(checkpoint, X_eval, y_eval, y_pred, err)
    return {"metrics": metrics, "X": X_eval, "y_true": y_eval, "y_pred": y_pred, "model": model}


def show_eval_result(checkpoint, X, y_true, y_pred, err):
    history = checkpoint.get("history", [])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    if history:
        axes[0].semilogy([h["epoch"] for h in history], [h["loss"] for h in history], label="total")
        axes[0].semilogy([h["epoch"] for h in history], [h["data_loss"] for h in history], label="data")
        axes[0].legend()
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("epoch")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(y_true[:, 0], y_pred[:, 0], s=4, alpha=0.35)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    axes[1].plot([lo, hi], [lo, hi], "r--", linewidth=1)
    axes[1].set_title("Prediction scatter")
    axes[1].set_xlabel("true u")
    axes[1].set_ylabel("pred u")
    axes[1].grid(True, alpha=0.3)

    axes[2].hist(err[:, 0], bins=80)
    axes[2].set_title("Error histogram")
    axes[2].set_xlabel("prediction error")
    plt.tight_layout()
    plt.show()

    ts = np.unique(X[:, 2])
    chosen_ts = np.linspace(0, len(ts) - 1, min(3, len(ts))).astype(int)

    for ti in chosen_ts:
        t_value = float(ts[ti])
        _, _, true_grid = make_frame(X, y_true, t_value)
        _, _, pred_grid = make_frame(X, y_pred, t_value)
        diff_grid = pred_grid - true_grid

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
        im0 = axes[0].imshow(true_grid.T, origin="lower", cmap="viridis")
        axes[0].set_title(f"True u, t={t_value:.4g}")
        plt.colorbar(im0, ax=axes[0], fraction=0.046)

        im1 = axes[1].imshow(pred_grid.T, origin="lower", cmap="viridis")
        axes[1].set_title(f"Pred u, t={t_value:.4g}")
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        im2 = axes[2].imshow(diff_grid.T, origin="lower", cmap="coolwarm")
        axes[2].set_title("Pred - true")
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        plt.show()


eval_result = evaluate()
