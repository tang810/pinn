"""
Notebook-friendly eval file for bpinn3d_laplace.

Online Jupyter usage:
1. Upload the trained .pt model as a public asset.
2. Upload or keep the dataset as a public asset.
3. Fill MODEL_HASH and DATA_HASH, then run this whole file/cell.

This file loads the model and part of the dataset, prints final metrics,
and shows figures with plt.show(). It does not save images.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = "025ce7295fa0459986a31f1a6f288a68"
MODEL_HASH = "86a1cff74fe04313b6b1960ff99d09b5"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/trained_model_bpinn3d_laplace.pt")

MAX_EVAL_POINTS = 50000


# =========================
# 2. Utilities and dataset
# =========================

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def as_2d(arr, cols=None):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    else:
        arr = arr.reshape(arr.shape[0], -1)
    if cols is not None and arr.shape[1] != cols:
        raise ValueError(f"Expected array with {cols} columns, got shape {arr.shape}.")
    return arr


def resolve_dataset_path(data_path):
    if not data_path:
        raise FileNotFoundError("DATA_PATH is empty. Fill DATA_HASH first.")
    data_path = os.path.expanduser(data_path)
    if os.path.isfile(data_path):
        return data_path
    if os.path.isdir(data_path):
        npz_path = os.path.join(data_path, "train.npz")
        if os.path.isfile(npz_path):
            return npz_path
        needed = ["x_u.csv", "y_u.csv", "x_f.csv", "y_f.csv"]
        if all(os.path.isfile(os.path.join(data_path, name)) for name in needed):
            return data_path
    raise FileNotFoundError(
        "Could not find bpinn3d_laplace data. Upload train.npz, or upload "
        "x_u.csv/y_u.csv/x_f.csv/y_f.csv into the data asset folder. "
        f"Current DATA_PATH={data_path!r}"
    )


def load_dataset(data_path):
    resolved = resolve_dataset_path(data_path)
    if os.path.isfile(resolved) and resolved.lower().endswith(".npz"):
        data = np.load(resolved)
        required = ["x_u", "y_u", "x_f", "y_f"]
        missing = [k for k in required if k not in data.files]
        if missing:
            raise KeyError(f"train.npz missing keys: {missing}")
        return {
            "x_u": as_2d(data["x_u"], cols=3),
            "y_u": as_2d(data["y_u"], cols=1),
            "x_f": as_2d(data["x_f"], cols=3),
            "y_f": as_2d(data["y_f"], cols=1),
            "data_path": resolved,
        }

    if os.path.isdir(resolved):
        def read_csv(name, cols):
            path = os.path.join(resolved, name)
            try:
                arr = np.loadtxt(path, delimiter=",")
            except ValueError:
                arr = np.loadtxt(path, delimiter=",", skiprows=1)
            return as_2d(arr, cols=cols)

        return {
            "x_u": read_csv("x_u.csv", 3),
            "y_u": read_csv("y_u.csv", 1),
            "x_f": read_csv("x_f.csv", 3),
            "y_f": read_csv("y_f.csv", 1),
            "data_path": resolved,
        }

    raise ValueError(f"Unsupported dataset path: {resolved}")


# =========================
# 3. Model and physics
# =========================

class LaplaceNet(nn.Module):
    def __init__(self, width=64, depth=4):
        super().__init__()
        layers = [nn.Linear(3, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)
        self.log_alpha = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, x):
        return self.net(x)

    def alpha(self):
        return torch.exp(self.log_alpha)


def laplacian(model, xyz):
    xyz = xyz.clone().detach().requires_grad_(True)
    u = model(xyz)
    grad = torch.autograd.grad(u, xyz, torch.ones_like(u), create_graph=True)[0]
    uxx = torch.autograd.grad(grad[:, 0:1], xyz, torch.ones_like(grad[:, 0:1]), create_graph=True)[0][:, 0:1]
    uyy = torch.autograd.grad(grad[:, 1:2], xyz, torch.ones_like(grad[:, 1:2]), create_graph=True)[0][:, 1:2]
    uzz = torch.autograd.grad(grad[:, 2:3], xyz, torch.ones_like(grad[:, 2:3]), create_graph=True)[0][:, 2:3]
    return u, uxx + uyy + uzz


def rhs_prediction(model, xyz):
    u, lap = laplacian(model, xyz)
    return 0.01 * lap + model.alpha() * torch.tanh(u)


def load_checkpoint(model_path, device):
    model_path = os.path.expanduser(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"MODEL_PATH does not exist: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    model = LaplaceNet(width=checkpoint.get("width", 64), depth=checkpoint.get("depth", 4)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return checkpoint, model


# =========================
# 4. Eval and visualization
# =========================

def evaluate(model_path=MODEL_PATH, data_path=DATA_PATH):
    device = get_device()
    checkpoint, model = load_checkpoint(model_path, device)
    data = load_dataset(data_path)

    x_u = data["x_u"]
    y_u = data["y_u"]
    x_f = data["x_f"]
    y_f = data["y_f"]

    if x_u.shape[0] > MAX_EVAL_POINTS:
        idx = np.linspace(0, x_u.shape[0] - 1, MAX_EVAL_POINTS).astype(int)
        x_u_eval = x_u[idx]
        y_u_eval = y_u[idx]
    else:
        x_u_eval = x_u
        y_u_eval = y_u

    if x_f.shape[0] > MAX_EVAL_POINTS:
        idx = np.linspace(0, x_f.shape[0] - 1, MAX_EVAL_POINTS).astype(int)
        x_f_eval = x_f[idx]
        y_f_eval = y_f[idx]
    else:
        x_f_eval = x_f
        y_f_eval = y_f

    x_mean = checkpoint["x_mean"]
    x_std = checkpoint["x_std"]
    u_mean = checkpoint["u_mean"]
    u_std = checkpoint["u_std"]
    f_mean = checkpoint["f_mean"]
    f_std = checkpoint["f_std"]

    with torch.no_grad():
        xu_n = torch.tensor((x_u_eval - x_mean) / x_std, dtype=torch.float32, device=device)
        pred_u = model(xu_n).cpu().numpy() * u_std + u_mean

    xf_n = torch.tensor((x_f_eval - x_mean) / x_std, dtype=torch.float32, device=device)
    pred_f = rhs_prediction(model, xf_n).detach().cpu().numpy()

    u_err = pred_u - y_u_eval
    f_err = pred_f - y_f_eval
    metrics = {
        "u_rmse": float(np.sqrt(np.mean(u_err**2))),
        "u_relative_l2": float(np.linalg.norm(u_err) / (np.linalg.norm(y_u_eval) + 1e-12)),
        "f_rmse": float(np.sqrt(np.mean(f_err**2))),
        "f_relative_l2": float(np.linalg.norm(f_err) / (np.linalg.norm(y_f_eval) + 1e-12)),
        "alpha": float(model.alpha().detach().cpu()),
    }

    print("Eval result")
    print("device:", device)
    print("model_path:", model_path)
    print("data_path:", data["data_path"])
    for key, value in metrics.items():
        print(f"{key}: {value:.6e}")

    show_eval_result(checkpoint, x_u_eval, y_u_eval, pred_u, x_f_eval, y_f_eval, pred_f)
    return {"metrics": metrics, "model": model, "x_u": x_u_eval, "pred_u": pred_u}


def show_eval_result(checkpoint, x_u, y_u, pred_u, x_f, y_f, pred_f):
    history = checkpoint.get("history", [])
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    if history:
        axes[0, 0].semilogy([h["epoch"] for h in history], [h["loss"] for h in history], label="total")
        axes[0, 0].semilogy([h["epoch"] for h in history], [h["data_loss"] for h in history], label="u data")
        axes[0, 0].semilogy([h["epoch"] for h in history], [h["pde_loss"] for h in history], label="f pde")
        axes[0, 0].legend()
    axes[0, 0].set_title("Training loss")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].scatter(y_u[:, 0], pred_u[:, 0], s=8, alpha=0.4)
    lo = float(min(y_u.min(), pred_u.min()))
    hi = float(max(y_u.max(), pred_u.max()))
    axes[0, 1].plot([lo, hi], [lo, hi], "r--", linewidth=1)
    axes[0, 1].set_title("u prediction scatter")
    axes[0, 1].set_xlabel("true u")
    axes[0, 1].set_ylabel("pred u")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].scatter(y_f[:, 0], pred_f[:, 0], s=8, alpha=0.4)
    lo = float(min(y_f.min(), pred_f.min()))
    hi = float(max(y_f.max(), pred_f.max()))
    axes[1, 0].plot([lo, hi], [lo, hi], "r--", linewidth=1)
    axes[1, 0].set_title("f prediction scatter")
    axes[1, 0].set_xlabel("true f")
    axes[1, 0].set_ylabel("pred f")
    axes[1, 0].grid(True, alpha=0.3)

    z_mid = np.median(x_u[:, 2])
    near = np.argsort(np.abs(x_u[:, 2] - z_mid))[: min(2500, x_u.shape[0])]
    sc = axes[1, 1].scatter(x_u[near, 0], x_u[near, 1], c=pred_u[near, 0], s=12, cmap="viridis")
    axes[1, 1].set_title("Predicted u near z-mid")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("y")
    plt.colorbar(sc, ax=axes[1, 1], fraction=0.046)

    plt.tight_layout()
    plt.show()


eval_result = evaluate()
