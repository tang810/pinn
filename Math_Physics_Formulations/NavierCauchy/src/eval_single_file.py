"""
Notebook-friendly eval file for NavierCauchy.

Online Jupyter usage:
1. Upload the trained .pt model as a public asset.
2. Upload or keep the six .npy dataset files as a public asset.
3. Fill MODEL_HASH and DATA_HASH, then run this whole file/cell.

This file loads the model and dataset, prints final metrics, and shows figures.
It does not save images.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = "144c9ae15fcf4cf39d3fda8f5b9f6efa"
MODEL_HASH = "283352cf8c214db8bd9e68c64c8e6d06"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/trained_model_navier_cauchy.pt")

REQUIRED_FILES = ["x_pde.npy", "x_dir.npy", "u_dir.npy", "x_neu.npy", "n_neu.npy", "t_neu.npy"]


# =========================
# 2. Utilities and dataset
# =========================

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_data_dir(data_path):
    if not data_path:
        raise FileNotFoundError("DATA_PATH is empty. Fill DATA_HASH first.")
    data_path = os.path.expanduser(data_path)
    if os.path.isdir(data_path):
        missing = [name for name in REQUIRED_FILES if not os.path.isfile(os.path.join(data_path, name))]
        if missing:
            raise FileNotFoundError(f"Dataset folder is missing files: {missing}. DATA_PATH={data_path!r}")
        return data_path
    raise FileNotFoundError(
        "DATA_PATH must be a folder containing x_pde.npy, x_dir.npy, u_dir.npy, "
        f"x_neu.npy, n_neu.npy, t_neu.npy. Current DATA_PATH={data_path!r}"
    )


def load_dataset(data_path, device):
    data_dir = resolve_data_dir(data_path)

    def load(name, cols=2):
        arr = np.load(os.path.join(data_dir, name)).astype(np.float32)
        arr = arr.reshape(arr.shape[0], -1)
        if arr.shape[1] != cols:
            raise ValueError(f"{name} must have shape N x {cols}, got {arr.shape}.")
        return torch.tensor(arr, dtype=torch.float32, device=device)

    return {
        "x_pde": load("x_pde.npy"),
        "x_dir": load("x_dir.npy"),
        "u_dir": load("u_dir.npy"),
        "x_neu": load("x_neu.npy"),
        "n_neu": load("n_neu.npy"),
        "t_neu": load("t_neu.npy"),
        "data_dir": data_dir,
    }


# =========================
# 3. Model and physics
# =========================

class PINN(nn.Module):
    def __init__(self, input_dim=2, output_dim=2, hidden_dim=64, num_layers=4):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def compute_strain_stress_div_sigma(model, x, lambda_val, mu_val):
    x = x.clone().detach().requires_grad_(True)
    u = model(x)

    grads = []
    for i in range(2):
        grad_u_i = torch.autograd.grad(u[:, i].sum(), x, create_graph=True)[0]
        grads.append(grad_u_i)
    grad_u = torch.stack(grads, dim=1)

    strain = 0.5 * (grad_u + grad_u.transpose(1, 2))
    trace_strain = strain[:, 0, 0] + strain[:, 1, 1]
    identity = torch.eye(2, device=x.device).unsqueeze(0).expand(x.shape[0], -1, -1)
    stress = lambda_val * trace_strain[:, None, None] * identity + 2.0 * mu_val * strain

    div_sigma = torch.zeros(x.shape[0], 2, device=x.device)
    for j in range(2):
        grad_sigma_j = torch.autograd.grad(stress[:, j, :].sum(), x, create_graph=True)[0]
        div_sigma[:, j] = grad_sigma_j.sum(dim=1)

    return u, strain, stress, div_sigma


def body_force(x):
    b = torch.zeros_like(x)
    b[:, 1] = -0.1
    return b


def compute_losses(model, data, lambda_val, mu_val):
    _, _, _, div_sigma = compute_strain_stress_div_sigma(model, data["x_pde"], lambda_val, mu_val)
    pde_loss = torch.mean((div_sigma + body_force(data["x_pde"])) ** 2)

    u_dir_pred = model(data["x_dir"])
    dir_loss = torch.mean((u_dir_pred - data["u_dir"]) ** 2)

    _, _, sigma_neu, _ = compute_strain_stress_div_sigma(model, data["x_neu"], lambda_val, mu_val)
    traction_pred = torch.einsum("nij,nj->ni", sigma_neu, data["n_neu"])
    neu_loss = torch.mean((traction_pred - data["t_neu"]) ** 2)

    total = pde_loss + 10.0 * dir_loss + 5.0 * neu_loss
    return total, pde_loss, dir_loss, neu_loss, u_dir_pred, traction_pred


def load_checkpoint(model_path, device):
    model_path = os.path.expanduser(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"MODEL_PATH does not exist: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    model = PINN(
        hidden_dim=checkpoint.get("hidden_dim", 64),
        num_layers=checkpoint.get("num_layers", 4),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return checkpoint, model


# =========================
# 4. Eval and visualization
# =========================

def evaluate(model_path=MODEL_PATH, data_path=DATA_PATH):
    device = get_device()
    checkpoint, model = load_checkpoint(model_path, device)
    data = load_dataset(data_path, device)

    lambda_val = checkpoint.get("lambda_val", 1.0)
    mu_val = checkpoint.get("mu_val", 0.5)
    total, pde_loss, dir_loss, neu_loss, u_dir_pred, traction_pred = compute_losses(
        model, data, lambda_val, mu_val
    )

    metrics = {
        "total_loss": float(total.detach().cpu()),
        "pde_loss": float(pde_loss.detach().cpu()),
        "dirichlet_rmse": float(torch.sqrt(dir_loss).detach().cpu()),
        "neumann_rmse": float(torch.sqrt(neu_loss).detach().cpu()),
    }

    print("Eval result")
    print("device:", device)
    print("model_path:", model_path)
    print("data_path:", data["data_dir"])
    for key, value in metrics.items():
        print(f"{key}: {value:.6e}")

    show_eval_result(checkpoint, model, data, u_dir_pred, traction_pred, device)
    return {"metrics": metrics, "model": model}


def show_eval_result(checkpoint, model, data, u_dir_pred, traction_pred, device):
    history = checkpoint.get("history", [])
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    if history:
        axes[0, 0].semilogy([h["epoch"] for h in history], [h["loss"] for h in history], label="total")
        axes[0, 0].semilogy([h["epoch"] for h in history], [h["pde_loss"] for h in history], label="pde")
        axes[0, 0].semilogy([h["epoch"] for h in history], [h["dir_loss"] for h in history], label="dir")
        axes[0, 0].semilogy([h["epoch"] for h in history], [h["neu_loss"] for h in history], label="neu")
        axes[0, 0].legend()
    axes[0, 0].set_title("Training loss")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].grid(True, alpha=0.3)

    u_true = data["u_dir"].detach().cpu().numpy()
    u_pred = u_dir_pred.detach().cpu().numpy()
    axes[0, 1].scatter(u_true[:, 0], u_pred[:, 0], s=12, alpha=0.5, label="ux")
    axes[0, 1].scatter(u_true[:, 1], u_pred[:, 1], s=12, alpha=0.5, label="uy")
    lo = float(min(u_true.min(), u_pred.min()))
    hi = float(max(u_true.max(), u_pred.max()))
    axes[0, 1].plot([lo, hi], [lo, hi], "r--", linewidth=1)
    axes[0, 1].set_title("Dirichlet displacement")
    axes[0, 1].set_xlabel("true")
    axes[0, 1].set_ylabel("pred")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    t_true = data["t_neu"].detach().cpu().numpy()
    t_pred = traction_pred.detach().cpu().numpy()
    axes[1, 0].scatter(t_true[:, 0], t_pred[:, 0], s=12, alpha=0.5, label="tx")
    axes[1, 0].scatter(t_true[:, 1], t_pred[:, 1], s=12, alpha=0.5, label="ty")
    lo = float(min(t_true.min(), t_pred.min()))
    hi = float(max(t_true.max(), t_pred.max()))
    axes[1, 0].plot([lo, hi], [lo, hi], "r--", linewidth=1)
    axes[1, 0].set_title("Neumann traction")
    axes[1, 0].set_xlabel("true")
    axes[1, 0].set_ylabel("pred")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    n = 80
    xs = np.linspace(0, 1, n, dtype=np.float32)
    ys = np.linspace(0, 1, n, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    pts = torch.tensor(np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1), dtype=torch.float32, device=device)
    with torch.no_grad():
        u = model(pts).cpu().numpy().reshape(n, n, 2)
    mag = np.sqrt(u[:, :, 0] ** 2 + u[:, :, 1] ** 2)

    im = axes[1, 1].imshow(mag.T, origin="lower", extent=[0, 1, 0, 1], cmap="viridis")
    axes[1, 1].set_title("Displacement magnitude")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("y")
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

    plt.tight_layout()
    plt.show()


eval_result = evaluate()
