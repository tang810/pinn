"""
Notebook-friendly train file for NavierCauchy.

Online Jupyter usage:
1. Upload these six .npy files as one public dataset asset:
   x_pde.npy, x_dir.npy, u_dir.npy, x_neu.npy, n_neu.npy, t_neu.npy
2. Fill DATA_HASH and MODEL_HASH.
3. Run this whole file/cell. It saves trained_model_navier_cauchy.pt to
   ~/minio/Resource/{MODEL_HASH}/ for later public-asset upload.
"""

import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = "144c9ae15fcf4cf39d3fda8f5b9f6efa"
MODEL_HASH = "144c9ae15fcf4cf39d3fda8f5b9f6efa"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/minio/Resource/{MODEL_HASH}/trained_model_navier_cauchy.pt")

TRAIN_EPOCHS = 2000
PRINT_EVERY = 100
SEED = 42

LR = 1e-3
HIDDEN_DIM = 64
NUM_LAYERS = 4
LAMBDA_VAL = 1.0
MU_VAL = 0.5

PDE_WEIGHT = 1.0
DIR_WEIGHT = 10.0
NEU_WEIGHT = 5.0


# =========================
# 2. Utilities and dataset
# =========================

REQUIRED_FILES = ["x_pde.npy", "x_dir.npy", "u_dir.npy", "x_neu.npy", "n_neu.npy", "t_neu.npy"]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

    data = {
        "x_pde": load("x_pde.npy"),
        "x_dir": load("x_dir.npy"),
        "u_dir": load("u_dir.npy"),
        "x_neu": load("x_neu.npy"),
        "n_neu": load("n_neu.npy"),
        "t_neu": load("t_neu.npy"),
        "data_dir": data_dir,
    }
    return data


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


def compute_losses(model, data):
    _, _, _, div_sigma = compute_strain_stress_div_sigma(model, data["x_pde"], LAMBDA_VAL, MU_VAL)
    pde_loss = torch.mean((div_sigma + body_force(data["x_pde"])) ** 2)

    u_dir_pred = model(data["x_dir"])
    dir_loss = torch.mean((u_dir_pred - data["u_dir"]) ** 2)

    _, _, sigma_neu, _ = compute_strain_stress_div_sigma(model, data["x_neu"], LAMBDA_VAL, MU_VAL)
    traction_pred = torch.einsum("nij,nj->ni", sigma_neu, data["n_neu"])
    neu_loss = torch.mean((traction_pred - data["t_neu"]) ** 2)

    total = PDE_WEIGHT * pde_loss + DIR_WEIGHT * dir_loss + NEU_WEIGHT * neu_loss
    return total, pde_loss, dir_loss, neu_loss


# =========================
# 4. Train and visualize
# =========================

def train(data_path=DATA_PATH, model_path=MODEL_PATH):
    set_seed(SEED)
    device = get_device()
    t0 = time.time()

    data = load_dataset(data_path, device)
    model = PINN(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    history = []

    print("Using device:", device)
    print("Dataset:", data["data_dir"])
    print("x_pde:", tuple(data["x_pde"].shape), "x_dir:", tuple(data["x_dir"].shape), "x_neu:", tuple(data["x_neu"].shape))
    print("\nStart training")

    for epoch in range(1, TRAIN_EPOCHS + 1):
        model.train()
        total, pde_loss, dir_loss, neu_loss = compute_losses(model, data)

        optimizer.zero_grad()
        total.backward()
        optimizer.step()

        if epoch == 1 or epoch % PRINT_EVERY == 0 or epoch == TRAIN_EPOCHS:
            row = {
                "epoch": epoch,
                "loss": float(total.detach().cpu()),
                "pde_loss": float(pde_loss.detach().cpu()),
                "dir_loss": float(dir_loss.detach().cpu()),
                "neu_loss": float(neu_loss.detach().cpu()),
            }
            history.append(row)
            print(
                f"Epoch {epoch:5d}/{TRAIN_EPOCHS} | loss={row['loss']:.6e} | "
                f"pde={row['pde_loss']:.6e} | dir={row['dir_loss']:.6e} | neu={row['neu_loss']:.6e}"
            )

    model.eval()
    with torch.no_grad():
        u_dir_pred = model(data["x_dir"])
        dir_rmse = torch.sqrt(torch.mean((u_dir_pred - data["u_dir"]) ** 2)).item()

    checkpoint = {
        "project": "NavierCauchy",
        "state_dict": model.state_dict(),
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "lambda_val": LAMBDA_VAL,
        "mu_val": MU_VAL,
        "history": history,
        "metrics": {"dirichlet_rmse": float(dir_rmse)},
        "data_dir": data["data_dir"],
    }

    model_path = os.path.expanduser(model_path)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(checkpoint, model_path)

    print("\nTraining finished")
    print("elapsed:", f"{time.time() - t0:.1f}s")
    print("dirichlet_rmse:", f"{dir_rmse:.6e}")
    print("model saved to:", model_path)

    show_train_result(history, model, device)
    return checkpoint, model


def show_train_result(history, model, device):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].semilogy([h["epoch"] for h in history], [h["loss"] for h in history], label="total")
    axes[0].semilogy([h["epoch"] for h in history], [h["pde_loss"] for h in history], label="pde")
    axes[0].semilogy([h["epoch"] for h in history], [h["dir_loss"] for h in history], label="dir")
    axes[0].semilogy([h["epoch"] for h in history], [h["neu_loss"] for h in history], label="neu")
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("epoch")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    n = 80
    xs = np.linspace(0, 1, n, dtype=np.float32)
    ys = np.linspace(0, 1, n, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    pts = torch.tensor(np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1), dtype=torch.float32, device=device)
    with torch.no_grad():
        u = model(pts).cpu().numpy().reshape(n, n, 2)
    mag = np.sqrt(u[:, :, 0] ** 2 + u[:, :, 1] ** 2)

    im1 = axes[1].imshow(u[:, :, 0].T, origin="lower", extent=[0, 1, 0, 1], cmap="coolwarm")
    axes[1].set_title("Predicted ux")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(mag.T, origin="lower", extent=[0, 1, 0, 1], cmap="viridis")
    axes[2].set_title("Displacement magnitude")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    plt.show()


checkpoint, model = train()
