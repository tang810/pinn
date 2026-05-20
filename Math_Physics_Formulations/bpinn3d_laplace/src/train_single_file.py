"""
Notebook-friendly train file for bpinn3d_laplace.

Online Jupyter usage:
1. Upload the dataset as a public asset.
2. Fill DATA_HASH and MODEL_HASH.
3. Run this whole file/cell. It saves trained_model_bpinn3d_laplace.pt to
   ~/minio/Resource/{MODEL_HASH}/ for later public-asset upload.

Expected dataset format, either:
    train.npz with keys: x_u, y_u, x_f, y_f
or four CSV files:
    x_u.csv, y_u.csv, x_f.csv, y_f.csv

x_u and x_f must be N x 3 coordinates. y_u and y_f must be N x 1 values.
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

DATA_HASH = "025ce7295fa0459986a31f1a6f288a68"
MODEL_HASH = "025ce7295fa0459986a31f1a6f288a68"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/minio/Resource/{MODEL_HASH}/trained_model_bpinn3d_laplace.pt")

TRAIN_EPOCHS = 2000
PRINT_EVERY = 100
SEED = 42

LR = 1e-3
WIDTH = 64
DEPTH = 4
BATCH_U = 2048
BATCH_F = 2048
MAX_U_POINTS = 30000
MAX_F_POINTS = 30000

PDE_WEIGHT = 1.0
DATA_WEIGHT = 1.0
ALPHA_INIT = 1.0


# =========================
# 2. Utilities and dataset
# =========================

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
        x_u = as_2d(data["x_u"], cols=3)
        y_u = as_2d(data["y_u"], cols=1)
        x_f = as_2d(data["x_f"], cols=3)
        y_f = as_2d(data["y_f"], cols=1)
        return {"x_u": x_u, "y_u": y_u, "x_f": x_f, "y_f": y_f, "data_path": resolved}

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


def subsample(x, y, max_points):
    if x.shape[0] <= max_points:
        return x, y
    idx = np.random.choice(x.shape[0], max_points, replace=False)
    return x[idx], y[idx]


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
        self.log_alpha = nn.Parameter(torch.tensor(np.log(ALPHA_INIT), dtype=torch.float32))

        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

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


# =========================
# 4. Train and visualize
# =========================

def train(data_path=DATA_PATH, model_path=MODEL_PATH):
    set_seed(SEED)
    device = get_device()
    t0 = time.time()

    data = load_dataset(data_path)
    x_u, y_u = subsample(data["x_u"], data["y_u"], MAX_U_POINTS)
    x_f, y_f = subsample(data["x_f"], data["y_f"], MAX_F_POINTS)

    x_mean = np.vstack([x_u, x_f]).mean(axis=0, keepdims=True).astype(np.float32)
    x_std = (np.vstack([x_u, x_f]).std(axis=0, keepdims=True) + 1e-8).astype(np.float32)
    u_mean = y_u.mean(axis=0, keepdims=True).astype(np.float32)
    u_std = (y_u.std(axis=0, keepdims=True) + 1e-8).astype(np.float32)
    f_mean = y_f.mean(axis=0, keepdims=True).astype(np.float32)
    f_std = (y_f.std(axis=0, keepdims=True) + 1e-8).astype(np.float32)

    xu_t = torch.tensor((x_u - x_mean) / x_std, dtype=torch.float32, device=device)
    yu_t = torch.tensor((y_u - u_mean) / u_std, dtype=torch.float32, device=device)
    xf_t = torch.tensor((x_f - x_mean) / x_std, dtype=torch.float32, device=device)
    yf_t = torch.tensor((y_f - f_mean) / f_std, dtype=torch.float32, device=device)

    model = LaplaceNet(width=WIDTH, depth=DEPTH).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    history = []

    print("Using device:", device)
    print("Dataset:", data["data_path"])
    print("x_u:", x_u.shape, "x_f:", x_f.shape)
    print("\nStart training")

    nu = xu_t.shape[0]
    nf = xf_t.shape[0]
    for epoch in range(1, TRAIN_EPOCHS + 1):
        idx_u = torch.randperm(nu, device=device)[: min(BATCH_U, nu)]
        idx_f = torch.randperm(nf, device=device)[: min(BATCH_F, nf)]

        pred_u = model(xu_t[idx_u])
        data_loss = torch.mean((pred_u - yu_t[idx_u]) ** 2)

        pred_f = rhs_prediction(model, xf_t[idx_f])
        pred_f_norm = (pred_f - torch.tensor(f_mean, dtype=torch.float32, device=device)) / torch.tensor(
            f_std, dtype=torch.float32, device=device
        )
        pde_loss = torch.mean((pred_f_norm - yf_t[idx_f]) ** 2)

        loss = DATA_WEIGHT * data_loss + PDE_WEIGHT * pde_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % PRINT_EVERY == 0 or epoch == TRAIN_EPOCHS:
            row = {
                "epoch": epoch,
                "loss": float(loss.detach().cpu()),
                "data_loss": float(data_loss.detach().cpu()),
                "pde_loss": float(pde_loss.detach().cpu()),
                "alpha": float(model.alpha().detach().cpu()),
            }
            history.append(row)
            print(
                f"Epoch {epoch:5d}/{TRAIN_EPOCHS} | loss={row['loss']:.6e} | "
                f"data={row['data_loss']:.6e} | pde={row['pde_loss']:.6e} | alpha={row['alpha']:.6g}"
            )

    model.eval()
    with torch.no_grad():
        pred_u_all = model(xu_t).cpu().numpy() * u_std + u_mean

    rel_l2_u = float(np.linalg.norm(pred_u_all - y_u) / (np.linalg.norm(y_u) + 1e-12))
    checkpoint = {
        "project": "bpinn3d_laplace",
        "state_dict": model.state_dict(),
        "width": WIDTH,
        "depth": DEPTH,
        "x_mean": x_mean,
        "x_std": x_std,
        "u_mean": u_mean,
        "u_std": u_std,
        "f_mean": f_mean,
        "f_std": f_std,
        "history": history,
        "metrics": {"relative_l2_u": rel_l2_u, "alpha": float(model.alpha().detach().cpu())},
        "data_path": data["data_path"],
    }

    model_path = os.path.expanduser(model_path)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(checkpoint, model_path)

    print("\nTraining finished")
    print("elapsed:", f"{time.time() - t0:.1f}s")
    print("relative_l2_u:", f"{rel_l2_u:.6e}")
    print("model saved to:", model_path)

    show_result(history, y_u, pred_u_all, x_u)
    return checkpoint, model


def show_result(history, y_true, y_pred, x_u):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].semilogy([h["epoch"] for h in history], [h["loss"] for h in history], label="total")
    axes[0].semilogy([h["epoch"] for h in history], [h["data_loss"] for h in history], label="u data")
    axes[0].semilogy([h["epoch"] for h in history], [h["pde_loss"] for h in history], label="f pde")
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("epoch")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].scatter(y_true[:, 0], y_pred[:, 0], s=8, alpha=0.45)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    axes[1].plot([lo, hi], [lo, hi], "r--", linewidth=1)
    axes[1].set_title("u prediction scatter")
    axes[1].set_xlabel("true u")
    axes[1].set_ylabel("pred u")
    axes[1].grid(True, alpha=0.3)

    z_mid = np.median(x_u[:, 2])
    near = np.argsort(np.abs(x_u[:, 2] - z_mid))[: min(2000, x_u.shape[0])]
    sc = axes[2].scatter(x_u[near, 0], x_u[near, 1], c=y_pred[near, 0], s=12, cmap="viridis")
    axes[2].set_title("Predicted u near z-mid")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    plt.colorbar(sc, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    plt.show()


checkpoint, model = train()
