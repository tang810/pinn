"""
Notebook-friendly train file for 2D wave equation.

Online Jupyter usage:
1. Upload data.csv as a public dataset asset.
2. Fill DATA_HASH and MODEL_HASH.
3. Run this whole file/cell. It saves trained_model_2d_wave_equation.pt to
   ~/minio/Resource/{MODEL_HASH}/ for later public-asset upload.

Expected dataset file:
    data.csv

The CSV must contain columns:
    x, y, t, u
"""

import csv
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

DATA_HASH = "91e36f764219474d9a89fc69eddd39ec"
MODEL_HASH = "91e36f764219474d9a89fc69eddd39ec"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/minio/Resource/{MODEL_HASH}/trained_model_2d_wave_equation.pt")

CSV_FILE_NAME = "data.csv"

TRAIN_EPOCHS = 2000
PRINT_EVERY = 100
SEED = 42

BATCH_SIZE = 8192
MAX_TRAIN_POINTS = 120000
LR = 1e-3
WIDTH = 96
DEPTH = 4


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
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"Invalid wave CSV shape: {arr.shape}")

    X = arr[:, :3]
    y = arr[:, 3:4]
    return X, y


def make_eval_grid(X, y, max_time_slices=4):
    xs = np.unique(X[:, 0])
    ys = np.unique(X[:, 1])
    ts = np.unique(X[:, 2])
    chosen_ts = np.linspace(0, len(ts) - 1, min(max_time_slices, len(ts))).astype(int)

    frames = []
    for ti in chosen_ts:
        t_val = ts[ti]
        mask = np.isclose(X[:, 2], t_val)
        frame_x = X[mask]
        frame_y = y[mask, 0]
        grid = np.full((len(xs), len(ys)), np.nan, dtype=np.float32)
        x_map = {v: i for i, v in enumerate(xs)}
        y_map = {v: i for i, v in enumerate(ys)}
        for row, val in zip(frame_x, frame_y):
            grid[x_map[row[0]], y_map[row[1]]] = val
        frames.append((float(t_val), grid))
    return xs, ys, frames


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

        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.net(x)


def wave_pde_residual(model, x_norm, x_mean, x_std, sample_count=2048):
    if x_norm.shape[0] > sample_count:
        idx = torch.randperm(x_norm.shape[0], device=x_norm.device)[:sample_count]
        pts = x_norm[idx].clone().detach().requires_grad_(True)
    else:
        pts = x_norm.clone().detach().requires_grad_(True)

    u = model(pts)
    grad = torch.autograd.grad(u, pts, torch.ones_like(u), create_graph=True)[0]
    u_x = grad[:, 0:1] / x_std[:, 0:1]
    u_y = grad[:, 1:2] / x_std[:, 1:2]
    u_t = grad[:, 2:3] / x_std[:, 2:3]

    u_xx = torch.autograd.grad(u_x, pts, torch.ones_like(u_x), create_graph=True)[0][:, 0:1] / x_std[:, 0:1]
    u_yy = torch.autograd.grad(u_y, pts, torch.ones_like(u_y), create_graph=True)[0][:, 1:2] / x_std[:, 1:2]
    u_tt = torch.autograd.grad(u_t, pts, torch.ones_like(u_t), create_graph=True)[0][:, 2:3] / x_std[:, 2:3]

    return torch.mean((u_tt - u_xx - u_yy) ** 2)


# =========================
# 4. Train and visualize
# =========================

def train(data_path=DATA_PATH, model_path=MODEL_PATH, epochs=TRAIN_EPOCHS):
    set_seed(SEED)
    device = get_device()
    t0 = time.time()

    csv_path = resolve_csv_path(data_path)
    X, y = load_wave_csv(csv_path)

    if X.shape[0] > MAX_TRAIN_POINTS:
        idx = np.random.choice(X.shape[0], MAX_TRAIN_POINTS, replace=False)
        X_train = X[idx]
        y_train = y[idx]
    else:
        X_train = X
        y_train = y

    x_mean = X_train.mean(axis=0, keepdims=True).astype(np.float32)
    x_std = (X_train.std(axis=0, keepdims=True) + 1e-8).astype(np.float32)
    y_mean = y_train.mean(axis=0, keepdims=True).astype(np.float32)
    y_std = (y_train.std(axis=0, keepdims=True) + 1e-8).astype(np.float32)

    Xn = (X_train - x_mean) / x_std
    yn = (y_train - y_mean) / y_std

    X_tensor = torch.tensor(Xn, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(yn, dtype=torch.float32, device=device)
    x_mean_t = torch.tensor(x_mean, dtype=torch.float32, device=device)
    x_std_t = torch.tensor(x_std, dtype=torch.float32, device=device)

    model = WaveMLP(width=WIDTH, depth=DEPTH).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = []
    n = X_tensor.shape[0]

    print("Using device:", device)
    print("Dataset:", csv_path)
    print("train points:", n, "full points:", X.shape[0])
    print("\nStart training")

    for epoch in range(1, epochs + 1):
        batch_idx = torch.randperm(n, device=device)[: min(BATCH_SIZE, n)]
        xb = X_tensor[batch_idx]
        yb = y_tensor[batch_idx]

        pred = model(xb)
        data_loss = torch.mean((pred - yb) ** 2)
        pde_loss = wave_pde_residual(model, xb, x_mean_t, x_std_t, sample_count=1024)
        loss = data_loss + 1e-4 * pde_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % PRINT_EVERY == 0 or epoch == epochs:
            row = {
                "epoch": epoch,
                "loss": float(loss.detach().cpu()),
                "data_loss": float(data_loss.detach().cpu()),
                "pde_loss": float(pde_loss.detach().cpu()),
            }
            history.append(row)
            print(
                f"Epoch {epoch:5d}/{epochs} | loss={row['loss']:.6e} | "
                f"data={row['data_loss']:.6e} | pde={row['pde_loss']:.6e}"
            )

    model.eval()
    with torch.no_grad():
        sample_n = min(50000, X.shape[0])
        sample_idx = np.linspace(0, X.shape[0] - 1, sample_n).astype(int)
        X_sample = X[sample_idx]
        y_sample = y[sample_idx]
        pred_n = model(torch.tensor((X_sample - x_mean) / x_std, dtype=torch.float32, device=device)).cpu().numpy()
        pred = pred_n * y_std + y_mean

    err = pred - y_sample
    metrics = {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "relative_l2": float(np.linalg.norm(err) / (np.linalg.norm(y_sample) + 1e-12)),
    }

    checkpoint = {
        "project": "2d_wave_equation",
        "state_dict": model.state_dict(),
        "width": WIDTH,
        "depth": DEPTH,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "history": history,
        "metrics": metrics,
        "csv_path": csv_path,
    }

    model_path = os.path.expanduser(model_path)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(checkpoint, model_path)

    print("\nTraining finished")
    print("elapsed:", f"{time.time() - t0:.1f}s")
    print("metrics:", metrics)
    print("model saved to:", model_path)

    show_train_result(history, X_sample, y_sample, pred)
    return checkpoint, model


def show_train_result(history, X_sample, y_true, y_pred):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].semilogy([h["epoch"] for h in history], [h["loss"] for h in history], label="total")
    axes[0].semilogy([h["epoch"] for h in history], [h["data_loss"] for h in history], label="data")
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("epoch")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].scatter(y_true[:, 0], y_pred[:, 0], s=4, alpha=0.35)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    axes[1].plot([lo, hi], [lo, hi], "r--", linewidth=1)
    axes[1].set_title("Prediction scatter")
    axes[1].set_xlabel("true u")
    axes[1].set_ylabel("pred u")
    axes[1].grid(True, alpha=0.3)

    order = np.argsort(X_sample[:, 2])[: min(1200, X_sample.shape[0])]
    axes[2].plot(y_true[order, 0], label="true", linewidth=2)
    axes[2].plot(y_pred[order, 0], "--", label="pred", linewidth=1.5)
    axes[2].set_title("Wave values sample")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.show()


checkpoint, model = train()
