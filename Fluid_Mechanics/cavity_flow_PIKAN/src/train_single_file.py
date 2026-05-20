"""
Notebook-friendly train file for cavity_flow_PIKAN.

Upload cavity_Re2000_256.mat first. This file trains a compact ChebyKAN
surrogate for (x, y) -> (u, v, p) and saves trained_model_cavity_flow_pikan.pt.
"""

import os
import time
import random

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


DATA_HASH = "95e16e13bdc649f5b75b2759684676a6"
MODEL_HASH = "95e16e13bdc649f5b75b2759684676a6"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/minio/Resource/{MODEL_HASH}/trained_model_cavity_flow_pikan.pt")

TRAIN_EPOCHS = 2000
PRINT_EVERY = 100
SEED = 42

BATCH_SIZE = 8192
LR = 1e-3
WIDTH = 32
DEGREE = 8


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


def load_cavity_data(data_path):
    try:
        import scipy.io as sio
    except Exception as exc:
        raise ImportError("scipy is required to read .mat data.") from exc

    if not data_path:
        raise FileNotFoundError("DATA_PATH is empty. Set it to the uploaded dataset folder path.")
    resolved_path = os.path.join(data_path, "cavity_Re2000_256.mat") if os.path.isdir(data_path) else data_path
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"cavity_Re2000_256.mat does not exist: {resolved_path}")

    data = sio.loadmat(resolved_path)
    X = data["X_ref"].astype(np.float32)
    Y = data["Y_ref"].astype(np.float32)
    U = data["U_ref"].astype(np.float32)
    V = data["V_ref"].astype(np.float32)
    P = data["P_ref"].astype(np.float32) if "P_ref" in data else np.zeros_like(U, dtype=np.float32)

    xy = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1).astype(np.float32)
    uvp = np.stack([U.reshape(-1), V.reshape(-1), P.reshape(-1)], axis=1).astype(np.float32)
    return {"path": resolved_path, "X": X, "Y": Y, "U": U, "V": V, "P": P, "xy": xy, "uvp": uvp}


class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.eps = 1e-6
        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1.0 / (input_dim * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1).float())

    def forward(self, x):
        x = torch.tanh(x)
        x = x.view(-1, self.input_dim, 1).expand(-1, -1, self.degree + 1)
        x = torch.clamp(x, -1.0 + self.eps, 1.0 - self.eps)
        x = torch.acos(x)
        x = torch.cos(x * self.arange)
        y = torch.einsum("bid,iod->bo", x, self.cheby_coeffs)
        return y.view(-1, self.output_dim)


class CavityChebyKAN(nn.Module):
    def __init__(self, width=32, degree=8):
        super().__init__()
        self.net = nn.Sequential(
            ChebyKANLayer(2, width, degree),
            nn.Tanh(),
            ChebyKANLayer(width, width, degree),
            nn.Tanh(),
            ChebyKANLayer(width, width, degree),
            nn.Tanh(),
            ChebyKANLayer(width, 3, degree),
        )

    def forward(self, x):
        return self.net(x)


def train(data_path=DATA_PATH, model_path=MODEL_PATH):
    set_seed(SEED)
    device = get_device()
    print("Using device:", device)
    dataset = load_cavity_data(data_path)
    print("Using data:", dataset["path"])

    xy = torch.tensor(dataset["xy"], dtype=torch.float32, device=device)
    y = torch.tensor(dataset["uvp"], dtype=torch.float32, device=device)
    y_mean = y.mean(dim=0, keepdim=True)
    y_std = y.std(dim=0, keepdim=True).clamp_min(1e-6)
    y_norm = (y - y_mean) / y_std

    model = CavityChebyKAN(WIDTH, DEGREE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    history = []
    n = xy.shape[0]
    t0 = time.time()

    print("points:", n, "batch_size:", BATCH_SIZE)
    print("\nStart training")
    for epoch in range(1, TRAIN_EPOCHS + 1):
        model.train()
        idx = torch.randint(0, n, (min(BATCH_SIZE, n),), device=device)
        pred = model(xy[idx])
        loss = torch.mean((pred - y_norm[idx]) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))

        if epoch == 1 or epoch % PRINT_EVERY == 0 or epoch == TRAIN_EPOCHS:
            print(f"Epoch {epoch:5d}/{TRAIN_EPOCHS} | normalized mse={loss.item():.6e}")

    model.eval()
    with torch.no_grad():
        pred_all = model(xy).cpu() * y_std.cpu() + y_mean.cpu()
    pred_np = pred_all.numpy()
    true_np = dataset["uvp"]
    rel_l2_uv = float(
        np.linalg.norm(pred_np[:, :2] - true_np[:, :2]) / (np.linalg.norm(true_np[:, :2]) + 1e-12)
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "width": WIDTH,
        "degree": DEGREE,
        "y_mean": y_mean.detach().cpu(),
        "y_std": y_std.detach().cpu(),
        "history": history,
        "data_path": dataset["path"],
        "rel_l2_uv": rel_l2_uv,
    }
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    torch.save(checkpoint, model_path)

    print("\nTraining finished")
    print(f"elapsed: {time.time() - t0:.1f}s")
    print(f"relative L2 velocity: {rel_l2_uv:.6e}")
    print("model saved to:", model_path)
    show_result(dataset, pred_np, history)
    return checkpoint, model


def show_result(dataset, pred, history):
    shape = dataset["U"].shape
    u_pred = pred[:, 0].reshape(shape)
    v_pred = pred[:, 1].reshape(shape)
    speed_pred = np.sqrt(u_pred**2 + v_pred**2)
    speed_true = np.sqrt(dataset["U"]**2 + dataset["V"]**2)
    speed_err = np.abs(speed_pred - speed_true)

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.5))
    axes[0].semilogy(history)
    axes[0].set_title("Training loss")
    axes[0].grid(True, alpha=0.3)

    im1 = axes[1].imshow(speed_pred.T, origin="lower", extent=[0, 1, 0, 1], cmap="viridis")
    axes[1].set_title("Predicted speed")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(speed_true.T, origin="lower", extent=[0, 1, 0, 1], cmap="viridis")
    axes[2].set_title("Reference speed")
    fig.colorbar(im2, ax=axes[2])

    im3 = axes[3].imshow(speed_err.T, origin="lower", extent=[0, 1, 0, 1], cmap="magma")
    axes[3].set_title(f"Speed abs error, max={speed_err.max():.2e}")
    fig.colorbar(im3, ax=axes[3])

    plt.tight_layout()
    plt.show()


checkpoint, model = train()
