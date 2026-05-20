"""
Notebook-friendly eval file for cavity_flow_PIKAN.

Loads trained_model_cavity_flow_pikan.pt and cavity_Re2000_256.mat, then shows
velocity-field results.
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


DATA_HASH = "95e16e13bdc649f5b75b2759684676a6"
MODEL_HASH = "e89c68c883724a6e8300791db81d8dc3"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/trained_model_cavity_flow_pikan.pt")

SEED = 42


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_torch(path, device="cpu"):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


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


def load_model(model_path, device):
    if not model_path:
        raise FileNotFoundError("MODEL_PATH is empty. Set it to the uploaded trained_model_cavity_flow_pikan.pt path.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"MODEL_PATH does not exist: {model_path}")

    ckpt = load_torch(model_path, device)
    model = CavityChebyKAN(int(ckpt.get("width", 32)), int(ckpt.get("degree", 8))).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("Model loaded:", model_path)
    return model, ckpt, model_path


def evaluate(model_path=MODEL_PATH, data_path=DATA_PATH):
    set_seed(SEED)
    device = get_device()
    print("Using device:", device)

    model, ckpt, model_path = load_model(model_path, device)
    dataset = load_cavity_data(data_path)
    print("Using data:", dataset["path"])

    xy = torch.tensor(dataset["xy"], dtype=torch.float32, device=device)
    y_mean = ckpt["y_mean"].to(device)
    y_std = ckpt["y_std"].to(device)
    with torch.no_grad():
        pred = (model(xy) * y_std + y_mean).cpu().numpy()

    true = dataset["uvp"]
    vel_err = pred[:, :2] - true[:, :2]
    rel_l2_uv = float(np.linalg.norm(vel_err) / (np.linalg.norm(true[:, :2]) + 1e-12))
    mae_uv = float(np.mean(np.abs(vel_err)))
    rmse_uv = float(np.sqrt(np.mean(vel_err**2)))

    print("\nEval result")
    print("  model_path:", model_path)
    print("  data_path:", dataset["path"])
    print(f"  velocity_mae: {mae_uv:.6e}")
    print(f"  velocity_rmse: {rmse_uv:.6e}")
    print(f"  velocity_relative_l2: {rel_l2_uv:.6e}")

    show_result(dataset, pred, ckpt)
    return {"velocity_mae": mae_uv, "velocity_rmse": rmse_uv, "velocity_relative_l2": rel_l2_uv}


def show_result(dataset, pred, ckpt):
    shape = dataset["U"].shape
    u_pred = pred[:, 0].reshape(shape)
    v_pred = pred[:, 1].reshape(shape)
    speed_pred = np.sqrt(u_pred**2 + v_pred**2)
    speed_true = np.sqrt(dataset["U"]**2 + dataset["V"]**2)
    speed_err = np.abs(speed_pred - speed_true)

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.5))
    history = ckpt.get("history", [])
    if history:
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


eval_result = evaluate()
