"""
Notebook-friendly eval file for InverseResistance.

Online Jupyter usage:
1. Upload trained_model_inverse_resistance.pt as a public model asset.
2. Upload or keep the same data folder / .npz dataset public asset.
3. Fill DATA_HASH and MODEL_HASH, then run this whole file/cell.
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = "72b9832f8b0f441f9c408dc5dad59ad2"
MODEL_HASH = "72697bc7a95e436c81ba5704f870142a"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/trained_model_inverse_resistance_1.pt")

SEED = 42


# =========================
# 2. Utilities
# =========================

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


def resolve_model_path(model_path):
    if not model_path:
        raise FileNotFoundError("MODEL_PATH is empty. Set it to the uploaded model path.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"MODEL_PATH does not exist: {model_path}")
    return model_path


def find_npz_file(data_path, filename):
    if not data_path:
        raise FileNotFoundError("DATA_PATH is empty. Set it to the uploaded data folder path.")
    path = os.path.join(data_path, filename) if os.path.isdir(data_path) else data_path
    if os.path.isfile(path) and os.path.basename(path) == filename:
        return path
    raise FileNotFoundError(f"Could not find {filename} at DATA_PATH: {data_path}")


def load_abmn_npz(path, dim, device):
    data = np.load(path, allow_pickle=True)
    A = torch.tensor(data["A"], dtype=torch.float32, device=device)
    B = torch.tensor(data["B"], dtype=torch.float32, device=device)
    M = torch.tensor(data["M"], dtype=torch.float32, device=device)
    N = torch.tensor(data["N"], dtype=torch.float32, device=device)
    d_obs = torch.tensor(np.asarray(data["d_obs"]).reshape(1, 1), dtype=torch.float32, device=device)
    if A.numel() != dim:
        raise ValueError(f"{path} has dimension {A.numel()}, expected {dim}")
    return {"path": path, "A": A, "B": B, "M": M, "N": N, "d_obs": d_obs}


# =========================
# 3. Model and physics
# =========================

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim=1, width=64, depth=5):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def data_misfit(net_phi, obs):
    pred = net_phi(obs["M"].unsqueeze(0)) - net_phi(obs["N"].unsqueeze(0))
    return float(torch.abs(pred - obs["d_obs"]).detach().cpu())


def predict_rho_2d(net_logrho, device):
    X, Z = np.meshgrid(np.linspace(0, 50, 120), np.linspace(0, 30, 80))
    pts = torch.tensor(np.c_[X.ravel(), Z.ravel()], dtype=torch.float32, device=device)
    with torch.no_grad():
        rho = torch.exp(net_logrho(pts)).cpu().numpy().reshape(Z.shape)
    return X, Z, rho


def predict_rho_3d_slice(net_logrho, device):
    X, Y = np.meshgrid(np.linspace(0, 20, 70), np.linspace(0, 20, 70))
    Z = np.ones_like(X) * 5.0
    pts = torch.tensor(np.c_[X.ravel(), Y.ravel(), Z.ravel()], dtype=torch.float32, device=device)
    with torch.no_grad():
        rho = torch.exp(net_logrho(pts)).cpu().numpy().reshape(X.shape)
    return X, Y, rho


# =========================
# 4. Eval and visualize
# =========================

def load_models(checkpoint, device):
    width = int(checkpoint.get("width", 64))
    depth = int(checkpoint.get("depth", 5))
    phi2d = MLP(2, 1, width, depth).to(device)
    rho2d = MLP(2, 1, width, depth).to(device)
    phi3d = MLP(3, 1, width, depth).to(device)
    rho3d = MLP(3, 1, width, depth).to(device)
    phi2d.load_state_dict(checkpoint["phi_2d_state_dict"])
    rho2d.load_state_dict(checkpoint["rho_2d_state_dict"])
    phi3d.load_state_dict(checkpoint["phi_3d_state_dict"])
    rho3d.load_state_dict(checkpoint["rho_3d_state_dict"])
    for model in (phi2d, rho2d, phi3d, rho3d):
        model.eval()
    return phi2d, rho2d, phi3d, rho3d


def show_eval_results(checkpoint, rho2d_grid, rho3d_grid):
    hist2d = checkpoint.get("history_2d", {})
    hist3d = checkpoint.get("history_3d", {})

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    if hist2d.get("loss"):
        axes[0, 0].semilogy(hist2d["loss"], label="total")
        axes[0, 0].semilogy(hist2d.get("loss_pde", []), label="pde", alpha=0.8)
        axes[0, 0].semilogy(hist2d.get("loss_data", []), label="data", alpha=0.8)
    axes[0, 0].set_title("2D training loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    im1 = axes[0, 1].imshow(rho2d_grid, extent=[0, 50, 30, 0], cmap="jet", aspect="auto")
    axes[0, 1].set_title("2D recovered resistivity")
    fig.colorbar(im1, ax=axes[0, 1], label="rho")

    if hist3d.get("loss"):
        axes[1, 0].semilogy(hist3d["loss"], label="total")
        axes[1, 0].semilogy(hist3d.get("loss_pde", []), label="pde", alpha=0.8)
        axes[1, 0].semilogy(hist3d.get("loss_data", []), label="data", alpha=0.8)
    axes[1, 0].set_title("3D training loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    im2 = axes[1, 1].imshow(rho3d_grid, extent=[0, 20, 20, 0], cmap="jet", aspect="equal")
    axes[1, 1].set_title("3D recovered resistivity, z=5m")
    fig.colorbar(im2, ax=axes[1, 1], label="rho")

    plt.tight_layout()
    plt.show()


def evaluate(model_path=MODEL_PATH, data_path=DATA_PATH):
    set_seed(SEED)
    device = get_device()
    print("Using device:", device)

    model_path = resolve_model_path(model_path)
    checkpoint = load_torch(model_path, device)
    phi2d, rho2d_net, phi3d, rho3d_net = load_models(checkpoint, device)

    obs2d = load_abmn_npz(find_npz_file(data_path, "abmn_2d_obs.npz"), 2, device)
    obs3d = load_abmn_npz(find_npz_file(data_path, "abmn_3d_obs.npz"), 3, device)

    misfit2d = data_misfit(phi2d, obs2d)
    misfit3d = data_misfit(phi3d, obs3d)
    _, _, rho2d = predict_rho_2d(rho2d_net, device)
    _, _, rho3d = predict_rho_3d_slice(rho3d_net, device)

    print("\nEval result")
    print("  model_path:", model_path)
    print("  data_2d:", obs2d["path"])
    print("  data_3d:", obs3d["path"])
    print(f"  2D ABMN data misfit: {misfit2d:.6e}")
    print(f"  3D ABMN data misfit: {misfit3d:.6e}")
    print(f"  2D rho range: [{rho2d.min():.6e}, {rho2d.max():.6e}]")
    print(f"  3D rho range: [{rho3d.min():.6e}, {rho3d.max():.6e}]")

    show_eval_results(checkpoint, rho2d, rho3d)
    return {
        "misfit_2d": misfit2d,
        "misfit_3d": misfit3d,
        "rho_2d": rho2d,
        "rho_3d_slice": rho3d,
    }


eval_result = evaluate()
