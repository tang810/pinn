"""
Notebook-friendly train file for InverseResistance.

Online Jupyter usage:
1. Upload the data folder or the two .npz files as public dataset assets.
2. Fill DATA_HASH and MODEL_HASH.
3. Run this whole file/cell. It saves one model file for eval upload.

Expected dataset files:
    abmn_2d_obs.npz
    abmn_3d_obs.npz
"""

import os
import time
import random

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = "72b9832f8b0f441f9c408dc5dad59ad2"
MODEL_HASH = "72b9832f8b0f441f9c408dc5dad59ad2"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/minio/Resource/{MODEL_HASH}/trained_model_inverse_resistance.pt")

EPOCHS_2D = 1500
EPOCHS_3D = 2000
PRINT_EVERY = 100
SEED = 42

LR = 1e-3
WIDTH = 64
DEPTH = 5


# =========================
# 2. Utilities
# =========================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    """Detect device in priority: GPU (cuda) → NPU (npu/ascend) → CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_npu
        if torch_npu.npu.is_available():
            return torch.device("npu")
    except (ImportError, AttributeError):
        pass
    return torch.device("cpu")


def find_npz_file(data_path, filename):
    if not data_path:
        raise FileNotFoundError("DATA_PATH is empty. Set it to the uploaded data folder path.")
    path = os.path.join(data_path, filename) if os.path.isdir(data_path) else data_path
    if os.path.isfile(path) and os.path.basename(path) == filename:
        return path
    raise FileNotFoundError(f"Could not find {filename} at DATA_PATH: {data_path}")


def load_abmn_npz(path, dim, device):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
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


def gaussian_source(x, src, eps=0.5):
    return torch.exp(-((x - src) ** 2).sum(dim=1) / eps).unsqueeze(1)


def pde_residual(net_phi, net_logrho, x, A, B):
    phi = net_phi(x)
    rho = torch.exp(net_logrho(x))
    sigma = 1.0 / rho
    grad_phi = torch.autograd.grad(phi, x, torch.ones_like(phi), create_graph=True)[0]

    div = 0.0
    for i in range(x.shape[1]):
        div += torch.autograd.grad(
            sigma * grad_phi[:, i:i + 1],
            x,
            torch.ones_like(phi),
            create_graph=True,
        )[0][:, i:i + 1]

    src = gaussian_source(x, A) - gaussian_source(x, B)
    return div + src


def pde_loss(net_phi, net_logrho, x, A, B):
    return torch.mean(pde_residual(net_phi, net_logrho, x, A, B) ** 2)


def data_loss(net_phi, M, N, d_obs):
    return torch.mean((net_phi(M.unsqueeze(0)) - net_phi(N.unsqueeze(0)) - d_obs) ** 2)


def build_collocation_points(dim, n_points, device):
    if dim == 2:
        x = torch.rand(n_points, 2, device=device)
        x[:, 0] *= 50.0
        x[:, 1] *= 30.0
    else:
        x = torch.rand(n_points, 3, device=device)
        x[:, 0] *= 20.0
        x[:, 1] *= 20.0
        x[:, 2] *= 10.0
    return x.requires_grad_(True)


def make_model_pair(dim, device):
    return MLP(dim, 1, WIDTH, DEPTH).to(device), MLP(dim, 1, WIDTH, DEPTH).to(device)


# =========================
# 4. Train and visualize
# =========================

def train_one(dim, obs, epochs, device):
    n_points = 4000 if dim == 2 else 5000
    x = build_collocation_points(dim, n_points, device)
    net_phi, net_logrho = make_model_pair(dim, device)
    optimizer = torch.optim.Adam(list(net_phi.parameters()) + list(net_logrho.parameters()), lr=LR)
    history = {"loss": [], "loss_pde": [], "loss_data": []}

    print(f"\nStart {dim}D training")
    print("dataset:", obs["path"])
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        loss_p = pde_loss(net_phi, net_logrho, x, obs["A"], obs["B"])
        loss_d = data_loss(net_phi, obs["M"], obs["N"], obs["d_obs"])
        loss = loss_p + loss_d
        loss.backward()
        optimizer.step()

        history["loss"].append(float(loss.detach().cpu()))
        history["loss_pde"].append(float(loss_p.detach().cpu()))
        history["loss_data"].append(float(loss_d.detach().cpu()))

        if epoch == 1 or epoch % PRINT_EVERY == 0 or epoch == epochs:
            print(
                f"[{dim}D] Epoch {epoch:5d}/{epochs} | "
                f"loss={loss.item():.3e} | pde={loss_p.item():.3e} | data={loss_d.item():.3e}"
            )

    return net_phi, net_logrho, history


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


def show_training_results(hist2d, rho2d, hist3d, rho3d):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].semilogy(hist2d["loss"], label="total")
    axes[0, 0].semilogy(hist2d["loss_pde"], label="pde", alpha=0.8)
    axes[0, 0].semilogy(hist2d["loss_data"], label="data", alpha=0.8)
    axes[0, 0].set_title("2D training loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    im1 = axes[0, 1].imshow(rho2d, extent=[0, 50, 30, 0], cmap="jet", aspect="auto")
    axes[0, 1].set_title("2D recovered resistivity")
    fig.colorbar(im1, ax=axes[0, 1], label="rho")

    axes[1, 0].semilogy(hist3d["loss"], label="total")
    axes[1, 0].semilogy(hist3d["loss_pde"], label="pde", alpha=0.8)
    axes[1, 0].semilogy(hist3d["loss_data"], label="data", alpha=0.8)
    axes[1, 0].set_title("3D training loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    im2 = axes[1, 1].imshow(rho3d, extent=[0, 20, 20, 0], cmap="jet", aspect="equal")
    axes[1, 1].set_title("3D recovered resistivity, z=5m")
    fig.colorbar(im2, ax=axes[1, 1], label="rho")

    plt.tight_layout()
    plt.show()


def train(data_path=DATA_PATH, model_path=MODEL_PATH):
    data_path = os.path.expanduser(data_path) if data_path else ""
    candidates = [
        data_path,
        os.path.expanduser(f"~/public/Resource/{DATA_HASH}"),
        os.path.expanduser(f"~/Resource/{DATA_HASH}"),
        f"/Resource/{DATA_HASH}",
    ]
    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            data_path = candidate
            break
    else:
        raise FileNotFoundError(f"DATA_PATH not found in public assets: {data_path}. Set DATA_HASH to an uploaded asset folder.")

    set_seed(SEED)
    device = get_device()
    print("Using device:", device)

    obs2d_path = find_npz_file(data_path, "abmn_2d_obs.npz")
    obs3d_path = find_npz_file(data_path, "abmn_3d_obs.npz")
    obs2d = load_abmn_npz(obs2d_path, 2, device)
    obs3d = load_abmn_npz(obs3d_path, 3, device)

    t0 = time.time()
    phi2d, rho2d_net, hist2d = train_one(2, obs2d, EPOCHS_2D, device)
    phi3d, rho3d_net, hist3d = train_one(3, obs3d, EPOCHS_3D, device)

    _, _, rho2d = predict_rho_2d(rho2d_net, device)
    _, _, rho3d = predict_rho_3d_slice(rho3d_net, device)

    checkpoint = {
        "project": "InverseResistance",
        "width": WIDTH,
        "depth": DEPTH,
        "phi_2d_state_dict": phi2d.state_dict(),
        "rho_2d_state_dict": rho2d_net.state_dict(),
        "phi_3d_state_dict": phi3d.state_dict(),
        "rho_3d_state_dict": rho3d_net.state_dict(),
        "history_2d": hist2d,
        "history_3d": hist3d,
        "data_2d_path": obs2d_path,
        "data_3d_path": obs3d_path,
        "epochs_2d": EPOCHS_2D,
        "epochs_3d": EPOCHS_3D,
    }
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    torch.save(checkpoint, model_path)

    print("\nTraining finished")
    print(f"elapsed: {time.time() - t0:.1f}s")
    print("model saved to:", model_path)
    show_training_results(hist2d, rho2d, hist3d, rho3d)
    return checkpoint


checkpoint = train()
