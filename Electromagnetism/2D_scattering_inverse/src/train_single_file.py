"""
Notebook-friendly train file for the 2D electromagnetic inverse scattering PINN.

Usage in the online Jupyter environment:
1. Upload the dataset as a public asset.
2. Fill DATA_HASH and MODEL_HASH.
3. Run this whole file/cell. It saves MODEL_PATH, then upload that model for eval.

Expected dataset files, if a folder is provided:
    src_points/x_src.pt          dict with key "x_src"
    collocation/xy_col.pt       dict with key "xy_col"
    observations/obs_data*.pt   dict with keys xy_obs_list, Eobs_re_list, Eobs_im_list

The online asset folder can also contain the three .pt files directly:
    x_src.pt
    xy_col.pt
    obs_data*.pt
"""

import glob
import time
import random
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = "a044220c2a874a239832fb3caf7acdd0"
MODEL_HASH = "a044220c2a874a239832fb3caf7acdd0"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/minio/Resource/{MODEL_HASH}/trained_model_2d_scattering_inverse.pt")
DATA_DIR_NAME = "data_small"
OBS_FILE_NAME = "obs_data_small.pt"

EPOCHS = 200
PRINT_EVERY = 50
SEED = 42

WIDTH = 64
LR = 5e-4


# =========================
# 2. Physics configuration
# =========================

FREQUENCIES = [3.0, 4.0]
ANGLES = [0.0, np.pi / 3.0, 2.0 * np.pi / 3.0]

L = 1.0
BETA = 60.0
TRUE_EPS = 1.5
TRUE_R = 0.15


# =========================
# 3. Utility functions
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


def load_torch(path, device="cpu"):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def to_tensor_list(values, device):
    return [torch.as_tensor(v, dtype=torch.float32, device=device) for v in values]


# =========================
# 4. Physics and model
# =========================

def ez_inc(xy, theta, k):
    theta = torch.as_tensor(theta, dtype=xy.dtype, device=xy.device)
    phase = k * (xy[:, 0:1] * torch.cos(theta) + xy[:, 1:2] * torch.sin(theta))
    return torch.cos(phase), torch.sin(phase)


def contrast_chi(xy, eps, r):
    rr = torch.sqrt(xy[:, 0:1] ** 2 + xy[:, 1:2] ** 2)
    mask = torch.sigmoid(BETA * (r - rr))
    return (eps - 1.0) * mask


def green_2d(x_obs, x_src, k):
    diff = x_obs[:, None, :] - x_src[None, :, :]
    rr = torch.norm(diff, dim=2) + 1e-6
    amp = 1.0 / torch.sqrt(8.0 * np.pi * k * rr)
    phase = k * rr - np.pi / 4.0
    return amp * torch.cos(phase), amp * torch.sin(phase)


def lippmann_schwinger_sc(x_obs, x_src, eps, r, k, theta):
    Ei_re, Ei_im = ez_inc(x_src, theta, k)
    chi = contrast_chi(x_src, eps, r)
    G_re, G_im = green_2d(x_obs, x_src, k)

    Ei_re, Ei_im, chi = Ei_re.T, Ei_im.T, chi.T
    Es1_re = k**2 * torch.mean(G_re * chi * Ei_re - G_im * chi * Ei_im, dim=1, keepdim=True)
    Es1_im = k**2 * torch.mean(G_re * chi * Ei_im + G_im * chi * Ei_re, dim=1, keepdim=True)

    Et_re = Ei_re + Es1_re
    Et_im = Ei_im + Es1_im
    Es_re = k**2 * torch.mean(G_re * chi * Et_re - G_im * chi * Et_im, dim=1, keepdim=True)
    Es_im = k**2 * torch.mean(G_re * chi * Et_im + G_im * chi * Et_re, dim=1, keepdim=True)
    return Es_re, Es_im


class MultiFreqPINN(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.eps_param = nn.Parameter(torch.tensor(0.0))
        self.r_param = nn.Parameter(torch.tensor(0.0))
        self.net = nn.Sequential(
            nn.Linear(3, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 2 * len(ANGLES)),
        )

    def forward(self, x, freq_idx):
        k = FREQUENCIES[freq_idx]
        k_feat = torch.ones_like(x[:, 0:1]) * k
        return self.net(torch.cat([x, k_feat], dim=1))

    def physical_params(self):
        eps = 1.0 + 1.5 * torch.sigmoid(self.eps_param)
        r = 0.12 + 0.06 * torch.sigmoid(self.r_param)
        return eps, r


def pde_loss(model, xy, eps, r, fi, ai, theta, k):
    xy = xy.detach().clone().requires_grad_(True)
    out = model(xy, fi)
    Es_re = out[:, 2 * ai:2 * ai + 1]
    Es_im = out[:, 2 * ai + 1:2 * ai + 2]

    Ei_re, Ei_im = ez_inc(xy, theta, k)
    chi = contrast_chi(xy, eps, r)

    grad_re = torch.autograd.grad(Es_re.sum(), xy, create_graph=True)[0]
    grad_im = torch.autograd.grad(Es_im.sum(), xy, create_graph=True)[0]
    lap_re = (
        torch.autograd.grad(grad_re[:, 0].sum(), xy, create_graph=True)[0][:, 0:1]
        + torch.autograd.grad(grad_re[:, 1].sum(), xy, create_graph=True)[0][:, 1:2]
    )
    lap_im = (
        torch.autograd.grad(grad_im[:, 0].sum(), xy, create_graph=True)[0][:, 0:1]
        + torch.autograd.grad(grad_im[:, 1].sum(), xy, create_graph=True)[0][:, 1:2]
    )

    res_re = lap_re + k**2 * Es_re + k**2 * chi * (Ei_re + Es_re)
    res_im = lap_im + k**2 * Es_im + k**2 * chi * (Ei_im + Es_im)
    return torch.mean(res_re**2 + res_im**2)


def data_loss(model, xy, Eobs_re, Eobs_im, fi, ai, theta, k):
    out = model(xy, fi)
    Es_re = out[:, 2 * ai:2 * ai + 1]
    Es_im = out[:, 2 * ai + 1:2 * ai + 2]
    Ei_re, Ei_im = ez_inc(xy, theta, k)
    Et_re = Ei_re + Es_re
    Et_im = Ei_im + Es_im

    mse = torch.mean((Et_re - Eobs_re) ** 2 + (Et_im - Eobs_im) ** 2)
    amp_obs = torch.sqrt(Eobs_re**2 + Eobs_im**2)
    amp_pred = torch.sqrt(Et_re**2 + Et_im**2)
    amp = torch.mean((amp_pred - amp_obs) ** 2)
    return 0.8 * mse + 0.2 * amp


def radius_constraint_loss(r, target_r=TRUE_R):
    return 0.05 * (r - target_r) ** 2


def pde_weight(epoch):
    if epoch < 80:
        return 0.8
    if epoch < 160:
        return 1.2
    return 1.5


def radius_weight(epoch, total_epochs):
    return min(0.3, epoch / max(float(total_epochs), 1.0))


# =========================
# 5. Dataset loading
# =========================

def generate_observations(N_obs, x_src, eps, r, device):
    xy_obs_list, Eobs_re_list, Eobs_im_list = [], [], []
    for k in FREQUENCIES:
        for theta in ANGLES:
            t = torch.linspace(0, 2 * np.pi, N_obs, device=device)
            xy_obs = torch.stack([0.7 * torch.cos(t), 0.7 * torch.sin(t)], dim=1)
            E_re, E_im = lippmann_schwinger_sc(xy_obs, x_src, eps, r, k, theta)
            xy_obs_list.append(xy_obs)
            Eobs_re_list.append(E_re.detach())
            Eobs_im_list.append(E_im.detach())
    return xy_obs_list, Eobs_re_list, Eobs_im_list


def load_inverse_scattering_dataset(data_path, device):
    if not data_path:
        raise FileNotFoundError("DATA_PATH is empty. Set it to the uploaded data folder path.")
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"DATA_PATH must be the uploaded data folder: {data_path}")

    x_src_path = resolve_dataset_file(
        os.path.join(data_path, "src_points", "x_src.pt"),
        os.path.join(data_path, "x_src.pt"),
        pattern=os.path.join(data_path, "**", "x_src.pt"),
    )
    xy_col_path = resolve_dataset_file(
        os.path.join(data_path, "collocation", "xy_col.pt"),
        os.path.join(data_path, "xy_col.pt"),
        pattern=os.path.join(data_path, "**", "xy_col.pt"),
    )
    obs_path = resolve_dataset_file(
        os.path.join(data_path, "observations", OBS_FILE_NAME),
        os.path.join(data_path, OBS_FILE_NAME),
        pattern=os.path.join(data_path, "**", "obs_data*.pt"),
    )

    obj = load_torch(x_src_path, "cpu")
    x_src = obj.get("x_src", obj) if isinstance(obj, dict) else obj
    x_src = torch.as_tensor(x_src, dtype=torch.float32, device=device)

    obj = load_torch(xy_col_path, "cpu")
    xy_col = obj.get("xy_col", obj) if isinstance(obj, dict) else obj
    xy_col = torch.as_tensor(xy_col, dtype=torch.float32, device=device)

    obj = load_torch(obs_path, "cpu")
    if not isinstance(obj, dict) or not {"xy_obs_list", "Eobs_re_list", "Eobs_im_list"}.issubset(obj):
        raise ValueError(f"Observation file has unexpected format: {obs_path}")
    xy_obs_list = to_tensor_list(obj["xy_obs_list"], device)
    Eobs_re_list = to_tensor_list(obj["Eobs_re_list"], device)
    Eobs_im_list = to_tensor_list(obj["Eobs_im_list"], device)

    print("Dataset loaded:")
    print("  data_path:", data_path)
    print("  x_src_path:", x_src_path)
    print("  xy_col_path:", xy_col_path)
    print("  obs_path:", obs_path)
    print("  x_src:", tuple(x_src.shape))
    print("  xy_col:", tuple(xy_col.shape))
    print("  obs groups:", len(xy_obs_list))
    return {
        "data_path": data_path,
        "x_src": x_src,
        "xy_col": xy_col,
        "xy_obs_list": xy_obs_list,
        "Eobs_re_list": Eobs_re_list,
        "Eobs_im_list": Eobs_im_list,
    }


def resolve_dataset_file(*paths, pattern=None):
    for path in paths:
        if os.path.exists(path):
            return path
    if pattern:
        matches = sorted(glob.glob(pattern, recursive=True))
        if matches:
            return matches[-1]
    raise FileNotFoundError(
        "Required dataset file does not exist. Tried:\n  "
        + "\n  ".join(paths)
        + (f"\nPattern:\n  {pattern}" if pattern else "")
    )


def has_flat_dataset(path):
    return (
        os.path.isdir(path)
        and os.path.isfile(os.path.join(path, "x_src.pt"))
        and os.path.isfile(os.path.join(path, "xy_col.pt"))
        and os.path.isfile(os.path.join(path, OBS_FILE_NAME))
    )


def candidate_data_roots(data_path):
    roots = []
    data_path = os.path.expanduser(data_path) if data_path else ""
    if data_path:
        roots.append(data_path)
        roots.append(os.path.dirname(data_path) if os.path.isfile(data_path) else "")
        roots.append(os.path.join(data_path, DATA_DIR_NAME))

    if DATA_HASH:
        roots += [
            f"/Resource/{DATA_HASH}",
            os.path.expanduser(f"~/Resource/{DATA_HASH}"),
            os.path.expanduser(f"~/public/Resource/{DATA_HASH}"),
            f"/mnt/minio/userdk8e2v7l/Resource/{DATA_HASH}",
            f"/mnt/minio/userdk8e2v7l/public/Resource/{DATA_HASH}",
        ]
        roots += [os.path.join(root, DATA_DIR_NAME) for root in list(roots) if root]

    roots.append(os.path.abspath(os.path.join(os.getcwd(), "Electromagnetism", "2D_scattering_inverse", DATA_DIR_NAME)))

    out = []
    for root in roots:
        if root and root not in out:
            out.append(root)
    return out


# =========================
# 6. Train and show result
# =========================

def train_model(data_path=DATA_PATH, model_path=MODEL_PATH, epochs=EPOCHS, print_every=PRINT_EVERY):
    set_seed(SEED)
    device = get_device()
    print("Using device:", device)

    candidates = candidate_data_roots(data_path)
    for candidate in candidates:
        if has_flat_dataset(candidate) or (
            os.path.isdir(candidate)
            and os.path.isfile(os.path.join(candidate, "x_src.pt"))
            and os.path.isfile(os.path.join(candidate, "xy_col.pt"))
        ):
            data_path = candidate
            break
    else:
        raise FileNotFoundError(
            "DATA_PATH not found in public assets. Upload x_src.pt, xy_col.pt, and "
            f"{OBS_FILE_NAME} into the same asset folder, then set DATA_HASH.\n"
            f"Tried: {candidates}"
        )

    dataset = load_inverse_scattering_dataset(data_path, device)
    model = MultiFreqPINN(width=WIDTH).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    hist = defaultdict(list)
    t0 = time.time()
    best_loss = float("inf")

    print("\nStart training")
    for ep in range(epochs + 1):
        optimizer.zero_grad()
        eps, r = model.physical_params()

        loss_pde = torch.tensor(0.0, device=device)
        loss_data = torch.tensor(0.0, device=device)

        idx = 0
        for fi, k in enumerate(FREQUENCIES):
            for ai, theta in enumerate(ANGLES):
                loss_pde = loss_pde + pde_loss(model, dataset["xy_col"], eps, r, fi, ai, theta, k)
                loss_data = loss_data + data_loss(
                    model,
                    dataset["xy_obs_list"][idx],
                    dataset["Eobs_re_list"][idx],
                    dataset["Eobs_im_list"][idx],
                    fi,
                    ai,
                    theta,
                    k,
                )
                idx += 1

        loss_radius = radius_constraint_loss(r)
        loss = pde_weight(ep) * loss_pde + loss_data + radius_weight(ep, epochs) * loss_radius

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        hist["loss"].append(float(loss.detach().cpu()))
        hist["loss_pde"].append(float(loss_pde.detach().cpu()))
        hist["loss_data"].append(float(loss_data.detach().cpu()))
        hist["eps"].append(float(eps.detach().cpu()))
        hist["r"].append(float(r.detach().cpu()))

        if loss.item() < best_loss:
            best_loss = loss.item()

        if ep % print_every == 0 or ep == epochs:
            eps_err = abs(eps.item() - TRUE_EPS) / TRUE_EPS * 100.0
            r_err = abs(r.item() - TRUE_R) / TRUE_R * 100.0
            print(
                f"Epoch {ep:5d}/{epochs} | loss={loss.item():.3e} | "
                f"pde={loss_pde.item():.3e} | data={loss_data.item():.3e} | "
                f"eps={eps.item():.4f} ({eps_err:.2f}%) | r={r.item():.4f} ({r_err:.2f}%)"
            )

    eps_final, r_final = model.physical_params()
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "width": WIDTH,
        "frequencies": FREQUENCIES,
        "angles": ANGLES,
        "eps": float(eps_final.detach().cpu()),
        "r": float(r_final.detach().cpu()),
        "true_eps": TRUE_EPS,
        "true_r": TRUE_R,
        "history": {k: list(v) for k, v in hist.items()},
        "data_path": dataset["data_path"],
    }
    torch.save(checkpoint, model_path)

    print("\nTraining finished")
    print(f"  elapsed: {time.time() - t0:.1f}s")
    print(f"  eps={checkpoint['eps']:.6f}, r={checkpoint['r']:.6f}")
    print(f"  model saved to: {model_path}")

    show_training_result(model, hist, device)
    return checkpoint, model, hist


def show_training_result(model, hist, device):
    model.eval()
    eps, r = model.physical_params()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].semilogy(hist["loss"], label="total")
    axes[0].semilogy(hist["loss_pde"], label="pde", alpha=0.8)
    axes[0].semilogy(hist["loss_data"], label="data", alpha=0.8)
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(hist["eps"], label="pred eps")
    axes[1].axhline(TRUE_EPS, color="k", linestyle="--", label="true eps")
    axes[1].plot(hist["r"], label="pred r")
    axes[1].axhline(TRUE_R, color="gray", linestyle="--", label="true r")
    axes[1].set_title("Inverse parameters")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    grid = torch.linspace(-L, L, 80, device=device)
    X, Y = torch.meshgrid(grid, grid, indexing="ij")
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    chi = contrast_chi(xy, eps, r).detach().cpu().numpy().reshape(80, 80)
    im = axes[2].imshow(chi.T, extent=[-L, L, -L, L], origin="lower", cmap="viridis")
    axes[2].set_title(f"Recovered contrast\n eps={eps.item():.4f}, r={r.item():.4f}")
    axes[2].set_aspect("equal")
    fig.colorbar(im, ax=axes[2])

    plt.tight_layout()
    plt.show()


checkpoint, model, history = train_model()
