"""
Notebook-friendly eval file for the 2D electromagnetic inverse scattering PINN.

Usage in the online Jupyter environment:
1. Upload the trained .pt/.pth model as a public asset.
2. Upload or keep the dataset public asset.
3. Fill MODEL_HASH and DATA_HASH below.
4. Run this whole file/cell. It prints metrics and shows figures only.
"""

import random
import os
import glob

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = "a044220c2a874a239832fb3caf7acdd0"
MODEL_HASH = "ea1a71a05505414a9c0c00f37d910a01"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/trained_model_2d_scattering_inverse.pt")
DATA_DIR_NAME = "data_small"
OBS_FILE_NAME = "obs_data_small.pt"

SEED = 42
WIDTH = 64


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


# =========================
# 5. Dataset and checkpoint loading
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

    x_src_path = first_existing_path(
        os.path.join(data_path, "src_points", "x_src.pt"),
        os.path.join(data_path, "x_src.pt"),
    )
    obs_path = resolve_dataset_file(
        os.path.join(data_path, "observations", OBS_FILE_NAME),
        os.path.join(data_path, OBS_FILE_NAME),
        pattern=os.path.join(data_path, "**", "obs_data*.pt"),
    )
    for required_path in [x_src_path, obs_path]:
        if not os.path.exists(required_path):
            raise FileNotFoundError(f"Required dataset file does not exist: {required_path}")

    obj = load_torch(x_src_path, "cpu")
    x_src = obj.get("x_src", obj) if isinstance(obj, dict) else obj
    x_src = torch.as_tensor(x_src, dtype=torch.float32, device=device)

    obj = load_torch(obs_path, "cpu")
    if not isinstance(obj, dict) or not {"xy_obs_list", "Eobs_re_list", "Eobs_im_list"}.issubset(obj):
        raise ValueError(f"Observation file has unexpected format: {obs_path}")
    xy_obs_list = to_tensor_list(obj["xy_obs_list"], device)
    Eobs_re_list = to_tensor_list(obj["Eobs_re_list"], device)
    Eobs_im_list = to_tensor_list(obj["Eobs_im_list"], device)

    print("Dataset loaded:")
    print("  data_path:", data_path)
    print("  x_src:", tuple(x_src.shape))
    print("  obs groups:", len(xy_obs_list))
    return {
        "data_path": data_path,
        "x_src": x_src,
        "xy_obs_list": xy_obs_list,
        "Eobs_re_list": Eobs_re_list,
        "Eobs_im_list": Eobs_im_list,
    }


def first_existing_path(*paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return paths[0]


def resolve_dataset_file(*paths, pattern=None):
    for path in paths:
        if os.path.exists(path):
            return path
    if pattern:
        matches = sorted(glob.glob(pattern, recursive=True))
        if matches:
            return matches[-1]
    return paths[0]


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


def load_model(model_path, device):
    if not model_path:
        raise FileNotFoundError("MODEL_PATH is empty. Set it to the uploaded model path.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"MODEL_PATH does not exist: {model_path}")

    checkpoint = load_torch(model_path, device)
    width = checkpoint.get("width", WIDTH) if isinstance(checkpoint, dict) else WIDTH
    model = MultiFreqPINN(width=width).to(device)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("Model loaded:", model_path)
    return model, checkpoint, model_path


# =========================
# 6. Eval and show result
# =========================

def evaluate(model_path=MODEL_PATH, data_path=DATA_PATH):
    set_seed(SEED)
    device = get_device()
    print("Using device:", device)

    data_candidates = candidate_data_roots(data_path)
    for candidate in data_candidates:
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
            f"Tried: {data_candidates}"
        )
    model_path = os.path.expanduser(model_path) if model_path else ""
    local_model = os.path.abspath(
        os.path.join(os.getcwd(), "Electromagnetism", "2D_scattering_inverse", "trained_model_2d_scattering_inverse.pt")
    )
    model_candidates = [
        model_path,
        os.path.join(model_path, "trained_model_2d_scattering_inverse.pt") if model_path else "",
        local_model,
        os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/trained_model_2d_scattering_inverse.pt"),
        os.path.expanduser(f"~/Resource/{MODEL_HASH}/trained_model_2d_scattering_inverse.pt"),
        f"/Resource/{MODEL_HASH}/trained_model_2d_scattering_inverse.pt",
    ]
    for candidate in model_candidates:
        if candidate and os.path.isfile(candidate):
            model_path = candidate
            break
    else:
        raise FileNotFoundError("Model file not found in public assets. Expected one of: " + ", ".join(model_candidates))

    model, checkpoint, model_path = load_model(model_path, device)
    dataset = load_inverse_scattering_dataset(data_path, device)

    eps, r = model.physical_params()
    eps_val = float(eps.detach().cpu())
    r_val = float(r.detach().cpu())
    eps_err = abs(eps_val - TRUE_EPS) / TRUE_EPS * 100.0
    r_err = abs(r_val - TRUE_R) / TRUE_R * 100.0

    total_loss = 0.0
    idx = 0
    with torch.no_grad():
        for fi, k in enumerate(FREQUENCIES):
            for ai, theta in enumerate(ANGLES):
                loss = data_loss(
                    model,
                    dataset["xy_obs_list"][idx],
                    dataset["Eobs_re_list"][idx],
                    dataset["Eobs_im_list"][idx],
                    fi,
                    ai,
                    theta,
                    k,
                )
                total_loss += float(loss.detach().cpu())
                idx += 1
    avg_loss = total_loss / max(idx, 1)

    print("\nEval result")
    print("  model_path:", model_path)
    print("  data_path:", dataset["data_path"])
    print(f"  avg data loss: {avg_loss:.6e}")
    print(f"  recovered eps: {eps_val:.6f} | true eps: {TRUE_EPS:.6f} | error: {eps_err:.2f}%")
    print(f"  recovered r  : {r_val:.6f} | true r  : {TRUE_R:.6f} | error: {r_err:.2f}%")

    show_eval_result(model, dataset, avg_loss, device)
    return {
        "avg_data_loss": avg_loss,
        "eps": eps_val,
        "r": r_val,
        "eps_error_percent": eps_err,
        "r_error_percent": r_err,
        "model": model,
    }


def show_eval_result(model, dataset, avg_loss, device):
    eps, r = model.physical_params()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    grid = torch.linspace(-L, L, 80, device=device)
    X, Y = torch.meshgrid(grid, grid, indexing="ij")
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    chi_pred = contrast_chi(xy, eps, r).detach().cpu().numpy().reshape(80, 80)
    chi_true = contrast_chi(
        xy,
        torch.tensor(TRUE_EPS, dtype=torch.float32, device=device),
        torch.tensor(TRUE_R, dtype=torch.float32, device=device),
    ).detach().cpu().numpy().reshape(80, 80)

    im0 = axes[0].imshow(chi_pred.T, extent=[-L, L, -L, L], origin="lower", cmap="viridis")
    axes[0].set_title(f"Recovered contrast\n eps={eps.item():.4f}, r={r.item():.4f}")
    axes[0].set_aspect("equal")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(chi_true.T, extent=[-L, L, -L, L], origin="lower", cmap="viridis")
    axes[1].set_title(f"True contrast\n eps={TRUE_EPS:.4f}, r={TRUE_R:.4f}")
    axes[1].set_aspect("equal")
    fig.colorbar(im1, ax=axes[1])

    fi, ai, theta, k = 0, 0, ANGLES[0], FREQUENCIES[0]
    xy_obs = dataset["xy_obs_list"][0]
    Eobs_re = dataset["Eobs_re_list"][0]
    Eobs_im = dataset["Eobs_im_list"][0]
    with torch.no_grad():
        out = model(xy_obs, fi)
        Es_re = out[:, 2 * ai:2 * ai + 1]
        Es_im = out[:, 2 * ai + 1:2 * ai + 2]
        Ei_re, Ei_im = ez_inc(xy_obs, theta, k)
        pred_amp = torch.sqrt((Ei_re + Es_re) ** 2 + (Ei_im + Es_im) ** 2).cpu().numpy().reshape(-1)
        true_amp = torch.sqrt(Eobs_re**2 + Eobs_im**2).cpu().numpy().reshape(-1)

    axes[2].plot(true_amp, label="observed |E|", linewidth=2)
    axes[2].plot(pred_amp, "--", label="predicted |E|", linewidth=2)
    axes[2].set_title(f"Observation fit, avg loss={avg_loss:.2e}")
    axes[2].set_xlabel("observation point index")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


eval_result = evaluate()
