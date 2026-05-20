"""
Notebook-friendly train file for HybridMagMomEstimator.

Online Jupyter usage:
1. Upload the project's .mat data file as a public dataset asset.
2. Fill DATA_HASH and MODEL_HASH.
3. Run this whole file/cell. It saves MODEL_PATH for later eval upload.
"""

import os
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = "42c5bfa667864dd6af27ee589874502d"
MODEL_HASH = "42c5bfa667864dd6af27ee589874502d"
DATA_FILE_NAME = "hybrid_magmom_small.mat"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/minio/Resource/{MODEL_HASH}/trained_model_hybrid_magmom.pt")



EPOCHS = 300
PRINT_EVERY = 50
SEED = 42

LR = 5e-3
HIDDEN_DIMS = [64, 64, 32]
MOMENT_SCALE = 1.0e14
DISPLAY_SCALE = 1.0e9


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


def resolve_data_path(data_path):
    candidates = []
    if data_path:
        data_path = os.path.expanduser(data_path)
        candidates.append(data_path)
        candidates.append(os.path.join(data_path, DATA_FILE_NAME))
    local_data = os.path.abspath(os.path.join(os.getcwd(), "Electromagnetism", "HybridMagMomEstimator", "data"))
    candidates.append(os.path.join(local_data, DATA_FILE_NAME))

    tried = []
    for path in candidates:
        if not path or path in tried:
            continue
        tried.append(path)
        if os.path.isfile(path):
            return path
        if os.path.isdir(path):
            preferred = os.path.join(path, DATA_FILE_NAME)
            if os.path.isfile(preferred):
                return preferred
            mat_files = [os.path.join(path, name) for name in os.listdir(path) if name.lower().endswith(".mat")]
            if mat_files:
                return mat_files[0]
    raise FileNotFoundError(
        f"Could not find {DATA_FILE_NAME}. Upload it as the public data asset and set DATA_HASH.\n"
        f"Tried paths: {tried}"
    )


def load_mat_dataset(mat_path, device):
    try:
        import scipy.io as sio
    except Exception as exc:
        raise ImportError("This file needs scipy to read .mat data. Please install scipy in the notebook environment.") from exc

    mat_data = sio.loadmat(mat_path)
    matrix = mat_data["Data"]
    measure = matrix[0, 0]["measurepoint"]
    data_m = matrix[0, 0]["M"]
    mag_s0 = matrix[0, 0]["MagS0"]
    theta = float(np.asarray(matrix[0, 0]["theta"]).reshape(-1)[0])

    x = torch.tensor(measure[:, 0], dtype=torch.float32, device=device).view(-1, 1)
    y = torch.tensor(measure[:, 1], dtype=torch.float32, device=device).view(-1, 1)
    z = torch.tensor(measure[:, 2], dtype=torch.float32, device=device).view(-1, 1)
    theta_values = torch.full_like(x, theta)

    bx = torch.tensor(mag_s0[0, :], dtype=torch.float32, device=device).view(-1, 1)
    by = torch.tensor(mag_s0[1, :], dtype=torch.float32, device=device).view(-1, 1)
    bz = torch.tensor(mag_s0[2, :], dtype=torch.float32, device=device).view(-1, 1)

    inputs_xyz = torch.cat([x, y, z], dim=1)
    target_b = torch.cat([bx, by, bz], dim=1)
    model_inputs = torch.cat([inputs_xyz, theta_values, target_b], dim=1)
    gt_moments = torch.tensor(data_m[:48, 0], dtype=torch.float32, device=device).view(1, -1)

    return {
        "mat_path": mat_path,
        "inputs_xyz": inputs_xyz,
        "target_b": target_b,
        "model_inputs": model_inputs,
        "gt_moments": gt_moments,
    }


# =========================
# 3. Model and physics
# =========================

class MagneticMomentNet(nn.Module):
    def __init__(self, input_size=7, output_size=48, hidden_dims=(128, 256, 128), moment_scale=1e5):
        super().__init__()
        h1, h2, h3 = hidden_dims
        self.moment_scale = float(moment_scale)
        self.net = nn.Sequential(
            nn.Linear(input_size, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Linear(h3, output_size),
        )
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        # Average over measurement points to estimate one 48-component moment vector.
        return torch.mean(torch.tanh(self.net(x)), dim=0, keepdim=True) * self.moment_scale


def forward_magnetic_field(magnetic_moments, inputs):
    m_count = 15
    length = 152.96
    width = 17.28
    mu_0 = 4.0 * np.pi * 1e-7
    eps = 1e-12

    x, y, z = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3]
    theta = inputs[:, 3:4]
    moments = magnetic_moments.view(-1, m_count + 1, 3)

    delta_l = length / (m_count - 1)
    positions = torch.tensor(
        [[(i - (m_count - 1) / 2.0) * delta_l, 0.0, 0.0] for i in range(m_count)],
        dtype=torch.float32,
        device=inputs.device,
    )

    b_x_total = torch.zeros_like(x)
    b_y_total = torch.zeros_like(y)
    b_z_total = torch.zeros_like(z)

    for i in range(m_count):
        r_d = positions[i]
        rr = torch.sqrt((x - r_d[0]) ** 2 + (y - r_d[1]) ** 2 + (z - r_d[2]) ** 2 + eps)
        a_x = (mu_0 / (4 * np.pi)) * (3 * (x - r_d[0]) ** 2 / rr**5 - 1 / rr**3)
        a_y = (3 * mu_0 / (4 * np.pi)) * ((x - r_d[0]) * (y - r_d[1]) / rr**5)
        a_z = (3 * mu_0 / (4 * np.pi)) * ((x - r_d[0]) * (z - r_d[2]) / rr**5)
        b_y = (mu_0 / (4 * np.pi)) * (3 * (y - r_d[1]) ** 2 / rr**5 - 1 / rr**3)
        b_z = (3 * mu_0 / (4 * np.pi)) * ((y - r_d[1]) * (z - r_d[2]) / rr**5)
        c_z = (mu_0 / (4 * np.pi)) * (3 * (z - r_d[2]) ** 2 / rr**5 - 1 / rr**3)

        b_x, c_x, c_y = a_y, a_z, b_z
        m_x = moments[:, i, 0].unsqueeze(1)
        m_y = moments[:, i, 1].unsqueeze(1)
        m_z = moments[:, i, 2].unsqueeze(1)
        b_x_total += a_x * m_x + a_y * m_y + a_z * m_z
        b_y_total += b_x * m_x + b_y * m_y + b_z * m_z
        b_z_total += c_x * m_x + c_y * m_y + c_z * m_z

    ell_mx = moments[:, m_count, 0].unsqueeze(1)
    ell_my = moments[:, m_count, 1].unsqueeze(1)
    ell_mz = moments[:, m_count, 2].unsqueeze(1)
    length_t = torch.tensor(length, dtype=torch.float32, device=inputs.device)
    width_t = torch.tensor(width, dtype=torch.float32, device=inputs.device)

    r_norm = torch.sqrt(x**2 + y**2 + z**2 + eps)
    k = torch.sqrt((length_t / 2.0) ** 2 - (width_t / 2.0) ** 2 + eps)
    t = torch.sqrt((r_norm**2 + k**2) ** 2 - 4 * k**2 * x**2 + eps)
    a = torch.sqrt(0.5 * (r_norm**2 + k**2 + t) + eps)
    b = torch.sqrt(0.5 * (r_norm**2 - k**2 + t) + eps)

    log_term = torch.log(torch.clamp((a - k) / (a + k + eps), min=eps))
    a_x = (3 * mu_0 / (4 * np.pi)) * (a / (k**2 * t + eps) + (1 / (2 * k**3 + eps)) * log_term)
    a_y = (3 * mu_0 / (4 * np.pi)) * (x * y / (a * b**2 * t + eps))
    a_z = (3 * mu_0 / (4 * np.pi)) * (x * z / (a * b**2 * t + eps))
    b_y = (3 * mu_0 / (8 * np.pi)) * (
        (2 * a * y**2) / (b**4 * t + eps) - a / (b**2 * k**2 + eps) - (1 / (2 * k**3 + eps)) * log_term
    )
    b_z = (3 * mu_0 / (4 * np.pi)) * (a * y * z / (b**4 * t + eps))
    c_z = (3 * mu_0 / (8 * np.pi)) * (
        (2 * a * z**2) / (b**4 * t + eps) - a / (b**2 * k**2 + eps) - (1 / (2 * k**3 + eps)) * log_term
    )
    b_x, c_x, c_y = a_y, a_z, b_z

    b_x_total += a_x * ell_mx + a_y * ell_my + a_z * ell_mz
    b_y_total += b_x * ell_mx + b_y * ell_my + b_z * ell_mz
    b_z_total += c_x * ell_mx + c_y * ell_my + c_z * ell_mz

    b_x_prime = b_x_total * torch.cos(theta) - b_y_total * torch.sin(theta)
    b_y_prime = b_x_total * torch.sin(theta) + b_y_total * torch.cos(theta)
    b_z_prime = b_z_total
    return torch.cat([b_x_prime, b_y_prime, b_z_prime], dim=1)


def normalized_field_loss(predicted_moments, model_inputs):
    target_b = model_inputs[:, 4:7]
    pred_b = forward_magnetic_field(predicted_moments, model_inputs)
    scale = torch.mean(target_b**2).detach() + 1e-12
    return torch.mean((pred_b - target_b) ** 2) / scale


# =========================
# 4. Train and visualize
# =========================

def train(data_path=DATA_PATH, model_path=MODEL_PATH, epochs=EPOCHS):
    set_seed(SEED)
    device = get_device()
    print("Using device:", device)

    mat_path = resolve_data_path(data_path)
    dataset = load_mat_dataset(mat_path, device)
    model_inputs = dataset["model_inputs"]
    input_mean = model_inputs.mean(dim=0, keepdim=True)
    input_std = model_inputs.std(dim=0, keepdim=True).clamp_min(1e-6)
    model_inputs_norm = (model_inputs - input_mean) / input_std

    model = MagneticMomentNet(hidden_dims=HIDDEN_DIMS, moment_scale=MOMENT_SCALE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    loss_history = []
    t0 = time.time()

    print("Dataset:", mat_path)
    print("model_inputs:", tuple(model_inputs.shape), "target moments:", tuple(dataset["gt_moments"].shape))
    print("\nStart training")

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        predicted_moments = model(model_inputs_norm)
        loss = normalized_field_loss(predicted_moments, model_inputs)
        loss.backward()
        optimizer.step()

        loss_history.append(float(loss.detach().cpu()))

        if epoch == 1 or epoch % PRINT_EVERY == 0 or epoch == epochs:
            print(f"Epoch {epoch:5d}/{epochs} | normalized field loss={loss.item():.6e}")

    model.eval()
    with torch.no_grad():
        pred_moments = model(model_inputs_norm)
        pred_b = forward_magnetic_field(pred_moments, model_inputs)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_mean": input_mean.detach().cpu(),
        "input_std": input_std.detach().cpu(),
        "hidden_dims": HIDDEN_DIMS,
        "moment_scale": MOMENT_SCALE,
        "display_scale": DISPLAY_SCALE,
        "epoch": epochs,
        "loss": loss_history[-1],
        "loss_history": loss_history,
        "mat_path": mat_path,
    }
    saved_model_path = save_checkpoint(checkpoint, model_path)

    print("\nTraining finished")
    print(f"elapsed: {time.time() - t0:.1f}s")
    print("final normalized field loss:", loss_history[-1])
    print("model saved to:", saved_model_path)

    show_result(loss_history, pred_moments, dataset["gt_moments"], pred_b, dataset["target_b"])
    return checkpoint, model


def save_checkpoint(checkpoint, model_path):
    candidate_paths = [model_path]
    fallback_path = os.path.join("/tmp", os.path.basename(model_path) or "trained_model_hybrid_magmom.pt")
    if fallback_path not in candidate_paths:
        candidate_paths.append(fallback_path)

    last_error = None
    for path in candidate_paths:
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            torch.save(checkpoint, path)
            return path
        except Exception as exc:
            last_error = exc
            print(f"Could not save model to {path}: {exc}")

    raise RuntimeError(f"Failed to save model. Last error: {last_error}") from last_error


def show_result(loss_history, pred_moments, true_moments, pred_b, true_b):
    pred = pred_moments.detach().cpu().numpy().reshape(-1) / DISPLAY_SCALE
    truth = true_moments.detach().cpu().numpy().reshape(-1)
    pred_field = pred_b.detach().cpu().numpy()
    true_field = true_b.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    axes[0].semilogy(loss_history)
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("epoch")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(truth, label="true", linewidth=2)
    axes[1].plot(pred, "--", label="pred", linewidth=1.5)
    axes[1].set_title("48 magnetic moments")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(true_field[:200, 0], label="Bx true")
    axes[2].plot(pred_field[:200, 0], "--", label="Bx pred")
    axes[2].plot(true_field[:200, 2], label="Bz true", alpha=0.8)
    axes[2].plot(pred_field[:200, 2], "--", label="Bz pred", alpha=0.8)
    axes[2].set_title("Magnetic field fit, first 200 points")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


checkpoint, model = train()
