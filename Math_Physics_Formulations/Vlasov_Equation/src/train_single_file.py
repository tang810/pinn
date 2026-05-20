"""
Notebook-friendly train file for Vlasov_Equation.

Upload E.npy and f.npy as public assets, fill the hash fields below, then run
this whole file/cell. The model is saved to:
~/minio/Resource/{MODEL_HASH}/trained_model_vlasov_equation.pt
"""

import os
import time
import random
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# =========================
# 1. Parameters to edit
# =========================

# Case A: E.npy and f.npy are in the same uploaded folder asset.
DATA_HASH = "851695ee0e7c4288a189c37d45703ce7"

# Case B: E.npy and f.npy were uploaded as two separate file assets.
# Leave these empty when using DATA_HASH.
E_HASH = "851695ee0e7c4288a189c37d45703ce7"
F_HASH = "851695ee0e7c4288a189c37d45703ce7"

MODEL_HASH = "851695ee0e7c4288a189c37d45703ce7"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/minio/Resource/{MODEL_HASH}/trained_model_vlasov_equation.pt")

TRAIN_EPOCHS = 500
PRINT_EVERY = 50
SEED = 42

N_TRAIN_POINTS = 5000
BATCH_SIZE = 1024
LR = 1e-3
LAYERS = [3, 50, 50, 50, 2]

T_MAX = 62.5
X_MAX = 10.0
V_MIN = -5.0
V_MAX = 5.0
LAMBDA_PDE = 0.1

REQUIRED_FILES = ["E.npy", "f.npy"]


# =========================
# 2. Paths and data
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


def resource_roots():
    return [
        os.path.expanduser("~/public/Resource"),
        "/mnt/minio/public/Resource",
        os.path.expanduser("~/public/Resource/userdk8e2v7l"),
        "/mnt/minio/public/Resource/userdk8e2v7l",
        os.path.expanduser("~/minio/Resource"),
        "/mnt/minio/Resource",
        os.path.expanduser("~/minio/Resource/userdk8e2v7l"),
        "/mnt/minio/Resource/userdk8e2v7l",
    ]


def uniq(items):
    out = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


def is_filled(value):
    return bool(value) and not value.startswith("FILL_")


def candidate_folder_paths(data_path):
    candidates = [os.path.expanduser(data_path)] if data_path else []
    if is_filled(DATA_HASH):
        for root in resource_roots():
            candidates.append(os.path.join(root, DATA_HASH))
    return uniq(candidates)


def candidate_file_paths(file_hash, filename):
    if not is_filled(file_hash):
        return []
    candidates = [os.path.expanduser(file_hash)]
    for root in resource_roots():
        candidates.append(os.path.join(root, file_hash))
        candidates.append(os.path.join(root, file_hash, filename))
    return uniq(candidates)


def resolve_data_files(data_path):
    tried_folders = []
    missing_by_folder = {}
    for folder in candidate_folder_paths(data_path):
        tried_folders.append(folder)
        if not os.path.isdir(folder):
            continue
        missing = [name for name in REQUIRED_FILES if not os.path.isfile(os.path.join(folder, name))]
        if missing:
            missing_by_folder[folder] = missing
            continue
        return os.path.join(folder, "E.npy"), os.path.join(folder, "f.npy"), folder

    e_candidates = candidate_file_paths(E_HASH, "E.npy")
    f_candidates = candidate_file_paths(F_HASH, "f.npy")
    e_file = next((p for p in e_candidates if os.path.isfile(p)), None)
    f_file = next((p for p in f_candidates if os.path.isfile(p)), None)
    if e_file and f_file:
        return e_file, f_file, f"E={e_file}; f={f_file}"

    print("No data files found, generating simulated Vlasov data as fallback.")
    return None, None, "simul"


def load_vlasov_data(data_path):
    e_path, f_path, source = resolve_data_files(data_path)
    if e_path is None:
        nt, nx, nv = 10, 32, 32
        t = np.linspace(0.0, T_MAX, nt, dtype=np.float32)
        x = np.linspace(0.0, X_MAX, nx, dtype=np.float32)
        v = np.linspace(V_MIN, V_MAX, nv, dtype=np.float32)
        T, X, V = np.meshgrid(t, x, v, indexing="ij")
        E = (np.sin(2 * np.pi * X / X_MAX) * np.cos(T)).astype(np.float32)
        f = (np.exp(-V**2 / 2.0) * (1.0 + 0.1 * np.sin(2 * np.pi * X / X_MAX) * np.cos(T))).astype(np.float32)
        return {"source": source, "E": E, "f": f, "t": t, "x": x, "v": v}
    E = np.load(e_path).astype(np.float32)
    f = np.load(f_path).astype(np.float32)
    if E.shape != f.shape or E.ndim != 3:
        raise ValueError(f"E.npy and f.npy must have the same 3D shape. Got E={E.shape}, f={f.shape}")

    nt, nx, nv = f.shape
    t = np.linspace(0.0, T_MAX, nt, dtype=np.float32)
    x = np.linspace(0.0, X_MAX, nx, dtype=np.float32)
    v = np.linspace(V_MIN, V_MAX, nv, dtype=np.float32)
    return {"source": source, "E": E, "f": f, "t": t, "x": x, "v": v}


def sample_training_points(data, n_points, seed):
    nt, nx, nv = data["f"].shape
    total = nt * nx * nv
    n_points = min(int(n_points), total)
    rng = np.random.default_rng(seed)
    flat_idx = rng.choice(total, size=n_points, replace=False)
    it, ix, iv = np.unravel_index(flat_idx, (nt, nx, nv))
    coords = np.stack([data["t"][it], data["x"][ix], data["v"][iv]], axis=1).astype(np.float32)
    targets = np.stack([data["f"][it, ix, iv], data["E"][it, ix, iv]], axis=1).astype(np.float32)
    return coords, targets


def coord_scale_tensors(device):
    offset = torch.tensor([[T_MAX / 2.0, X_MAX / 2.0, 0.0]], dtype=torch.float32, device=device)
    scale = torch.tensor([[T_MAX / 2.0, X_MAX / 2.0, (V_MAX - V_MIN) / 2.0]], dtype=torch.float32, device=device)
    deriv_scale = torch.tensor([[2.0 / T_MAX, 2.0 / X_MAX, 2.0 / (V_MAX - V_MIN)]], dtype=torch.float32, device=device)
    return offset, scale, deriv_scale


# =========================
# 3. Model and loss
# =========================

class DNN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        blocks = []
        for i in range(len(layers) - 2):
            blocks.append((f"layer_{i}", nn.Linear(layers[i], layers[i + 1])))
            blocks.append((f"activation_{i}", nn.Tanh()))
        blocks.append((f"layer_{len(layers) - 2}", nn.Linear(layers[-2], layers[-1])))
        self.layers = nn.Sequential(OrderedDict(blocks))
        self.lambda_1 = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.lambda_2 = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        for module in self.layers:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x_norm):
        return self.layers(x_norm)


def model_outputs(model, coords, offset, scale):
    out = model((coords - offset) / scale)
    return out[:, 0:1], out[:, 1:2]


def vlasov_residual(model, coords, offset, scale, deriv_scale):
    coords = coords.clone().detach().requires_grad_(True)
    coords_norm = (coords - offset) / scale
    out = model(coords_norm)
    f_pred = out[:, 0:1]
    E_pred = out[:, 1:2]
    grad_norm = torch.autograd.grad(
        f_pred,
        coords_norm,
        grad_outputs=torch.ones_like(f_pred),
        retain_graph=True,
        create_graph=True,
    )[0]
    f_t = grad_norm[:, 0:1] * deriv_scale[:, 0:1]
    f_x = grad_norm[:, 1:2] * deriv_scale[:, 1:2]
    f_v = grad_norm[:, 2:3] * deriv_scale[:, 2:3]
    v = coords[:, 2:3]
    return f_t + model.lambda_1 * v * f_x + model.lambda_2 * E_pred * f_v


def loss_terms(model, coords, targets, offset, scale, deriv_scale):
    f_pred, E_pred = model_outputs(model, coords, offset, scale)
    g = vlasov_residual(model, coords, offset, scale, deriv_scale)
    loss_f = torch.mean((f_pred - targets[:, 0:1]) ** 2)
    loss_E = torch.mean((E_pred - targets[:, 1:2]) ** 2)
    loss_pde = torch.mean(g ** 2)
    total = loss_f + loss_E + LAMBDA_PDE * loss_pde
    return total, loss_f, loss_E, loss_pde


# =========================
# 4. Train and show result
# =========================

def train(data_path=DATA_PATH, model_path=MODEL_PATH):
    set_seed(SEED)
    device = get_device()
    print("Using device:", device)

    data = load_vlasov_data(data_path)
    coords_np, targets_np = sample_training_points(data, N_TRAIN_POINTS, SEED)
    coords_all = torch.tensor(coords_np, dtype=torch.float32, device=device)
    targets_all = torch.tensor(targets_np, dtype=torch.float32, device=device)

    offset, scale, deriv_scale = coord_scale_tensors(device)
    model = DNN(LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    history = {"total": [], "f": [], "E": [], "pde": [], "lambda_1": [], "lambda_2": []}

    print("Dataset:", data["source"])
    print("E/f shape:", data["E"].shape)
    print("training points:", coords_all.shape[0])
    t0 = time.time()

    for epoch in range(1, TRAIN_EPOCHS + 1):
        idx = torch.randint(0, coords_all.shape[0], (min(BATCH_SIZE, coords_all.shape[0]),), device=device)
        optimizer.zero_grad()
        total, loss_f, loss_E, loss_pde = loss_terms(model, coords_all[idx], targets_all[idx], offset, scale, deriv_scale)
        total.backward()
        optimizer.step()

        history["total"].append(float(total.detach().cpu()))
        history["f"].append(float(loss_f.detach().cpu()))
        history["E"].append(float(loss_E.detach().cpu()))
        history["pde"].append(float(loss_pde.detach().cpu()))
        history["lambda_1"].append(float(model.lambda_1.detach().cpu()))
        history["lambda_2"].append(float(model.lambda_2.detach().cpu()))

        if epoch == 1 or epoch % PRINT_EVERY == 0 or epoch == TRAIN_EPOCHS:
            print(
                f"Epoch {epoch:5d}/{TRAIN_EPOCHS} | loss={history['total'][-1]:.6e} | "
                f"f={history['f'][-1]:.3e} | E={history['E'][-1]:.3e} | pde={history['pde'][-1]:.3e}"
            )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "layers": LAYERS,
        "history": history,
        "epoch": TRAIN_EPOCHS,
        "data_shape": data["f"].shape,
        "ranges": {"t_max": T_MAX, "x_max": X_MAX, "v_min": V_MIN, "v_max": V_MAX},
    }
    model_path = os.path.abspath(os.path.expanduser(model_path))
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(checkpoint, model_path)

    print("\nTraining finished")
    print(f"elapsed: {time.time() - t0:.1f}s")
    print("final loss:", history["total"][-1])
    print("model saved to:", model_path)
    show_result(model, data, history, device)
    return checkpoint, model


def predict_chunked(model, coords_np, device, chunk=50000):
    offset, scale, _ = coord_scale_tensors(device)
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, coords_np.shape[0], chunk):
            coords = torch.tensor(coords_np[start:start + chunk], dtype=torch.float32, device=device)
            f_pred, E_pred = model_outputs(model, coords, offset, scale)
            preds.append(torch.cat([f_pred, E_pred], dim=1).cpu().numpy())
    return np.vstack(preds)


def show_result(model, data, history, device):
    t_idx = data["f"].shape[0] // 2
    x_idx = data["f"].shape[1] // 2
    v_idx = data["f"].shape[2] // 2
    v_line = data["v"]
    x_line = data["x"]

    coords_v = np.stack([np.full_like(v_line, data["t"][t_idx]), np.full_like(v_line, data["x"][x_idx]), v_line], axis=1)
    coords_x = np.stack([np.full_like(x_line, data["t"][t_idx]), x_line, np.full_like(x_line, data["v"][v_idx])], axis=1)
    pred_v = predict_chunked(model, coords_v.astype(np.float32), device)
    pred_x = predict_chunked(model, coords_x.astype(np.float32), device)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].semilogy(history["total"], label="total")
    axes[0].semilogy(history["f"], label="f")
    axes[0].semilogy(history["E"], label="E")
    axes[0].semilogy(history["pde"], label="pde")
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(v_line, data["f"][t_idx, x_idx, :], label="f true")
    axes[1].plot(v_line, pred_v[:, 0], "--", label="f pred")
    axes[1].set_title("Velocity distribution")
    axes[1].set_xlabel("v")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(x_line, data["E"][t_idx, :, v_idx], label="E true")
    axes[2].plot(x_line, pred_x[:, 1], "--", label="E pred")
    axes[2].set_title("Electric field slice")
    axes[2].set_xlabel("x")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


checkpoint, model = train()
