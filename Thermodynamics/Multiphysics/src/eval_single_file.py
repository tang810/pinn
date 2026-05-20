"""
Notebook-friendly eval file for Multiphysics.

Upload trained_model_multiphysics.pt as a public model asset and data_7246.txt
as a public data asset, fill MODEL_HASH/DATA_HASH, then run this whole file.

Self-contained: no imports from local src/ modules.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


# =========================
# 1. Parameters
# =========================

DATA_HASH = "1ffd9719549b48b89c21e364a9742b29"
MODEL_HASH = "bb4d02c297fa42628b433e017d0d4ef3"
DATA_FILE_NAME = "data_7246.txt"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/trained_model_multiphysics.pt")


# =========================
# 2. Utilities
# =========================

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


def load_torch(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def resolve_data_path(data_path):
    candidates = []
    if data_path:
        expanded = os.path.expanduser(data_path)
        candidates += [expanded, os.path.join(expanded, DATA_FILE_NAME)]
    if DATA_HASH:
        candidates += [
            os.path.expanduser(f"~/public/Resource/{DATA_HASH}"),
            os.path.expanduser(f"~/public/Resource/{DATA_HASH}/{DATA_FILE_NAME}"),
            os.path.expanduser(f"~/Resource/{DATA_HASH}"),
            os.path.expanduser(f"~/Resource/{DATA_HASH}/{DATA_FILE_NAME}"),
            f"/Resource/{DATA_HASH}",
            f"/Resource/{DATA_HASH}/{DATA_FILE_NAME}",
        ]
    candidates.append(os.path.join(os.getcwd(), "Thermodynamics", "Multiphysics", "data", DATA_FILE_NAME))
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
            txt_files = [os.path.join(path, name) for name in os.listdir(path) if name.lower().endswith(".txt")]
            if txt_files:
                return txt_files[0]
    raise FileNotFoundError(f"Cannot find {DATA_FILE_NAME}. Tried: {tried}")


def resolve_model_path(model_path):
    candidates = []
    if model_path:
        expanded = os.path.expanduser(model_path)
        candidates += [expanded, os.path.join(expanded, "trained_model_multiphysics.pt")]
    if MODEL_HASH:
        candidates += [
            os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/trained_model_multiphysics.pt"),
            os.path.expanduser(f"~/Resource/{MODEL_HASH}/trained_model_multiphysics.pt"),
            f"/Resource/{MODEL_HASH}/trained_model_multiphysics.pt",
        ]
    candidates.append(os.path.join(os.getcwd(), "Thermodynamics", "Multiphysics", "model", "pinnsformer_withsun.pt"))
    tried = []
    for path in candidates:
        if not path or path in tried:
            continue
        tried.append(path)
        if os.path.isfile(path):
            return path
        if os.path.isdir(path):
            preferred = os.path.join(path, "trained_model_multiphysics.pt")
            if os.path.isfile(preferred):
                return preferred
            pt_files = [os.path.join(path, name) for name in os.listdir(path) if name.lower().endswith(".pt")]
            if pt_files:
                return pt_files[0]
    raise FileNotFoundError(f"Model not found. Tried: {tried}")


def generate_small_data(n=400):
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, n)
    y = rng.uniform(0, 1, n)
    z = rng.uniform(0, 1, n)
    t = rng.uniform(0, 1, n)
    T = 273.15 + 30.0 * np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2) / 0.1)
    T = T * (1.0 + 0.03 * np.sin(2.0 * np.pi * t))
    return np.column_stack([x, y, z, t, T]).astype(np.float32)


def load_eval_data(data_path):
    try:
        path = resolve_data_path(data_path)
        arr = np.loadtxt(path, dtype=np.float32)
        print("data:", path)
    except FileNotFoundError:
        arr = generate_small_data()
        print("data: generated synthetic small dataset")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 5:
        raise ValueError(f"{DATA_FILE_NAME} must have at least 5 columns: x y z t T, got {arr.shape}")
    return arr[:, :5].astype(np.float32)


# =========================
# 3. Model
# =========================

class SimpleMultiphysicsNet(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


def load_model_package(model_path, device):
    model_path = resolve_model_path(model_path)
    ckpt = load_torch(model_path, device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    hidden_dim = ckpt.get("hidden_dim", 64) if isinstance(ckpt, dict) else 64
    model = SimpleMultiphysicsNet(hidden_dim=hidden_dim).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, ckpt if isinstance(ckpt, dict) else {}, model_path


# =========================
# 4. Eval
# =========================

def evaluate(model_path=MODEL_PATH, data_path=DATA_PATH):
    device = get_device()
    print(f"device={device}")

    model, ckpt, model_path = load_model_package(model_path, device)
    arr = load_eval_data(data_path)
    X_raw = arr[:, :4]
    T_true = arr[:, 4:5]

    x_mean = ckpt.get("x_mean", X_raw.mean(axis=0, keepdims=True))
    x_std = ckpt.get("x_std", X_raw.std(axis=0, keepdims=True) + 1e-6)
    y_mean = ckpt.get("y_mean", T_true.mean(axis=0, keepdims=True))
    y_std = ckpt.get("y_std", T_true.std(axis=0, keepdims=True) + 1e-6)
    X = (X_raw - x_mean) / x_std

    with torch.no_grad():
        pred_norm = model(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy()
    T_pred = pred_norm * y_std + y_mean
    err = np.abs(T_true - T_pred)
    mae = float(np.mean(err))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    print("Eval result")
    print("model_path:", model_path)
    print(f"data points: {len(T_true)}, T range: [{T_true.min():.1f}, {T_true.max():.1f}]")
    print(f"MAE: {mae:.4f} K, RMSE: {rmse:.4f} K")

    history = ckpt.get("history", [])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    if history:
        axes[0].semilogy(np.maximum(history, 1e-12))
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.3)

    axes[1].scatter(T_true, T_pred, s=8, alpha=0.5)
    axes[1].plot([T_true.min(), T_true.max()], [T_true.min(), T_true.max()], "r--", lw=1)
    axes[1].set_xlabel("True T (K)")
    axes[1].set_ylabel("Pred T (K)")
    axes[1].set_title("True vs Pred")
    axes[1].grid(alpha=0.3)

    axes[2].hist(err.reshape(-1), bins=40)
    axes[2].set_xlabel("Absolute Error (K)")
    axes[2].set_title("Error Distribution")
    plt.tight_layout()
    plt.show()
    return {"mae": mae, "rmse": rmse}


eval_result = evaluate()
