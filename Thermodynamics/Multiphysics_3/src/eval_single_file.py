"""
Notebook-friendly eval file for Multiphysics_3.

Uses shared data_7246.txt format and loads trained_model_multiphysics3.pt.
Self-contained: no imports from local src/ modules.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


DATA_HASH = "1ffd9719549b48b89c21e364a9742b29"
MODEL_HASH = "fdc70f68a4e14945877c3ae718672834"
DATA_FILE_NAME = "data_7246.txt"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/trained_model_multiphysics3.pt")


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
    candidates.append(os.path.join(os.getcwd(), "Thermodynamics", "Multiphysics", "data", DATA_FILE_NAME))
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
            f"/mnt/minio/userdk8e2v7l/Resource/{DATA_HASH}",
            f"/mnt/minio/userdk8e2v7l/Resource/{DATA_HASH}/{DATA_FILE_NAME}",
            f"/mnt/minio/userdk8e2v7l/public/Resource/{DATA_HASH}",
            f"/mnt/minio/userdk8e2v7l/public/Resource/{DATA_HASH}/{DATA_FILE_NAME}",
        ]
    candidates += [
        os.path.join(os.getcwd(), "Thermodynamics", "Multiphysics_3", "data", DATA_FILE_NAME),
    ]
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
        candidates += [expanded, os.path.join(expanded, "trained_model_multiphysics3.pt")]
    if MODEL_HASH:
        candidates += [
            os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/trained_model_multiphysics3.pt"),
            os.path.expanduser(f"~/Resource/{MODEL_HASH}/trained_model_multiphysics3.pt"),
            f"/Resource/{MODEL_HASH}/trained_model_multiphysics3.pt",
        ]
    candidates.append(os.path.join(os.getcwd(), "Thermodynamics", "Multiphysics_3", "model", "pinnsformer_inverse.pt"))
    tried = []
    for path in candidates:
        if not path or path in tried:
            continue
        tried.append(path)
        if os.path.isfile(path):
            return path
        if os.path.isdir(path):
            preferred = os.path.join(path, "trained_model_multiphysics3.pt")
            if os.path.isfile(preferred):
                return preferred
            pt_files = [os.path.join(path, name) for name in os.listdir(path) if name.lower().endswith(".pt")]
            if pt_files:
                return pt_files[0]
    raise FileNotFoundError(f"Model not found. Tried: {tried}")


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


def load_data(data_path):
    path = resolve_data_path(data_path)
    arr = np.loadtxt(path, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 5:
        raise ValueError(f"{DATA_FILE_NAME} must have columns x y z t T, got {arr.shape}")
    print("data:", path)
    return arr[:, :5]


def load_model(model_path, device):
    model_path = resolve_model_path(model_path)
    ckpt = load_torch(model_path, device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    hidden_dim = ckpt.get("hidden_dim", 64) if isinstance(ckpt, dict) else 64
    model = SimpleMultiphysicsNet(hidden_dim).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, ckpt if isinstance(ckpt, dict) else {}, model_path


def evaluate(model_path=MODEL_PATH, data_path=DATA_PATH):
    device = get_device()
    print(f"device={device}")
    model, ckpt, model_path = load_model(model_path, device)
    arr = load_data(data_path)

    x_raw = arr[:, :4]
    y_true = arr[:, 4:5]
    x_mean = ckpt.get("x_mean", x_raw.mean(axis=0, keepdims=True))
    x_std = ckpt.get("x_std", x_raw.std(axis=0, keepdims=True) + 1e-6)
    y_mean = ckpt.get("y_mean", y_true.mean(axis=0, keepdims=True))
    y_std = ckpt.get("y_std", y_true.std(axis=0, keepdims=True) + 1e-6)
    X = torch.tensor((x_raw - x_mean) / x_std, dtype=torch.float32, device=device)
    with torch.no_grad():
        y_pred = model(X).cpu().numpy() * y_std + y_mean

    err = np.abs(y_true - y_pred)
    mae = float(np.mean(err))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    print("Eval result")
    print("model_path:", model_path)
    print(f"MAE: {mae:.4f} K, RMSE: {rmse:.4f} K")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    history = ckpt.get("history", [])
    if history:
        axes[0].semilogy(np.maximum(history, 1e-12))
    axes[0].set_title("Training Loss")
    axes[0].grid(alpha=0.3)
    axes[1].scatter(y_true, y_pred, s=8, alpha=0.5)
    axes[1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=1)
    axes[1].set_title("True vs Pred")
    axes[2].hist(err.reshape(-1), bins=40)
    axes[2].set_title("Error Distribution")
    plt.tight_layout()
    plt.show()
    return {"mae": mae, "rmse": rmse}


eval_result = evaluate()
