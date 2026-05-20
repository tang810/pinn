"""
Notebook-friendly train file for Multiphysics_3.

Uses the shared dataset Thermodynamics/Multiphysics/data/data_7246.txt format
(x, y, z, t, T). Self-contained: no imports from local src/ modules.
"""

import os
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


DATA_HASH = "1ffd9719549b48b89c21e364a9742b29"
MODEL_HASH = "1ffd9719549b48b89c21e364a9742b29"
DATA_FILE_NAME = "data_7246.txt"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/minio/Resource/{MODEL_HASH}/trained_model_multiphysics3.pt")

EPOCHS = 300
PRINT_EVERY = 50
BATCH_SIZE = 128
HIDDEN_DIM = 64
LR = 1e-3
SEED = 42


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


def candidate_data_paths(data_path):
    items = []
    items.append(os.path.join(os.getcwd(), "Thermodynamics", "Multiphysics", "data", DATA_FILE_NAME))
    if data_path:
        expanded = os.path.expanduser(data_path)
        items += [expanded, os.path.join(expanded, DATA_FILE_NAME)]
    if DATA_HASH:
        items += [
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
    items += [
        os.path.join(os.getcwd(), "Thermodynamics", "Multiphysics_3", "data", DATA_FILE_NAME),
    ]
    out = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


def resolve_data_path(data_path):
    tried = []
    for path in candidate_data_paths(data_path):
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


def load_data(data_path):
    try:
        path = resolve_data_path(data_path)
        arr = np.loadtxt(path, dtype=np.float32)
        print("data:", path)
    except FileNotFoundError as exc:
        print(exc)
        print("data not found, generating synthetic data_7246-style samples")
        rng = np.random.default_rng(SEED)
        n = 100
        x = rng.uniform(0, 1, n)
        y = rng.uniform(0, 1, n)
        z = rng.uniform(0, 1, n)
        t = rng.uniform(0, 1, n)
        T = 273.15 + 30.0 * np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2) / 0.1)
        arr = np.column_stack([x, y, z, t, T]).astype(np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 5:
        raise ValueError(f"{DATA_FILE_NAME} must have columns x y z t T, got {arr.shape}")
    arr = arr[:, :5]
    return arr


class SimpleMultiphysicsNet(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM):
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


def train(data_path=DATA_PATH, model_path=MODEL_PATH, epochs=EPOCHS):
    t0 = time.time()
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = get_device()
    print(f"device={device}")

    arr = load_data(data_path)
    x_raw = arr[:, :4]
    y_raw = arr[:, 4:5]
    x_mean = x_raw.mean(axis=0, keepdims=True)
    x_std = x_raw.std(axis=0, keepdims=True) + 1e-6
    y_mean = y_raw.mean(axis=0, keepdims=True)
    y_std = y_raw.std(axis=0, keepdims=True) + 1e-6
    X = torch.tensor((x_raw - x_mean) / x_std, dtype=torch.float32, device=device)
    Y = torch.tensor((y_raw - y_mean) / y_std, dtype=torch.float32, device=device)

    model = SimpleMultiphysicsNet(HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    history = []

    print(f"Training {epochs} epochs, samples={len(X)}")
    for ep in range(1, epochs + 1):
        perm = torch.randperm(len(X), device=device)
        losses = []
        for start in range(0, len(X), BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_fn(model(X[idx]), Y[idx])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)))
        if ep == 1 or ep % PRINT_EVERY == 0 or ep == epochs:
            print(f"Epoch {ep:4d}/{epochs} | loss={history[-1]:.6e}")

    model_path = os.path.expanduser(model_path)
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "hidden_dim": HIDDEN_DIM,
        "history": history,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }, model_path)

    print("Training finished")
    print(f"model saved to: {model_path}")
    print(f"elapsed: {time.time() - t0:.1f}s")

    plt.figure(figsize=(7, 4))
    plt.semilogy(np.maximum(history, 1e-12))
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Multiphysics_3 Training Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    return {"model": model, "model_path": model_path, "history": history}


train_result = train()
