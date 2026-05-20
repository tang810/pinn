"""
Notebook-friendly train file for DEV-DiT4Science.

Online Jupyter usage:
1. (Optional) Upload simul_airfoil_flow.npz as a public asset.
2. Fill DATA_HASH and MODEL_HASH.
3. Run this whole file/cell. It saves dev_dit4science_surrogate.pt to
   ~/minio/Resource/{MODEL_HASH}/ for later public-asset upload.

Method: Linear regression surrogate for airfoil flow.
         4 inputs (α, logRe, x, y) → 3 outputs (p, u, v).
         Trained with gradient descent (MSE).

Self-contained — only depends on the .npz dataset.
"""

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import torch



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

# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = "811530c549444a5f9c77de677bfe085c"
MODEL_HASH = "811530c549444a5f9c77de677bfe085c"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/minio/Resource/{MODEL_HASH}/dev_dit4science_surrogate.pt")


# =========================
# 2. Data generation / loading
# =========================

def make_synthetic_data(n=2048, seed=42):
    rng = np.random.default_rng(seed)
    alpha = rng.uniform(-5.0, 15.0, size=(n, 1))
    reynolds = rng.uniform(1e5, 5e6, size=(n, 1))
    x = rng.uniform(0.0, 1.0, size=(n, 1))
    y = rng.uniform(-0.3, 0.3, size=(n, 1))
    features = np.hstack([alpha / 15.0, np.log10(reynolds) / 7.0, x, y]).astype(np.float32)
    p = (1.2 - x) * (1.0 + 0.02 * alpha) - 0.8 * y**2
    u = (1.0 + 0.05 * alpha) * (1.0 - 0.15 * y**2)
    v = 0.35 * y * (x - 0.5) + 0.01 * alpha
    targets = np.hstack([p, u, v]).astype(np.float32)
    return features, targets


def load_data(data_path, seed=42):
    data_path = os.path.expanduser(data_path) if data_path else ""
    if data_path and os.path.isfile(data_path):
        d = np.load(data_path)
        return d["x"].astype(np.float32), d["y"].astype(np.float32)
    print("Generating synthetic data...")
    return make_synthetic_data(seed=seed)


# =========================
# 3. Linear regression model
# =========================

def fit_linear(x, y, epochs=300, lr=1e-2):
    n, d = x.shape; out_dim = y.shape[1]
    w = np.zeros((d, out_dim), dtype=np.float32)
    b = np.zeros((1, out_dim), dtype=np.float32)
    losses = []
    for _ in range(epochs):
        pred = x @ w + b; err = pred - y
        loss = float(np.mean(err * err)); losses.append(loss)
        w -= lr * (2.0 / n) * (x.T @ err).astype(np.float32)
        b -= lr * (2.0 / n) * np.sum(err, axis=0, keepdims=True).astype(np.float32)
    return {"w": w, "b": b, "losses": np.array(losses, dtype=np.float32)}


def predict(x, model):
    return x @ model["w"] + model["b"]


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0, keepdims=True))**2, axis=0) + 1e-12
    return 1.0 - ss_res / ss_tot


# =========================
# 4. Visualization
# =========================

def show_train_result(model, x_val, y_val, y_pred):
    losses = model["losses"]
    if len(losses):
        plt.figure(figsize=(7, 4)); plt.plot(losses); plt.yscale("log")
        plt.xlabel("Epoch"); plt.ylabel("MSE Loss"); plt.title("Training Loss")
        plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

    r2 = r2_score(y_val, y_pred)
    labels = ["p", "u", "v"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for i, (ax, name) in enumerate(zip(axes, labels)):
        ax.scatter(y_val[:, i], y_pred[:, i], s=6, alpha=0.5)
        lo = min(y_val[:, i].min(), y_pred[:, i].min())
        hi = max(y_val[:, i].max(), y_pred[:, i].max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_xlabel(f"True {name}"); ax.set_ylabel(f"Pred {name}")
        ax.set_title(f"{name} (R2={r2[i]:.4f})"); ax.grid(alpha=0.3)
    plt.suptitle("DEV-DiT4Science: Linear Surrogate", fontsize=13)
    plt.tight_layout(); plt.show()
    print(f"R2: p={r2[0]:.4f} u={r2[1]:.4f} v={r2[2]:.4f}")
    return r2


# =========================
# 5. Train entry
# =========================

def train(data_path=DATA_PATH, model_path=MODEL_PATH, epochs=300, lr=1e-2, seed=42):
    x, y = load_data(data_path, seed=seed)
    n_train = int(x.shape[0] * 0.8)
    x_train, y_train = x[:n_train], y[:n_train]
    x_val, y_val = x[n_train:], y[n_train:]
    print(f"Data: {x.shape[0]} pts, {n_train} train, {x.shape[0] - n_train} val")

    model = fit_linear(x_train, y_train, epochs=epochs, lr=lr)
    y_pred = predict(x_val, model)
    val_mse = float(np.mean((y_pred - y_val)**2))
    val_r2 = r2_score(y_val, y_pred)

    model_path = os.path.expanduser(model_path)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump({"w": model["w"], "b": model["b"]}, f)

    print("Training finished"); print(f"model saved to: {model_path}")
    print(f"val_mse={val_mse:.6e}, val_r2_macro={float(np.mean(val_r2)):.4f}")
    show_train_result(model, x_val, y_val, y_pred)
    return {"model": model, "model_path": model_path, "val_r2": val_r2}


train_result = train()
