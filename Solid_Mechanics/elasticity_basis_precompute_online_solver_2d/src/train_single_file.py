"""
Notebook-friendly train file for elasticity_basis_precompute_online_solver_2d.

Online Jupyter usage:
1. Upload one .npz elasticity dataset as a public asset.
2. Fill DATA_HASH and MODEL_HASH.
3. Run this whole file/cell. It saves trained_model_elasticity_basis.pt to
   ~/minio/Resource/{MODEL_HASH}/ for later public-asset upload.

Expected dataset file:
    real_data.npz

The .npz file must contain x, y, q, ux, uy arrays.
"""

import os
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = "b536492325fa4d2ea77495db59dc626a"
MODEL_HASH = "b536492325fa4d2ea77495db59dc626a"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/minio/Resource/{MODEL_HASH}/trained_model_elasticity_basis.pt")

PREFERRED_DATA_FILE = "real_data.npz"


# =========================
# 2. Model definition
# =========================

class ElasticPINN(nn.Module):
    def __init__(self, in_dim=3, hidden=64, depth=3, out_dim=2):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# =========================
# 3. Data utilities
# =========================

def resolve_npz_path(data_path):
    if not data_path:
        raise FileNotFoundError("DATA_PATH is empty. Fill DATA_HASH first.")

    data_path = os.path.expanduser(data_path)
    candidates = []
    if os.path.isfile(data_path):
        candidates.append(data_path)
    elif os.path.isdir(data_path):
        candidates.append(os.path.join(data_path, PREFERRED_DATA_FILE))
        candidates.append(os.path.join(data_path, "real_data.npz"))
        for f in sorted(os.listdir(data_path)):
            if f.endswith(".npz"):
                candidates.append(os.path.join(data_path, f))

    for path in candidates:
        if os.path.isfile(path):
            return path

    raise FileNotFoundError(
        "Could not find elasticity .npz data. Upload a file like real_data.npz into the data asset folder. "
        f"Current DATA_PATH={data_path!r}"
    )


def load_elasticity_data(npz_path):
    data = np.load(npz_path)
    required = ["x", "y", "q", "ux", "uy"]
    for k in required:
        if k not in data:
            raise KeyError(f"The .npz file must contain {k}. Missing: {k}")
    x = data["x"].astype(np.float32).reshape(-1, 1)
    y = data["y"].astype(np.float32).reshape(-1, 1)
    q = data["q"].astype(np.float32).reshape(-1, 1)
    ux = data["ux"].astype(np.float32).reshape(-1, 1)
    uy = data["uy"].astype(np.float32).reshape(-1, 1)
    inp = np.hstack([x, y, q])
    out = np.hstack([ux, uy])
    return inp, out


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 4. Visualization
# =========================

def show_result(inputs, outputs, pred, history):
    ux_true = outputs[:, 0]
    ux_pred = pred[:, 0]
    x_axis = inputs[:, 0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history, linewidth=1)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(alpha=0.3)

    idx = np.argsort(x_axis)[: min(512, len(x_axis))]
    axes[1].plot(x_axis[idx], ux_true[idx], label="true", linewidth=2)
    axes[1].plot(x_axis[idx], ux_pred[idx], "--", label="pred", linewidth=1.8)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("u_x")
    axes[1].set_title("Displacement u_x")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].scatter(outputs.reshape(-1), pred.reshape(-1), s=6, alpha=0.4)
    lo = min(float(outputs.min()), float(pred.min()))
    hi = max(float(outputs.max()), float(pred.max()))
    axes[2].plot([lo, hi], [lo, hi], "r--", linewidth=1)
    axes[2].set_xlabel("True")
    axes[2].set_ylabel("Predicted")
    axes[2].set_title("Prediction Scatter")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# =========================
# 5. Train entry
# =========================

def train(data_path=DATA_PATH, model_path=MODEL_PATH,
          hidden=64, depth=3, epochs=300, batch_size=256, lr=1e-3):
    t0 = time.time()
    device = get_device()

    npz_path = resolve_npz_path(data_path)
    inputs, outputs = load_elasticity_data(npz_path)

    x_mean = inputs.mean(axis=0, keepdims=True)
    x_std = inputs.std(axis=0, keepdims=True) + 1e-8
    y_mean = outputs.mean(axis=0, keepdims=True)
    y_std = outputs.std(axis=0, keepdims=True) + 1e-8
    Xn = (inputs - x_mean) / x_std
    Yn = (outputs - y_mean) / y_std

    Xt = torch.tensor(Xn, dtype=torch.float32, device=device)
    Yt = torch.tensor(Yn, dtype=torch.float32, device=device)

    model = ElasticPINN(in_dim=3, hidden=hidden, depth=depth, out_dim=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history = []

    n = Xn.shape[0]
    print(f"device={device}")
    print(f"data={npz_path}")
    print(f"X={inputs.shape}, Y={outputs.shape}")
    print(f"Training {epochs} epochs...")

    for ep in range(1, epochs + 1):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        model.train()
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = Xt[idx]
            yb = Yt[idx]
            pred_b = model(xb)
            loss = loss_fn(pred_b, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
        avg_loss = total_loss / max(1, (n + batch_size - 1) // batch_size)
        history.append(avg_loss)
        if ep == 1 or ep % 50 == 0 or ep == epochs:
            print(f"Epoch {ep:4d}/{epochs} | loss={avg_loss:.6e}")

    model.eval()
    with torch.no_grad():
        pred_n = model(Xt).cpu().numpy()
    pred = pred_n * y_std + y_mean

    model_path = os.path.expanduser(model_path)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    checkpoint = {
        "project": "elasticity_basis_precompute_online_solver_2d",
        "state_dict": model.state_dict(),
        "in_dim": 3,
        "out_dim": 2,
        "hidden": hidden,
        "depth": depth,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "history": history,
        "npz_path": npz_path,
        "data_shape": list(inputs.shape),
    }
    torch.save(checkpoint, model_path)

    print("Training finished")
    print("dataset:", npz_path)
    print("model saved to:", model_path)
    print("shape:", inputs.shape)
    print("elapsed:", f"{time.time() - t0:.1f}s")

    err = pred - outputs
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    print(f"MAE: {mae:.6e}")
    print(f"RMSE: {rmse:.6e}")

    show_result(inputs, outputs, pred, history)
    return {"model": model, "model_path": model_path, "data_path": npz_path, "history": history}


train_result = train()
