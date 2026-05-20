"""
Notebook-friendly train file for kdv_operator_learning.

Online Jupyter usage:
1. (Optional) Upload real_data.npz as a public asset.
2. Fill DATA_HASH and MODEL_HASH.
3. Run this whole file/cell. It saves trained_model_kdv_pinn.pt to
   ~/minio/Resource/{MODEL_HASH}/ for later public-asset upload.

Physics: KdV equation soliton solutions — an MLP learns the mapping x → u(x)
         from generated reference data. u(x) = a / cosh(x/λ)^2.

Self-contained — only depends on the .npz dataset.
"""

import os
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

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

DATA_HASH = "b653b27230f64aca84000bcd9290e824"
MODEL_HASH = "b653b27230f64aca84000bcd9290e824"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/minio/Resource/{MODEL_HASH}/trained_model_kdv_pinn.pt")


# =========================
# 2. KdV soliton data generation
# =========================

def generate_virtual_data(n_samples=80, n_x=256):
    min_x, max_x = -40, 40
    x = np.linspace(min_x, max_x, n_x, dtype=np.float32).reshape(1, -1)
    x = np.repeat(x, n_samples, axis=0)

    rho = np.random.uniform(0.9, 0.99, size=(n_samples, 1)).astype(np.float32)
    h1 = np.random.uniform(4, 10, size=(n_samples, 1)).astype(np.float32)
    a = np.random.uniform(-0.4, -0.05, size=(n_samples, 1)).astype(np.float32)

    h2 = 1.0; g = 9.81
    h = h2 / h1
    c0 = np.sqrt((g * h2 * (1 - rho)) / (h + rho))
    c1 = -1.5 * c0 * ((rho - h**2) / (rho * h2 + h * h2))
    c2 = (1.0 / 6.0) * c0 * ((rho * h2**2 * h1 + h1**2 * h2) / (rho * h1 + h2))
    lam = np.sqrt(12 * c2 / (a * c1))
    ref = a / (np.cosh(x / lam) ** 2)
    return x.astype(np.float32), ref.astype(np.float32), rho, h1, a


def load_data(data_path, n_samples=80, n_x=256):
    data_path = os.path.expanduser(data_path) if data_path else ""
    if data_path and os.path.isfile(data_path):
        d = np.load(data_path)
        if 'x' in d and 'ref' in d:
            return d['x'].astype(np.float32), d['ref'].astype(np.float32)
    print("Generating virtual KdV data...")
    x, ref, _, _, _ = generate_virtual_data(n_samples=n_samples, n_x=n_x)
    return x, ref


# =========================
# 3. Model (MLP)
# =========================

class KdV_PINN(nn.Module):
    def __init__(self, n_input=1, n_output=1, n_hidden=50, n_layers=3):
        super().__init__()
        layers = [nn.Linear(n_input, n_hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.Tanh()]
        layers += [nn.Linear(n_hidden, n_output)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# =========================
# 4. Visualization
# =========================

def show_train_result(model, x_data, ref_data, history, device):
    if history:
        plt.figure(figsize=(7, 4)); plt.plot(history); plt.yscale("log")
        plt.xlabel("Epoch"); plt.ylabel("MSE Loss"); plt.title("Training Loss")
        plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

    # Show first 4 samples
    n_show = min(4, x_data.shape[0])
    fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 3.5))
    if n_show == 1: axes = [axes]
    model.eval()
    with torch.no_grad():
        for i in range(n_show):
            x_i = torch.tensor(x_data[i:i+1].reshape(-1, 1), dtype=torch.float32, device=device)
            pred = model(x_i).cpu().numpy().reshape(-1)
            axes[i].plot(x_data[i], ref_data[i], 'k-', lw=2, label="Ref")
            axes[i].plot(x_data[i], pred, 'r--', lw=1.5, label="Pred")
            axes[i].set_xlabel("x"); axes[i].set_ylabel("u(x)")
            axes[i].set_title(f"Sample {i+1}"); axes[i].legend(fontsize=7); axes[i].grid(alpha=0.3)
    plt.suptitle("KdV PINN — Reference vs Prediction", fontsize=13)
    plt.tight_layout(); plt.show()


# =========================
# 5. Train entry
# =========================

def train(data_path=DATA_PATH, model_path=MODEL_PATH, n_samples=30, n_x=100,
          n_hidden=50, n_layers=3, epochs=50, lr=1e-3, batch_size=64):
    t0 = time.time()
    device = get_device()
    print(f"device={device}")

    x_data, ref_data = load_data(data_path, n_samples=n_samples, n_x=n_x)
    n_samples, n_x = x_data.shape
    print(f"Data: {n_samples} samples, {n_x} x-points")

    x_flat = torch.tensor(x_data.reshape(-1, 1), dtype=torch.float32, device=device)
    y_flat = torch.tensor(ref_data.reshape(-1, 1), dtype=torch.float32, device=device)
    n_total = x_flat.shape[0]

    model = KdV_PINN(n_hidden=n_hidden, n_layers=n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history = []

    print(f"Training {epochs} epochs...")
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n_total, device=device)
        total_loss = 0.0; n_batches = 0
        model.train()
        for i in range(0, n_total, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = x_flat[idx], y_flat[idx]
            pred = model(xb); loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += float(loss.detach().cpu()); n_batches += 1
        avg = total_loss / max(n_batches, 1); history.append(avg)
        if ep == 1 or ep % 20 == 0 or ep == epochs:
            print(f"Epoch {ep:4d}/{epochs} | loss={avg:.6e}")

    model_path = os.path.expanduser(model_path)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "history": history,
                "n_hidden": n_hidden, "n_layers": n_layers}, model_path)

    print("Training finished"); print(f"model saved to: {model_path}")
    print(f"elapsed: {time.time() - t0:.1f}s"); print(f"final loss: {history[-1]:.6e}")
    show_train_result(model, x_data, ref_data, history, device)
    return {"model": model, "model_path": model_path, "history": history}


train_result = train()
