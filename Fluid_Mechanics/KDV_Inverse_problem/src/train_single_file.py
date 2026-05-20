"""
Notebook-friendly train file for KDV_Inverse_problem.

Online Jupyter usage:
1. Upload heat_eq_data.npz as a public asset.
2. Fill DATA_HASH and MODEL_HASH.
3. Run this whole file/cell. It saves trained_model_kdv.pt to
   ~/minio/Resource/{MODEL_HASH}/ for later public-asset upload.

Physics: KdV inverse problem — FNN learns u(x,t) from exact soliton data.
         Input (x, t) → Output u(x,t). Exact: u(x,t) = a / cosh((x - c*t)/lambda)^2.

Self-contained PyTorch-only — no DeepXDE dependency.
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

DATA_HASH = "1babec8f54bb4393b8d4e89cd52fbcce"
MODEL_HASH = "1babec8f54bb4393b8d4e89cd52fbcce"
DATA_FILENAME = "heat_eq_data.npz"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/minio/Resource/{MODEL_HASH}/trained_model_kdv.pt")


# =========================
# 2. Model (FNN replica of DeepXDE FNN)
# =========================

class DeepXDEFNN(nn.Module):
    """PyTorch replica of dde.maps.FNN([2]+[20]*3+[1], 'tanh', 'Glorot uniform')."""
    def __init__(self, layer_sizes=(2, 20, 20, 20, 1)):
        super().__init__()
        self.linears = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)
        ])
        self.activation = torch.tanh

    def forward(self, x):
        for layer in self.linears[:-1]:
            x = self.activation(layer(x))
        return self.linears[-1](x)


# =========================
# 3. Data loading
# =========================

def load_data(data_path):
    data = np.load(data_path)
    required = {"x", "t", "usol"}
    missing = required.difference(data.files)
    if missing:
        raise KeyError(f"Dataset missing keys: {missing}. Found: {data.files}")
    x = data["x"].reshape(-1).astype(np.float32)
    t = data["t"].reshape(-1).astype(np.float32)
    u = data["usol"].astype(np.float32)
    if u.shape != (len(x), len(t)):
        raise ValueError(f"Expected usol shape ({len(x)},{len(t)}), got {u.shape}")
    x_grid, t_grid = np.meshgrid(x, t, indexing="ij")
    xt = np.stack([x_grid.reshape(-1), t_grid.reshape(-1)], axis=1).astype(np.float32)
    y = u.reshape(-1, 1).astype(np.float32)
    return xt, y, {"x": x, "t": t, "u": u}


def sample_batch(xt, y, batch_size, device):
    idx = np.random.choice(xt.shape[0], size=min(batch_size, xt.shape[0]), replace=False)
    return (torch.tensor(xt[idx], dtype=torch.float32, device=device),
            torch.tensor(y[idx], dtype=torch.float32, device=device))


# =========================
# 4. Visualization
# =========================

def show_train_result(model, xt, y, meta, history, device, n_show=3):
    if history:
        plt.figure(figsize=(7,4)); plt.plot([h["loss"] for h in history]); plt.yscale("log")
        plt.xlabel("Iter"); plt.ylabel("MSE Loss"); plt.title("Training Loss")
        plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

    t_show = [0.5, 1.0, 2.5]
    x = meta["x"]; t = meta["t"]
    model.eval()
    with torch.no_grad():
        for ti in t_show:
            idx = int(np.argmin(np.abs(t - ti)))
            xt_line = np.stack([x, np.full_like(x, t[idx])], axis=1).astype(np.float32)
            pred = model(torch.tensor(xt_line, dtype=torch.float32, device=device)).cpu().numpy()
            plt.plot(x, meta["u"][:, idx], 'k-', lw=2, label=f"true t={t[idx]:.2f}" if ti==t_show[0] else "")
            plt.plot(x, pred.reshape(-1), 'r--', lw=1.5)
    plt.xlabel("x"); plt.ylabel("u(x,t)"); plt.title("KDV PINN — Reference vs Prediction")
    plt.legend(["Ref", "Pred"]); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()


# =========================
# 5. Train entry
# =========================

def train(data_path=DATA_PATH, model_path=MODEL_PATH, n_iters=1000, print_every=100,
          batch_size=4096, lr=1e-3, seed=42):
    t0 = time.time(); torch.manual_seed(seed); np.random.seed(seed)
    device = get_device()
    print(f"device={device}")

    data_path = os.path.expanduser(data_path) if data_path else ""
    candidates = []
    if data_path:
        candidates.append(data_path)
        if os.path.isdir(data_path) or not os.path.splitext(data_path)[1]:
            candidates.append(os.path.join(data_path, DATA_FILENAME))
    candidates.extend([
        os.path.expanduser(f"~/public/Resource/{DATA_HASH}/{DATA_FILENAME}"),
        os.path.expanduser(f"~/Resource/{DATA_HASH}/{DATA_FILENAME}"),
        f"/Resource/{DATA_HASH}/{DATA_FILENAME}",
    ])
    for candidate in candidates:
        candidate = os.path.abspath(os.path.normpath(candidate))
        if os.path.isfile(candidate):
            data_path = candidate
            break
    else:
        raise FileNotFoundError("Data file not found in public assets. Expected one of: " + ", ".join(candidates))
    xt, y, meta = load_data(data_path)
    print(f"Data: {xt.shape[0]} pts, x={len(meta['x'])}, t={len(meta['t'])}")

    model = DeepXDEFNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    print(f"Training {n_iters} iters...")
    for it in range(1, n_iters + 1):
        xb, yb = sample_batch(xt, y, batch_size, device)
        pred = model(xb); loss = torch.mean((pred - yb)**2)
        opt.zero_grad(); loss.backward(); opt.step()
        if it == 1 or it % print_every == 0 or it == n_iters:
            history.append({"iter": it, "loss": float(loss.detach().cpu())})
            print(f"Iter {it:5d}/{n_iters} | loss={history[-1]['loss']:.6e}")

    model_path = os.path.expanduser(model_path)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)

    print("Training finished"); print(f"model saved to: {model_path}")
    print(f"elapsed: {time.time() - t0:.1f}s")
    show_train_result(model, xt, y, meta, history, device)
    return {"model": model, "model_path": model_path, "history": history}


train_result = train()
