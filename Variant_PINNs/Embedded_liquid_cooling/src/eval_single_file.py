"""
Notebook-friendly eval file for Embedded_liquid_cooling.

Online Jupyter usage:
1. Upload trained_model_embedded_liquid_cooling.pt as a public asset.
2. Fill MODEL_HASH and DATA_HASH, then run this whole file/cell.

This file loads the PINN model, evaluates 2D solid-fluid thermal fields,
and shows figures with plt.show(). It does not save images.

Self-contained — evaluates on a dense [0,1]^2 grid.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = ""
MODEL_HASH = ""

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/trained_model_embedded_liquid_cooling.pt")
HIDDEN_DIM = 32
GRID_RES = 60


# =========================
# 2. Model
# =========================

class PINN_Net(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))


# =========================
# 3. Model loading
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


def resolve_model_path(model_path):
    candidates = []
    if model_path:
        model_path = os.path.expanduser(model_path)
        candidates.append(model_path)
        candidates.append(os.path.join(model_path, "trained_model_embedded_liquid_cooling.pt"))
    if MODEL_HASH:
        candidates += [
            os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/trained_model_embedded_liquid_cooling.pt"),
            os.path.expanduser(f"~/Resource/{MODEL_HASH}/trained_model_embedded_liquid_cooling.pt"),
            f"/Resource/{MODEL_HASH}/trained_model_embedded_liquid_cooling.pt",
        ]
    local_model = os.path.abspath(
        os.path.join(os.getcwd(), "Variant_PINNs", "Embedded_liquid_cooling", "model", "embedded_liquid_cooling_simul.pt")
    )
    candidates.append(local_model)

    tried = []
    for path in candidates:
        if not path or path in tried:
            continue
        tried.append(path)
        if os.path.isfile(path):
            return path
        if os.path.isdir(path):
            preferred = os.path.join(path, "trained_model_embedded_liquid_cooling.pt")
            if os.path.isfile(preferred):
                return preferred
            pt_files = [os.path.join(path, name) for name in os.listdir(path) if name.lower().endswith(".pt")]
            if pt_files:
                return pt_files[0]
    raise FileNotFoundError(f"Model not found. Tried: {tried}")


def infer_hidden_dim(state_dict):
    first_weight = state_dict.get("net.0.weight")
    if first_weight is None:
        return HIDDEN_DIM
    return int(first_weight.shape[0])


def load_checkpoint(model_path, device):
    model_path = resolve_model_path(model_path)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    hidden_dim = ckpt.get("hidden_dim", infer_hidden_dim(ckpt["model_s"]))
    ms = PINN_Net(hidden_dim=hidden_dim).to(device)
    mf = PINN_Net(hidden_dim=hidden_dim).to(device)
    ms.load_state_dict(ckpt["model_s"]); mf.load_state_dict(ckpt["model_f"])
    ms.eval(); mf.eval()
    return ms, mf, ckpt, model_path


# =========================
# 4. Visualization
# =========================

def show_eval_result(ms, mf, history, device):
    if history:
        plt.figure(figsize=(7,4)); plt.semilogy(history)
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training Loss")
        plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

    x = np.linspace(0, 1, GRID_RES); y = np.linspace(0, 1, GRID_RES)
    X, Y = np.meshgrid(x, y)
    xt = torch.tensor(X.reshape(-1, 1), dtype=torch.float32, device=device)
    yt = torch.tensor(Y.reshape(-1, 1), dtype=torch.float32, device=device)
    with torch.no_grad():
        Ts = ms(xt, yt).cpu().numpy().reshape(GRID_RES, GRID_RES)
        Tf = mf(xt, yt).cpu().numpy().reshape(GRID_RES, GRID_RES)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, val, title in [(axes[0], Ts, "Solid"), (axes[1], Tf, "Fluid")]:
        im = ax.contourf(X, Y, val, levels=30, cmap="jet")
        ax.set_title(f"{title} Thermal Field"); ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.colorbar(im, ax=ax)
    fig.suptitle("Embedded Liquid Cooling PINN Eval", fontsize=13)
    plt.tight_layout(); plt.show()

    print(f"Solid T range: [{Ts.min():.2f}, {Ts.max():.2f}] K")
    print(f"Fluid T range: [{Tf.min():.2f}, {Tf.max():.2f}] K")


# =========================
# 5. Eval entry
# =========================

def evaluate(model_path=MODEL_PATH, data_path=DATA_PATH):
    device = get_device()
    print(f"device={device}")

    ms, mf, ckpt, model_path = load_checkpoint(model_path, device)
    history = ckpt.get("history", [])

    print("Eval result"); print(f"model_path: {model_path}")
    show_eval_result(ms, mf, history, device)
    return {"ok": True}


eval_result = evaluate()
