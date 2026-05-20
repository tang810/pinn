"""
Notebook-friendly eval file for 3D_packaging_liquid_cooling.

Online Jupyter usage:
1. Upload trained_model_3d_liquid_cooling.pt as a public asset.
2. Upload data_3d_liquid_cooling_simul.csv as a public asset (optional).
3. Fill MODEL_HASH and DATA_HASH, then run this whole file/cell.

This file loads the PINN model, evaluates the 3D temperature field,
and shows figures with plt.show(). It does not save images.

Self-contained — data is auto-generated if .csv not found.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = "8b12566940a44ff6be794d77dc9e73b5"
MODEL_HASH = "96743dc6d4a0434fa07a7390eec18b9b"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/trained_model_3d_liquid_cooling.pt")

SEED = 42; T_IN = 298.0
DATA_FILE_NAME = "data_3d_liquid_cooling_small.csv"
HIDDEN_DIM = 32
MAX_POINTS_PER_LABEL = 80


# =========================
# 2. Model
# =========================

class PINN3D(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, x, y, z):
        return self.net(torch.cat([x, y, z], dim=1))


# =========================
# 3. Data loading
# =========================

def load_data(data_path=None, device=None):
    data_path = os.path.expanduser(data_path) if data_path else ""
    if data_path and os.path.isfile(data_path):
        df = pd.read_csv(data_path)
    elif data_path and os.path.isdir(data_path):
        preferred = os.path.join(data_path, DATA_FILE_NAME)
        if os.path.isfile(preferred):
            df = pd.read_csv(preferred)
        else:
            for f in sorted(os.listdir(data_path)):
                if f.endswith('.csv'): df = pd.read_csv(os.path.join(data_path, f)); break
            else:
                rng = np.random.default_rng(SEED)
                n = 80
                def s(n_s, l, yr):
                    x=rng.random(n_s); y=rng.uniform(*yr,n_s); z=rng.random(n_s)
                    return pd.DataFrame({"x":x,"y":y,"z":z,"T":T_IN+20*np.exp(-((y-0.2)**2)*10),"label":l})
                df = pd.concat([s(n,0,(0.5,1)),s(n,1,(0,.5)),s(n//4,2,(.45,.55)),s(n//4,3,(0,.05)),s(n//4,4,(.9,1))])
    else:
        rng = np.random.default_rng(SEED); n=80
        def s(n_s,l,yr):
            x=rng.random(n_s); y=rng.uniform(*yr,n_s); z=rng.random(n_s)
            return pd.DataFrame({"x":x,"y":y,"z":z,"T":T_IN+20*np.exp(-((y-0.2)**2)*10),"label":l})
        df = pd.concat([s(n,0,(0.5,1)),s(n,1,(0,.5)),s(n//4,2,(.45,.55)),s(n//4,3,(0,.05)),s(n//4,4,(.9,1))])
    df = pd.concat(
        [
            group.sample(n=min(MAX_POINTS_PER_LABEL, len(group)), random_state=SEED)
            for _, group in df.groupby("label")
        ],
        ignore_index=True,
    )
    data = {}
    for lbl in sorted(df["label"].unique()):
        sub = df[df["label"] == lbl]
        x = torch.tensor(sub["x"].values[:,None], dtype=torch.float32, device=device)
        y = torch.tensor(sub["y"].values[:,None], dtype=torch.float32, device=device)
        z = torch.tensor(sub["z"].values[:,None], dtype=torch.float32, device=device)
        temp = torch.tensor(sub["T"].values[:,None], dtype=torch.float32, device=device)
        data[int(lbl)] = (x, y, z, temp)
    return data


# =========================
# 4. Model loading
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
        candidates.append(os.path.join(model_path, "trained_model_3d_liquid_cooling.pt"))
    if MODEL_HASH:
        candidates += [
            os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/trained_model_3d_liquid_cooling.pt"),
            os.path.expanduser(f"~/Resource/{MODEL_HASH}/trained_model_3d_liquid_cooling.pt"),
            f"/Resource/{MODEL_HASH}/trained_model_3d_liquid_cooling.pt",
        ]
    local_model = os.path.abspath(
        os.path.join(os.getcwd(), "Variant_PINNs", "3D_packaging_liquid_cooling", "model", "packaging_liquid_cooling_simul.pt")
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
            preferred = os.path.join(path, "trained_model_3d_liquid_cooling.pt")
            if os.path.isfile(preferred):
                return preferred
            pt_files = [os.path.join(path, name) for name in os.listdir(path) if name.lower().endswith(".pt")]
            if pt_files:
                return pt_files[0]
    raise FileNotFoundError(f"Model not found. Tried: {tried}")


def infer_hidden_dim(state_dict):
    weight = state_dict.get("net.0.weight")
    return int(weight.shape[0]) if weight is not None else HIDDEN_DIM


def load_checkpoint(model_path, device):
    model_path = resolve_model_path(model_path)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    hidden_dim = ckpt.get("hidden_dim", infer_hidden_dim(ckpt["solid"]))
    ms = PINN3D(hidden_dim=hidden_dim).to(device)
    mf = PINN3D(hidden_dim=hidden_dim).to(device)
    ms.load_state_dict(ckpt["solid"]); mf.load_state_dict(ckpt["fluid"])
    ms.eval(); mf.eval()
    return ms, mf, ckpt, model_path


# =========================
# 5. Visualization
# =========================

def show_eval_result(ms, data, history):
    if history:
        plt.figure(figsize=(7,4)); plt.semilogy(history); plt.xlabel("Epoch")
        plt.ylabel("Loss"); plt.title("Training Loss"); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.show()

    x, y, z, temp_true = data[0]
    with torch.no_grad():
        temp_pred = ms(x, y, z)
    rmse = float(torch.sqrt(torch.mean((temp_pred - temp_true)**2)).cpu())

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x.detach().cpu().numpy().reshape(-1), y.detach().cpu().numpy().reshape(-1),
                    z.detach().cpu().numpy().reshape(-1), c=temp_pred.detach().cpu().numpy().reshape(-1),
                    cmap="jet", s=6)
    fig.colorbar(sc, label="T (K)"); ax.set_title("3D Liquid Cooling Temperature Field")
    plt.show()

    print(f"RMSE: {rmse:.4f} K")
    return rmse


# =========================
# 6. Eval entry
# =========================

def evaluate(model_path=MODEL_PATH, data_path=DATA_PATH):
    device = get_device()
    print(f"device={device}")

    data = load_data(data_path, device=device)
    ms, mf, ckpt, model_path = load_checkpoint(model_path, device)
    history = ckpt.get("history", [])

    print("Eval result"); print(f"model_path: {model_path}")
    rmse = show_eval_result(ms, data, history)
    return {"rmse": rmse}


eval_result = evaluate()
