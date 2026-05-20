"""
Notebook-friendly eval file for airfoil_disp_pinn.

Online Jupyter usage:
1. Upload trained_model_airfoil_disp.pt as a public asset.
2. Upload data.mat as a public asset.
3. Fill MODEL_HASH and DATA_HASH, then run this whole file/cell.

This file loads the PINN model, evaluates 3D airfoil displacement (u,v,w),
and shows figures with plt.show(). It does not save images.

Self-contained — only depends on the .mat dataset and .pt model.
"""

import os
from collections import OrderedDict

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = "8ba4730343624aa08babb31bfc9de5a7"
MODEL_HASH = "03fc64f9eac34c40a91749eea6935b98"
DATA_FILENAME = "data.mat"
MODEL_FILENAME = "trained_model_airfoil_disp.pt"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/trained_model_airfoil_disp.pt")


# =========================
# 2. Model (FCNet)
# =========================

class FCNet(nn.Module):
    def __init__(self, num_ins=3, num_outs=3, num_layers=6, hidden_size=50):
        super().__init__()
        sizes = [num_ins] + [hidden_size] * num_layers + [num_outs]
        layer_list = []
        for i in range(len(sizes) - 2):
            layer_list.append((f"layer_{i}", nn.Linear(sizes[i], sizes[i + 1])))
            layer_list.append((f"activation_{i}", nn.Tanh()))
        layer_list.append((f"layer_{len(sizes) - 2}", nn.Linear(sizes[-2], sizes[-1])))
        self.layers = nn.Sequential(OrderedDict(layer_list))

    def forward(self, x):
        return self.layers(x)


# =========================
# 3. Data loading
# =========================

def load_eval_data(filename):
    data = scipy.io.loadmat(filename)
    x_s = data['x']; y_s = data['y']; z_s = data['z'].astype(np.float32)
    u_s = data['u_ref']; v_s = data['v_ref']; w_s = data['w_ref']
    return x_s, y_s, z_s, u_s, v_s, w_s


def resolve_public_file(path, hash_value, filename, label):
    raw_path = os.path.expanduser(path) if path else ""
    candidates = []
    if raw_path:
        candidates.append(raw_path)
        if os.path.isdir(raw_path) or not os.path.splitext(raw_path)[1]:
            candidates.append(os.path.join(raw_path, filename))
    candidates.extend([
        os.path.expanduser(f"~/public/Resource/{hash_value}/{filename}"),
        os.path.expanduser(f"~/Resource/{hash_value}/{filename}"),
        f"/Resource/{hash_value}/{filename}",
    ])
    for candidate in candidates:
        candidate = os.path.abspath(os.path.normpath(candidate))
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"{label} file not found in public assets. Expected one of: "
        + ", ".join(candidates)
    )


# =========================
# 4. Predictor
# =========================

class AirfoilPredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = None

    def load(self, model_path):
        model_path = os.path.expanduser(model_path)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        state = ckpt.get("state_dict") or ckpt.get("model") or ckpt
        hidden = ckpt.get("hidden", 50) if isinstance(ckpt, dict) else 50
        layers = ckpt.get("layers", 6) if isinstance(ckpt, dict) else 6
        self.net = FCNet(num_ins=3, num_outs=3, num_layers=layers, hidden_size=hidden).to(self.device)
        self.net.load_state_dict(state); self.net.eval()
        return ckpt.get("history", []) if isinstance(ckpt, dict) else []

    def predict(self, x, y, z):
        with torch.no_grad():
            xx = torch.tensor(x.reshape(-1, 1), dtype=torch.float32, device=self.device)
            yy = torch.tensor(y.reshape(-1, 1), dtype=torch.float32, device=self.device)
            zz = torch.tensor(z.reshape(-1, 1), dtype=torch.float32, device=self.device)
            X = torch.cat([xx, yy, zz], dim=1)
            out = self.net(X)
        return (out[:, 0:1].cpu().numpy(), out[:, 1:2].cpu().numpy(), out[:, 2:3].cpu().numpy())


# =========================
# 5. Visualization
# =========================

def show_eval_result(x_s, y_s, z_s, u_s, v_s, w_s, u_p, v_p, w_p, history):
    if history:
        plt.figure(figsize=(7,4)); plt.plot(history); plt.yscale("log")
        plt.xlabel("Epoch"); plt.ylabel("Total Loss"); plt.title("Training Loss")
        plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

    eu = np.linalg.norm(u_s.reshape(-1) - u_p.reshape(-1)) / (np.linalg.norm(u_s.reshape(-1)) + 1e-12)
    ev = np.linalg.norm(v_s.reshape(-1) - v_p.reshape(-1)) / (np.linalg.norm(v_s.reshape(-1)) + 1e-12)
    ew = np.linalg.norm(w_s.reshape(-1) - w_p.reshape(-1)) / (np.linalg.norm(w_s.reshape(-1)) + 1e-12)

    fig = plt.figure(figsize=(18, 8))
    for idx, (ref, pred, title) in enumerate([
        (u_s, u_p, "U (Ref)"), (v_s, v_p, "V (Ref)"), (w_s, w_p, "W (Ref)"),
        (u_p, u_p, "U (PINN)"), (v_p, v_p, "V (PINN)"), (w_p, w_p, "W (PINN)"),
    ]):
        ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
        sc = ax.scatter(y_s.reshape(-1), x_s.reshape(-1), z_s.reshape(-1),
                         c=ref.reshape(-1) if idx < 3 else pred.reshape(-1), cmap='jet', s=5)
        ax.set_xlabel("y"); ax.set_ylabel("x"); ax.set_zlabel("z"); ax.set_title(title)
        ax.set_box_aspect([15, 5, 5])
    plt.suptitle(f"Airfoil Displacement — L2: u={eu:.3e} v={ev:.3e} w={ew:.3e}", fontsize=14)
    plt.tight_layout(); plt.show()

    fig2, ax = plt.subplots(figsize=(6,4))
    ax.bar(["u", "v", "w"], [eu, ev, ew]); ax.set_title("Relative L2 Error")
    ax.set_ylabel("Relative L2")
    for i, v in enumerate([eu, ev, ew]): ax.text(i, v, f"{v:.3e}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout(); plt.show()
    return eu, ev, ew


# =========================
# 6. Eval entry
# =========================

def evaluate(model_path=MODEL_PATH, data_path=DATA_PATH):
    print(f"device={'cuda' if torch.cuda.is_available() else 'cpu'}")

    data_path = resolve_public_file(data_path, DATA_HASH, DATA_FILENAME, "Data")
    model_path = resolve_public_file(model_path, MODEL_HASH, MODEL_FILENAME, "Model")
    x_s, y_s, z_s, u_s, v_s, w_s = load_eval_data(data_path)

    pred = AirfoilPredictor()
    history = pred.load(model_path)
    u_p, v_p, w_p = pred.predict(x_s, y_s, z_s)

    print("Eval result"); print(f"model_path: {model_path}")
    eu, ev, ew = show_eval_result(x_s, y_s, z_s, u_s, v_s, w_s, u_p, v_p, w_p, history)
    print(f"Relative L2: u={eu:.4e} v={ev:.4e} w={ew:.4e}")
    return {"errors": {"u": eu, "v": ev, "w": ew}}


eval_result = evaluate()
