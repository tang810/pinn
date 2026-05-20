"""
Notebook-friendly eval file for cylinder_flow (Re=100).

Online Jupyter usage:
1. Upload trained_model_cylinder_flow.pt as a public asset.
2. Upload simul_cylinder_re100.mat as a public asset.
3. Fill MODEL_HASH and DATA_HASH, then run this whole file/cell.

This file loads the PINN model, evaluates cylinder flow (u,v,p) fields,
and shows figures with plt.show(). It does not save images.

Self-contained — only depends on the .mat dataset and .pt model.
"""

import os

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = "6f9ab9dc1b784b8287cd41287a663053"
MODEL_HASH = "97d5029816e74e76adfc4d8af42ae7bd"
DATA_FILENAME = "simul_cylinder_re100.mat"
MODEL_FILENAME = "trained_model_cylinder_flow.pt"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/trained_model_cylinder_flow.pt")


# =========================
# 2. Model definition (FCNet)
# =========================

class FCNet(nn.Module):
    def __init__(self, num_ins=3, num_outs=3, hidden_size=120, num_layers=6):
        super().__init__()
        layers = [nn.Linear(num_ins, hidden_size), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers += [nn.Linear(hidden_size, num_outs)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# =========================
# 3. PINN predictor (eval-only, no training)
# =========================

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    try:
        import torch_npu
        if torch_npu.npu.is_available(): return torch.device("npu")
    except (ImportError, AttributeError): pass
    return torch.device("cpu")


def normalize_state_dict(state):
    """Support checkpoints saved from either FCNet naming style."""
    if not state:
        return state

    keys = list(state.keys())
    if keys and keys[0].startswith("layers.layer_"):
        converted = {}
        for key, value in state.items():
            prefix = "layers.layer_"
            if not key.startswith(prefix):
                converted[key] = value
                continue
            layer_id, suffix = key[len(prefix):].split(".", 1)
            converted[f"net.{int(layer_id) * 2}.{suffix}"] = value
        return converted

    return state


class CylinderPredictor:
    def __init__(self):
        self.device = get_device()
        self.net = None

    def load(self, model_path):
        model_path = os.path.expanduser(model_path)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        state = ckpt.get("state_dict") or ckpt.get("model") or ckpt
        state = normalize_state_dict(state)
        hidden = ckpt.get("hidden", 120) if isinstance(ckpt, dict) else 120
        layers = ckpt.get("layers", 6) if isinstance(ckpt, dict) else 6
        self.net = FCNet(num_ins=3, num_outs=3, hidden_size=hidden, num_layers=layers).to(self.device)
        self.net.load_state_dict(state)
        self.net.eval()
        history = ckpt.get("history", []) if isinstance(ckpt, dict) else []
        return history

    def predict(self, t, x, y):
        with torch.no_grad():
            tt = torch.tensor(t.reshape(-1, 1), dtype=torch.float32, device=self.device)
            xx = torch.tensor(x.reshape(-1, 1), dtype=torch.float32, device=self.device)
            yy = torch.tensor(y.reshape(-1, 1), dtype=torch.float32, device=self.device)
            X = torch.cat([tt, xx, yy], dim=1)
            out = self.net(X)
        u = out[:, 0:1].cpu().numpy(); v = out[:, 1:2].cpu().numpy(); p = out[:, 2:3].cpu().numpy()
        return u, v, p


# =========================
# 4. Data loading
# =========================

def generate_small_mat(filename, n_points=500, n_steps=10):
    xs, ys = [], []
    while len(xs) < n_points:
        bx = np.random.uniform(-7.5, 22.5, n_points * 3)
        by = np.random.uniform(-10.0, 10.0, n_points * 3)
        mask = (bx**2 + by**2) >= 0.5**2
        xs.extend(bx[mask]); ys.extend(by[mask])
    X_star = np.column_stack([xs[:n_points], ys[:n_points]]).astype(np.float32)
    t = np.linspace(0.0, 6.0, n_steps, dtype=np.float32).reshape(-1, 1)
    U_star = np.zeros((n_points, 2, n_steps), dtype=np.float32)
    P_star = np.zeros((n_points, n_steps), dtype=np.float32)
    x_arr = X_star[:, 0:1]; y_arr = X_star[:, 1:2]
    for i in range(n_steps):
        ti = t[i, 0]
        u = 1.0 - 0.42 * np.exp(-((x_arr-0.8)**2+(1.4*y_arr)**2)/3.0) * (1.0+0.18*np.sin(1.8*ti))
        v = 0.15 * np.exp(-((x_arr-1.0)**2+y_arr**2)/4.5) * np.sin(0.35*y_arr+1.6*ti)
        p = 0.35 * np.exp(-((x_arr+0.2)**2+y_arr**2)/0.8) * np.cos(1.2*ti) - 0.015*x_arr
        U_star[:, 0, i] = u.reshape(-1); U_star[:, 1, i] = v.reshape(-1)
        P_star[:, i] = p.reshape(-1)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    scipy.io.savemat(filename, {'X_star': X_star, 'U_star': U_star, 'P_star': P_star, 't': t})
    return {'X_star': X_star, 'U_star': U_star, 'P_star': P_star, 't': t}


def load_mat_data(filename):
    if not os.path.isfile(filename):
        print('Generating small synthetic dataset...')
        data = generate_small_mat(filename)
    else:
        data = scipy.io.loadmat(filename)
    return {
        "U_star": data["U_star"], "P_star": data["P_star"],
        "t_all": data["t"], "X_star": data["X_star"],
    }


def resolve_data_path(data_path, data_hash=DATA_HASH, data_filename=DATA_FILENAME):
    """Resolve public asset paths to the actual .mat file."""
    raw_path = os.path.expanduser(data_path) if data_path else ""
    candidates = []

    if raw_path:
        candidates.append(raw_path)
        if os.path.isdir(raw_path) or not os.path.splitext(raw_path)[1]:
            candidates.append(os.path.join(raw_path, data_filename))

    candidates.extend([
        os.path.expanduser(f"~/public/Resource/{data_hash}/{data_filename}"),
        os.path.expanduser(f"~/Resource/{data_hash}/{data_filename}"),
        f"/Resource/{data_hash}/{data_filename}",
    ])

    for candidate in candidates:
        candidate = os.path.abspath(os.path.normpath(candidate))
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        "Data file not found in public assets. Expected one of: "
        + ", ".join(candidates)
    )


def resolve_model_path(model_path, model_hash=MODEL_HASH, model_filename=MODEL_FILENAME):
    """Resolve public asset paths to a trained model file."""
    raw_path = os.path.expanduser(model_path) if model_path else ""
    candidates = []

    if raw_path:
        candidates.append(raw_path)
        if os.path.isdir(raw_path) or not os.path.splitext(raw_path)[1]:
            candidates.append(os.path.join(raw_path, model_filename))

    candidates.extend([
        os.path.expanduser(f"~/public/Resource/{model_hash}/{model_filename}"),
        os.path.expanduser(f"~/Resource/{model_hash}/{model_filename}"),
        f"/Resource/{model_hash}/{model_filename}",
    ])

    for candidate in candidates:
        candidate = os.path.abspath(os.path.normpath(candidate))
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        "Model file not found in public assets. Expected one of: "
        + ", ".join(candidates)
    )


# =========================
# 5. Visualization
# =========================

def show_eval_result(predictor, data):
    X = data["X_star"]; snap = min(10, data["t_all"].shape[0])
    x_eval = np.tile(X[:, 0:1], (1, snap)).reshape(-1, 1)
    y_eval = np.tile(X[:, 1:2], (1, snap)).reshape(-1, 1)
    t_eval = np.tile(data["t_all"][:snap].reshape(1, -1), (X.shape[0], 1)).reshape(-1, 1)
    u_true = data["U_star"][:, 0, :snap].reshape(-1, 1)
    v_true = data["U_star"][:, 1, :snap].reshape(-1, 1)
    p_true = data["P_star"][:, :snap].reshape(-1, 1)

    u_pred, v_pred, p_pred = predictor.predict(t_eval, x_eval, y_eval)
    speed_t = np.sqrt(u_true.reshape(-1)**2 + v_true.reshape(-1)**2)
    speed_p = np.sqrt(u_pred.reshape(-1)**2 + v_pred.reshape(-1)**2)

    def _sc(ax, val, title):
        sc = ax.scatter(x_eval.reshape(-1), y_eval.reshape(-1), c=val, s=1, cmap="jet")
        ax.set_title(title); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
        plt.colorbar(sc, ax=ax, shrink=0.8)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    _sc(axes[0, 0], speed_t, "Speed True"); _sc(axes[0, 1], speed_p, "Speed Pred")
    _sc(axes[1, 0], p_true.reshape(-1), "Pressure True"); _sc(axes[1, 1], p_pred.reshape(-1), "Pressure Pred")
    fig.suptitle("Cylinder Flow (Re=100) — PINN Prediction", fontsize=14)
    plt.tight_layout(); plt.show()

    eu = np.linalg.norm(u_true - u_pred) / (np.linalg.norm(u_true) + 1e-12)
    ev = np.linalg.norm(v_true - v_pred) / (np.linalg.norm(v_true) + 1e-12)
    ep = np.linalg.norm(p_true - p_pred) / (np.linalg.norm(p_true) + 1e-12)

    fig2, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["u", "v", "p"], [eu, ev, ep])
    ax.set_title("Relative L2 Error"); ax.set_ylabel("Relative L2")
    for i, v in enumerate([eu, ev, ep]): ax.text(i, v, f"{v:.3e}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout(); plt.show()
    return eu, ev, ep


# =========================
# 6. Eval entry
# =========================

def evaluate(model_path=MODEL_PATH, data_path=DATA_PATH):
    print(f"device={'cuda' if torch.cuda.is_available() else 'cpu'}")

    data_path = resolve_data_path(data_path)
    model_path = resolve_model_path(model_path)
    print(f"data={data_path}")
    data = load_mat_data(data_path)

    predictor = CylinderPredictor()
    history = predictor.load(model_path)

    if history:
        plt.figure(figsize=(7, 4)); plt.plot(history); plt.yscale("log")
        plt.xlabel("Epoch"); plt.ylabel("Total Loss"); plt.title("Training Loss")
        plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

    print("Eval result"); print(f"model_path: {model_path}")
    eu, ev, ep = show_eval_result(predictor, data)
    print(f"Relative L2: u={eu:.4e} v={ev:.4e} p={ep:.4e}")
    return {"errors": {"u": eu, "v": ev, "p": ep}}


eval_result = evaluate()
