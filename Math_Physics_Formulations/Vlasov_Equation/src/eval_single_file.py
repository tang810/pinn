"""
Notebook-friendly eval file for Vlasov_Equation.

Upload E.npy/f.npy and trained_model_vlasov_equation.pt as public assets.
Fill the hash fields below, then run this whole file/cell.
"""

import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# =========================
# 1. Parameters to edit
# =========================

# Case A: E.npy and f.npy are in the same uploaded folder asset.
DATA_HASH = "851695ee0e7c4288a189c37d45703ce7"

# Case B: E.npy and f.npy were uploaded as two separate file assets.
# Leave these empty when using DATA_HASH.
E_HASH = "851695ee0e7c4288a189c37d45703ce7"
F_HASH = "851695ee0e7c4288a189c37d45703ce7"

MODEL_HASH = "fa7582bc8c2c462f881b49f2354193e3"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/trained_model_vlasov_equation.pt")

T_MAX = 62.5
X_MAX = 10.0
V_MIN = -5.0
V_MAX = 5.0
REQUIRED_FILES = ["E.npy", "f.npy"]


# =========================
# 2. Paths and data
# =========================

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resource_roots():
    return [
        os.path.expanduser("~/public/Resource"),
        "/mnt/minio/public/Resource",
        os.path.expanduser("~/public/Resource/userdk8e2v7l"),
        "/mnt/minio/public/Resource/userdk8e2v7l",
        os.path.expanduser("~/minio/Resource"),
        "/mnt/minio/Resource",
        os.path.expanduser("~/minio/Resource/userdk8e2v7l"),
        "/mnt/minio/Resource/userdk8e2v7l",
    ]


def uniq(items):
    out = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


def is_filled(value):
    return bool(value) and not value.startswith("FILL_")


def candidate_folder_paths(data_path):
    candidates = [os.path.expanduser(data_path)] if data_path else []
    if is_filled(DATA_HASH):
        for root in resource_roots():
            candidates.append(os.path.join(root, DATA_HASH))
    return uniq(candidates)


def candidate_file_paths(file_hash, filename):
    if not is_filled(file_hash):
        return []
    candidates = [os.path.expanduser(file_hash)]
    for root in resource_roots():
        candidates.append(os.path.join(root, file_hash))
        candidates.append(os.path.join(root, file_hash, filename))
    return uniq(candidates)


def resolve_data_files(data_path):
    tried_folders = []
    missing_by_folder = {}
    for folder in candidate_folder_paths(data_path):
        tried_folders.append(folder)
        if not os.path.isdir(folder):
            continue
        missing = [name for name in REQUIRED_FILES if not os.path.isfile(os.path.join(folder, name))]
        if missing:
            missing_by_folder[folder] = missing
            continue
        return os.path.join(folder, "E.npy"), os.path.join(folder, "f.npy"), folder

    e_candidates = candidate_file_paths(E_HASH, "E.npy")
    f_candidates = candidate_file_paths(F_HASH, "f.npy")
    e_file = next((p for p in e_candidates if os.path.isfile(p)), None)
    f_file = next((p for p in f_candidates if os.path.isfile(p)), None)
    if e_file and f_file:
        return e_file, f_file, f"E={e_file}; f={f_file}"

    print("No data files found, generating simulated Vlasov data as fallback.")
    return None, None, "simul"


def load_vlasov_data(data_path):
    e_path, f_path, source = resolve_data_files(data_path)
    if e_path is None:
        nt, nx, nv = 21, 64, 64
        t = np.linspace(0.0, T_MAX, nt, dtype=np.float32)
        x = np.linspace(0.0, X_MAX, nx, dtype=np.float32)
        v = np.linspace(V_MIN, V_MAX, nv, dtype=np.float32)
        T, X, V = np.meshgrid(t, x, v, indexing="ij")
        E = (np.sin(2*np.pi*X/X_MAX)*np.cos(T)).astype(np.float32)
        f = (np.exp(-V**2/2.0)*(1.0+0.1*np.sin(2*np.pi*X/X_MAX)*np.cos(T))).astype(np.float32)
        return {"source": source, "E": E, "f": f, "t": t, "x": x, "v": v}
    E = np.load(e_path).astype(np.float32)
    f = np.load(f_path).astype(np.float32)
    if E.shape != f.shape or E.ndim != 3:
        raise ValueError(f"E.npy and f.npy must have the same 3D shape. Got E={E.shape}, f={f.shape}")
    nt, nx, nv = f.shape
    t = np.linspace(0.0, T_MAX, nt, dtype=np.float32)
    x = np.linspace(0.0, X_MAX, nx, dtype=np.float32)
    v = np.linspace(V_MIN, V_MAX, nv, dtype=np.float32)
    return {"source": source, "E": E, "f": f, "t": t, "x": x, "v": v}


def coord_scale_tensors(device):
    offset = torch.tensor([[T_MAX / 2.0, X_MAX / 2.0, 0.0]], dtype=torch.float32, device=device)
    scale = torch.tensor([[T_MAX / 2.0, X_MAX / 2.0, (V_MAX - V_MIN) / 2.0]], dtype=torch.float32, device=device)
    deriv_scale = torch.tensor([[2.0 / T_MAX, 2.0 / X_MAX, 2.0 / (V_MAX - V_MIN)]], dtype=torch.float32, device=device)
    return offset, scale, deriv_scale


# =========================
# 3. Model and metrics
# =========================

class DNN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        blocks = []
        for i in range(len(layers) - 2):
            blocks.append((f"layer_{i}", nn.Linear(layers[i], layers[i + 1])))
            blocks.append((f"activation_{i}", nn.Tanh()))
        blocks.append((f"layer_{len(layers) - 2}", nn.Linear(layers[-2], layers[-1])))
        self.layers = nn.Sequential(OrderedDict(blocks))
        self.lambda_1 = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.lambda_2 = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        for module in self.layers:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x_norm):
        return self.layers(x_norm)


def model_outputs(model, coords, offset, scale):
    out = model((coords - offset) / scale)
    return out[:, 0:1], out[:, 1:2]


def vlasov_residual(model, coords, offset, scale, deriv_scale):
    coords = coords.clone().detach().requires_grad_(True)
    coords_norm = (coords - offset) / scale
    out = model(coords_norm)
    f_pred = out[:, 0:1]
    E_pred = out[:, 1:2]
    grad_norm = torch.autograd.grad(
        f_pred,
        coords_norm,
        grad_outputs=torch.ones_like(f_pred),
        retain_graph=True,
        create_graph=True,
    )[0]
    f_t = grad_norm[:, 0:1] * deriv_scale[:, 0:1]
    f_x = grad_norm[:, 1:2] * deriv_scale[:, 1:2]
    f_v = grad_norm[:, 2:3] * deriv_scale[:, 2:3]
    v = coords[:, 2:3]
    return f_t + model.lambda_1 * v * f_x + model.lambda_2 * E_pred * f_v


def load_checkpoint(model_path, device):
    model_path = os.path.expanduser(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"MODEL_PATH does not exist: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict):
        state = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint.get("model")
            or checkpoint
        )
    else:
        state = checkpoint

    if not isinstance(state, dict):
        raise ValueError("Unsupported checkpoint format: expected a state_dict-like object.")

    # Handle old formats: "net.layer_" -> "layers.layer_", "linear_" -> "layer_"
    renamed = {}
    for k, v in state.items():
        new_k = k
        new_k = new_k.replace("net.layer_", "layers.layer_")
        new_k = new_k.replace("net.linear_", "layers.layer_")
        new_k = new_k.replace("linear_", "layer_")
        renamed[new_k] = v
    state = renamed
    layer_keys = [
        k for k, v in state.items()
        if k.endswith(".weight") and getattr(v, "ndim", 0) == 2
    ]
    if layer_keys:
        def layer_order(key):
            for token in reversed(key.split(".")):
                if token.startswith("linear_"):
                    return int(token.split("_", 1)[1])
                if token.isdigit():
                    return int(token) // 2
            return 10**9

        layer_keys = sorted(layer_keys, key=layer_order)
        layers_cfg = [state[layer_keys[0]].shape[1]] + [state[key].shape[0] for key in layer_keys]
    else:
        layers_cfg = checkpoint.get("layers", [3, 64, 64, 64, 64, 2]) if isinstance(checkpoint, dict) else [3, 64, 64, 64, 64, 2]

    model = DNN(layers_cfg).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    critical_missing = [k for k in missing if not k.startswith("lambda_")]
    if critical_missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch. Missing={critical_missing}, unexpected={unexpected}")
    model.eval()
    return checkpoint, model


def predict_chunked(model, coords_np, device, chunk=50000):
    offset, scale, _ = coord_scale_tensors(device)
    preds = []
    with torch.no_grad():
        for start in range(0, coords_np.shape[0], chunk):
            coords = torch.tensor(coords_np[start:start + chunk], dtype=torch.float32, device=device)
            f_pred, E_pred = model_outputs(model, coords, offset, scale)
            preds.append(torch.cat([f_pred, E_pred], dim=1).cpu().numpy())
    return np.vstack(preds)


def eval_sample_metrics(model, data, device, n_eval=20000, seed=42):
    nt, nx, nv = data["f"].shape
    n_eval = min(int(n_eval), nt * nx * nv)
    rng = np.random.default_rng(seed)
    flat_idx = rng.choice(nt * nx * nv, size=n_eval, replace=False)
    it, ix, iv = np.unravel_index(flat_idx, (nt, nx, nv))
    coords = np.stack([data["t"][it], data["x"][ix], data["v"][iv]], axis=1).astype(np.float32)
    targets = np.stack([data["f"][it, ix, iv], data["E"][it, ix, iv]], axis=1).astype(np.float32)
    preds = predict_chunked(model, coords, device)
    err = preds - targets

    offset, scale, deriv_scale = coord_scale_tensors(device)
    sample = torch.tensor(coords[: min(4096, n_eval)], dtype=torch.float32, device=device)
    g = vlasov_residual(model, sample, offset, scale, deriv_scale)
    return {
        "eval_points": int(n_eval),
        "f_rmse": float(np.sqrt(np.mean(err[:, 0] ** 2))),
        "E_rmse": float(np.sqrt(np.mean(err[:, 1] ** 2))),
        "f_relative_l2": float(np.linalg.norm(err[:, 0]) / (np.linalg.norm(targets[:, 0]) + 1e-12)),
        "E_relative_l2": float(np.linalg.norm(err[:, 1]) / (np.linalg.norm(targets[:, 1]) + 1e-12)),
        "pde_residual_mse": float(torch.mean(g ** 2).detach().cpu()),
    }


# =========================
# 4. Eval and show result
# =========================

def evaluate(model_path=MODEL_PATH, data_path=DATA_PATH):
    device = get_device()
    print("Using device:", device)
    data = load_vlasov_data(data_path)
    checkpoint, model = load_checkpoint(model_path, device)
    metrics = eval_sample_metrics(model, data, device)

    print("\nModel loaded:", os.path.expanduser(model_path))
    print("Dataset:", data["source"])
    print("E/f shape:", data["E"].shape)
    print("\nEval result")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6e}" if isinstance(value, float) else f"  {key}: {value}")

    show_result(model, data, checkpoint.get("history", {}), device)
    return {"metrics": metrics, "checkpoint": checkpoint, "model": model}


def show_result(model, data, history, device):
    t_idx = data["f"].shape[0] // 2
    x_idx = data["f"].shape[1] // 2
    v_idx = data["f"].shape[2] // 2
    v_line = data["v"]
    x_line = data["x"]

    coords_v = np.stack([np.full_like(v_line, data["t"][t_idx]), np.full_like(v_line, data["x"][x_idx]), v_line], axis=1)
    coords_x = np.stack([np.full_like(x_line, data["t"][t_idx]), x_line, np.full_like(x_line, data["v"][v_idx])], axis=1)
    pred_v = predict_chunked(model, coords_v.astype(np.float32), device)
    pred_x = predict_chunked(model, coords_x.astype(np.float32), device)

    X_grid, V_grid = np.meshgrid(data["x"], data["v"], indexing="ij")
    coords_map = np.stack(
        [np.full(X_grid.size, data["t"][t_idx], dtype=np.float32), X_grid.reshape(-1), V_grid.reshape(-1)],
        axis=1,
    ).astype(np.float32)
    pred_map = predict_chunked(model, coords_map, device)[:, 0].reshape(len(data["x"]), len(data["v"]))
    true_map = data["f"][t_idx]

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    if history and "total" in history:
        axes[0, 0].semilogy(history["total"], label="total")
        axes[0, 0].semilogy(history.get("f", []), label="f")
        axes[0, 0].semilogy(history.get("E", []), label="E")
        axes[0, 0].semilogy(history.get("pde", []), label="pde")
        axes[0, 0].legend()
    axes[0, 0].set_title("Training loss")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(v_line, data["f"][t_idx, x_idx, :], label="f true")
    axes[0, 1].plot(v_line, pred_v[:, 0], "--", label="f pred")
    axes[0, 1].set_title("Velocity distribution")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(x_line, data["E"][t_idx, :, v_idx], label="E true")
    axes[0, 2].plot(x_line, pred_x[:, 1], "--", label="E pred")
    axes[0, 2].set_title("Electric field slice")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    im0 = axes[1, 0].imshow(true_map.T, origin="lower", aspect="auto", extent=[0, X_MAX, V_MIN, V_MAX])
    axes[1, 0].set_title("True f")
    plt.colorbar(im0, ax=axes[1, 0])

    im1 = axes[1, 1].imshow(pred_map.T, origin="lower", aspect="auto", extent=[0, X_MAX, V_MIN, V_MAX])
    axes[1, 1].set_title("Predicted f")
    plt.colorbar(im1, ax=axes[1, 1])

    im2 = axes[1, 2].imshow(np.abs(pred_map - true_map).T, origin="lower", aspect="auto", extent=[0, X_MAX, V_MIN, V_MAX])
    axes[1, 2].set_title("Absolute error")
    plt.colorbar(im2, ax=axes[1, 2])
    plt.tight_layout()
    plt.show()


eval_result = evaluate()
