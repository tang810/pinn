"""
Notebook-friendly eval file for 2D_scattering_inverse.

The data folder is flat: x_src.pt, xy_col.pt, obs_data_small.pt.
The model is the .pt produced by train_single_file.py.
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


DATA_HASH = "a044220c2a874a239832fb3caf7acdd0"
MODEL_HASH = "a044220c2a874a239832fb3caf7acdd0"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/trained_model_2d_scattering_inverse.pt")

OBS_FILE_NAME = "obs_data_small.pt"
WIDTH = 32
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


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_torch(path, device="cpu"):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def local_project_dir():
    try:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        return os.path.abspath(os.path.join(os.getcwd(), "Electromagnetism", "2D_scattering_inverse"))


def candidate_data_roots(data_path):
    roots = []
    if data_path:
        roots.append(os.path.expanduser(data_path))
    roots += [
        os.path.expanduser(f"~/public/Resource/{DATA_HASH}"),
        os.path.expanduser(f"~/Resource/{DATA_HASH}"),
        f"/Resource/{DATA_HASH}",
        os.path.join(local_project_dir(), "data_small"),
        os.path.join(os.getcwd(), "Electromagnetism", "2D_scattering_inverse", "data_small"),
    ]
    seen = []
    for root in roots:
        if root and root not in seen:
            seen.append(root)
    return seen


def has_dataset(root):
    return (
        os.path.isdir(root)
        and os.path.isfile(os.path.join(root, "x_src.pt"))
        and os.path.isfile(os.path.join(root, "xy_col.pt"))
        and os.path.isfile(os.path.join(root, OBS_FILE_NAME))
    )


def resolve_data_root(data_path):
    for root in candidate_data_roots(data_path):
        if has_dataset(root):
            return root
    raise FileNotFoundError(
        "Cannot find flat dataset folder with x_src.pt, xy_col.pt, obs_data_small.pt. "
        f"Tried: {candidate_data_roots(data_path)}"
    )


def tensor_from_obj(obj, key, device):
    if isinstance(obj, dict):
        obj = obj[key]
    return torch.as_tensor(obj, dtype=torch.float32, device=device)


def load_dataset(data_path, device):
    root = resolve_data_root(data_path)
    x_src = tensor_from_obj(load_torch(os.path.join(root, "x_src.pt"), "cpu"), "x_src", device)
    xy_col = tensor_from_obj(load_torch(os.path.join(root, "xy_col.pt"), "cpu"), "xy_col", device)
    obs = load_torch(os.path.join(root, OBS_FILE_NAME), "cpu")
    xy_obs_list = [torch.as_tensor(v, dtype=torch.float32, device=device) for v in obs["xy_obs_list"]]
    re_list = [torch.as_tensor(v, dtype=torch.float32, device=device) for v in obs["Eobs_re_list"]]
    im_list = [torch.as_tensor(v, dtype=torch.float32, device=device) for v in obs["Eobs_im_list"]]
    freqs = [float(v) for v in obs.get("frequencies", [3.0, 4.0])]
    angles = [float(v) for v in obs.get("angles", [0.0, np.pi / 3.0, 2.0 * np.pi / 3.0])]

    xs, ys = [], []
    idx = 0
    for freq in freqs:
        for angle in angles:
            xy = xy_obs_list[idx]
            meta = torch.zeros((xy.shape[0], 2), dtype=torch.float32, device=device)
            meta[:, 0] = freq
            meta[:, 1] = angle
            xs.append(torch.cat([xy, meta], dim=1))
            ys.append(torch.cat([re_list[idx], im_list[idx]], dim=1))
            idx += 1
    return {"x": torch.cat(xs, dim=0), "y": torch.cat(ys, dim=0), "x_src": x_src, "xy_col": xy_col, "data_path": root}


class FieldMLP(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 2),
        )

    def forward(self, x):
        return self.net(x)


def model_candidates(model_path):
    roots = []
    if model_path:
        roots.append(os.path.expanduser(model_path))
        roots.append(os.path.join(os.path.expanduser(model_path), "trained_model_2d_scattering_inverse.pt"))
    roots += [
        os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/trained_model_2d_scattering_inverse.pt"),
        os.path.expanduser(f"~/minio/Resource/{MODEL_HASH}/trained_model_2d_scattering_inverse.pt"),
        os.path.expanduser(f"~/Resource/{MODEL_HASH}/trained_model_2d_scattering_inverse.pt"),
        f"/Resource/{MODEL_HASH}/trained_model_2d_scattering_inverse.pt",
        os.path.join(local_project_dir(), "trained_model_2d_scattering_inverse.pt"),
        os.path.join(local_project_dir(), "model", "trained_model_2d_scattering_inverse.pt"),
    ]
    return roots


def load_model(model_path, device):
    for path in model_candidates(model_path):
        if path and os.path.isfile(path):
            ckpt = load_torch(path, device)
            width = int(ckpt.get("width", WIDTH))
            model = FieldMLP(width).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            return model, ckpt, path
    raise FileNotFoundError("Model file not found. Tried: " + ", ".join(model_candidates(model_path)))


def show_eval_result(y_true, y_pred, history):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    if history and "loss" in history:
        axes[0].semilogy(history["loss"])
    axes[0].set_title("Training loss")
    axes[0].grid(True, alpha=0.3)
    axes[1].scatter(y_true[:, 0], y_pred[:, 0], s=12, alpha=0.7, label="real")
    axes[1].scatter(y_true[:, 1], y_pred[:, 1], s=12, alpha=0.7, label="imag")
    axes[1].set_title("Field parity")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    err = np.linalg.norm(y_pred - y_true, axis=1)
    axes[2].hist(err, bins=30)
    axes[2].set_title("Point error")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
    plt.close(fig)


def evaluate(model_path=MODEL_PATH, data_path=DATA_PATH):
    set_seed(SEED)
    device = get_device()
    print("Using device:", device)
    data = load_dataset(data_path, device)
    model, ckpt, resolved_model = load_model(model_path, device)

    with torch.no_grad():
        pred = model(data["x"]).detach().cpu().numpy()
    y_true = data["y"].detach().cpu().numpy()
    rel_l2 = float(np.linalg.norm(pred - y_true) / (np.linalg.norm(y_true) + 1e-12))
    mae = float(np.mean(np.abs(pred - y_true)))

    print("Eval result")
    print("model_path:", resolved_model)
    print("data_path:", data["data_path"])
    print(f"relative_l2: {rel_l2:.6e}")
    print(f"mae: {mae:.6e}")
    show_eval_result(y_true, pred, ckpt.get("history", {}))
    return {"relative_l2": rel_l2, "mae": mae, "model": model}


eval_result = evaluate()
