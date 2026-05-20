"""
Notebook-friendly eval file for MEMS cantilever wt/lt inverse problem.

Upload wt_lt_small_compact.npz and trained_model_mems_wtlt.pt as public assets.
Fill DATA_HASH and MODEL_HASH, then run this whole file/cell.
"""

import os
import sys
import types

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = "e335bf92f88043abb610e81f6ac23b5b"
MODEL_HASH = "a64b5579ccfc4037be214e626f27b310"
DATA_FILE_PATH = "e335bf92f88043abb610e81f6ac23b5b"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/trained_model_mems_wtlt.pt")
COMPACT_FILE_NAME = "wt_lt_small_compact.npz"
H5_FILE_NAME = "wt_lt_big.h5"
HIDDEN_DIM = 64
HIDDEN_LAYERS = 2
Y_SCALE_FACTOR = 1e8


# =========================
# 2. Path and data helpers
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


def resource_roots():
    return [
        os.path.expanduser("~/Resource"),
        "/mnt/minio/userdk8e2v7l/Resource",
        "/mnt/minio/userdk8e2v7l/public/Resource",
        os.path.expanduser("~/public/Resource"),
        "/mnt/minio/public/Resource",
        os.path.expanduser("~/public/Resource/userdk8e2v7l"),
        "/mnt/minio/public/Resource/userdk8e2v7l",
        os.path.expanduser("~/minio/Resource"),
        "/mnt/minio/Resource",
        os.path.expanduser("~/minio/userdk8e2v7l/Resource"),
        os.path.expanduser("~/minio/Resource/userdk8e2v7l"),
        "/mnt/minio/Resource/userdk8e2v7l",
    ]


def is_filled(value):
    return bool(value) and not value.startswith("FILL_")


def uniq(items):
    out = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


def candidate_data_paths(data_path):
    candidates = []
    if DATA_FILE_PATH:
        exact_path = os.path.expanduser(DATA_FILE_PATH)
        candidates.append(exact_path)
        if exact_path.startswith("jupyterfile-"):
            candidates.append("/mnt/minio/" + exact_path.replace("jupyterfile-", "", 1))
    if data_path:
        expanded = os.path.expanduser(data_path)
        candidates.append(expanded)
        if expanded.startswith("jupyterfile-"):
            candidates.append("/mnt/minio/" + expanded.replace("jupyterfile-", "", 1))
    if is_filled(DATA_HASH):
        for root in resource_roots():
            candidates.append(os.path.join(root, DATA_HASH))
            candidates.append(os.path.join(root, DATA_HASH, COMPACT_FILE_NAME))
            candidates.append(os.path.join(root, DATA_HASH, H5_FILE_NAME))
    return uniq(candidates)


def resolve_dataset_path(data_path):
    tried = []
    for path in candidate_data_paths(data_path):
        tried.append(path)
        if os.path.isfile(path):
            return path
        if os.path.isdir(path):
            compact_path = os.path.join(path, COMPACT_FILE_NAME)
            if os.path.isfile(compact_path):
                return compact_path
            file_path = os.path.join(path, H5_FILE_NAME)
            if os.path.isfile(file_path):
                return file_path
            npz_files = [
                os.path.join(path, name)
                for name in os.listdir(path)
                if name.lower().endswith(".npz")
            ]
            if npz_files:
                return npz_files[0]
            h5_files = [
                os.path.join(path, name)
                for name in os.listdir(path)
                if name.lower().endswith((".h5", ".hdf5"))
            ]
            if h5_files:
                return h5_files[0]
    raise FileNotFoundError(
        f"Cannot find a MEMS dataset file. Upload {COMPACT_FILE_NAME} as a public dataset asset and set DATA_HASH.\n"
        f"If the platform shows a concrete asset path, paste it into DATA_FILE_PATH.\n"
        f"Tried paths: {tried}"
    )


def patch_numpy_pickle_compat():
    try:
        import numpy.core.numeric as numeric
        import numpy.core.multiarray as multiarray
        core_mod = types.ModuleType("numpy._core")
        core_mod.__path__ = []
        core_mod.numeric = numeric
        core_mod.multiarray = multiarray
        sys.modules.setdefault("numpy._core", core_mod)
        sys.modules.setdefault("numpy._core.numeric", numeric)
        sys.modules.setdefault("numpy._core.multiarray", multiarray)
    except Exception:
        pass


def standard_transform(values, scaler):
    return (np.asarray(values, dtype=np.float32) - scaler["mean"]) / scaler["std"]


def minmax_inverse(values, scaler):
    values = np.asarray(values, dtype=np.float32)
    return values * (scaler["max"] - scaler["min"] + 1e-12) + scaler["min"]


def load_h5_raw(h5_path):
    patch_numpy_pickle_compat()
    try:
        import pandas as pd
    except Exception as exc:
        raise ImportError("This file needs pandas and pytables to read wt_lt_big.h5.") from exc

    with pd.HDFStore(h5_path, "r") as store:
        df = store["data_valid"]
        constants = store.get_storer("data").attrs.constants

    trans_factor = (
        constants["eps_0"] * constants["V"] * constants["electrode_length"] * constants["t"]
        / constants["d"] ** 2
    )

    features = []
    targets = []
    freq_list = []
    response_list = []
    for _, row in df.iterrows():
        freq = np.asarray(row["freq"], dtype=np.float32)
        mc = np.asarray(row["m_c"], dtype=np.float32)
        omega = (2.0 * np.pi * freq).astype(np.float32)
        y = (mc * 1e-9 / (omega * trans_factor + 1e-30)).astype(np.float32)
        y_scaled = y * Y_SCALE_FACTOR
        features.append([
            float(omega.min()), float(omega.max()), float(omega.std()), float(omega.mean()),
            float(y_scaled.min()), float(y_scaled.max()), float(y_scaled.std()), float(y_scaled.mean()),
        ])
        targets.append([float(row["w_t"]), float(row["l_t"])])
        freq_list.append(freq)
        response_list.append(mc)
    return {
        "source": h5_path,
        "X_raw": np.asarray(features, dtype=np.float32),
        "Y_raw": np.asarray(targets, dtype=np.float32),
        "freq_list": freq_list,
        "response_list": response_list,
    }


def load_compact_npz(npz_path):
    data = np.load(npz_path)
    return {
        "source": npz_path,
        "X_raw": np.asarray(data["X_raw"], dtype=np.float32),
        "Y_raw": np.asarray(data["Y_raw"], dtype=np.float32),
        "freq_list": [],
        "response_list": [],
    }


def load_dataset_raw(dataset_path):
    suffix = os.path.splitext(dataset_path)[1].lower()
    if suffix == ".npz":
        return load_compact_npz(dataset_path)
    if suffix in (".h5", ".hdf5"):
        return load_h5_raw(dataset_path)
    raise ValueError(f"Unsupported dataset file: {dataset_path}")


# =========================
# 3. Model
# =========================

class WTLTInverseMLP(nn.Module):
    def __init__(self, input_dim=8, output_dim=2, hidden_dim=128, hidden_layers=4):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, output_dim), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# Stub classes for unpickling older full-model saves
class PINNInverseMLP(nn.Module):
    def __init__(self, input_dim=8, output_dim=2, hidden_dim=128, hidden_layers=4):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, output_dim), nn.Sigmoid()]
        self.model = nn.Sequential(*layers)
    def forward(self, x): return self.model(x)

class PINNDataset: pass

def infer_mlp_shape(state_dict):
    prefix = "net."
    weight_keys = sorted(
        [key for key in state_dict if key.startswith("net.") and key.endswith(".weight")],
        key=lambda key: int(key.split(".")[1]),
    )
    if not weight_keys:
        prefix = "model."
        weight_keys = sorted(
            [key for key in state_dict if key.startswith(prefix) and key.endswith(".weight")],
            key=lambda key: int(key.split(".")[1]),
        )
    if not weight_keys:
        return HIDDEN_DIM, HIDDEN_LAYERS
    hidden_dim = int(state_dict[weight_keys[0]].shape[0])
    hidden_layers = max(1, len(weight_keys) - 1)
    return hidden_dim, hidden_layers


def load_checkpoint(model_path, device):
    model_path = os.path.expanduser(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"MODEL_PATH does not exist: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if hasattr(checkpoint, "state_dict") and not isinstance(checkpoint, dict):
        state = checkpoint.state_dict()
        hidden_dim, hidden_layers = infer_mlp_shape(state)
        model = WTLTInverseMLP(hidden_dim=hidden_dim, hidden_layers=hidden_layers).to(device)
        model.load_state_dict(state)
        model.eval()
        return {}, model

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
        hidden_dim, hidden_layers = infer_mlp_shape(state)
        model = WTLTInverseMLP(
            hidden_dim=checkpoint.get("hidden_dim", hidden_dim),
            hidden_layers=checkpoint.get("hidden_layers", hidden_layers),
        ).to(device)
        model.load_state_dict(state)
        model.eval()
        return checkpoint, model

    hidden_dim, hidden_layers = infer_mlp_shape(checkpoint)
    model = WTLTInverseMLP(hidden_dim=hidden_dim, hidden_layers=hidden_layers).to(device)
    model.load_state_dict(checkpoint)
    model.eval()
    return {}, model


def predict(model, X_raw, x_scaler, y_scaler, device, chunk=8192):
    X = standard_transform(X_raw, x_scaler)
    preds = []
    with torch.no_grad():
        for start in range(0, len(X), chunk):
            xb = torch.tensor(X[start:start + chunk], dtype=torch.float32, device=device)
            preds.append(model(xb).cpu().numpy())
    pred_norm = np.vstack(preds)
    return minmax_inverse(pred_norm, y_scaler)


# =========================
# 4. Eval and visualize
# =========================

def evaluate(model_path=MODEL_PATH, data_path=DATA_PATH):
    device = get_device()
    print("Using device:", device)

    dataset_path = resolve_dataset_path(data_path)
    checkpoint, model = load_checkpoint(model_path, device)
    data = load_dataset_raw(dataset_path)
    pred = predict(model, data["X_raw"], checkpoint["x_scaler"], checkpoint["y_scaler"], device)
    true = data["Y_raw"]

    err_um = (pred - true) * 1e6
    rmse_um = np.sqrt(np.mean(err_um ** 2, axis=0))
    mae_um = np.mean(np.abs(err_um), axis=0)
    rel_l2 = np.linalg.norm(pred - true, axis=0) / (np.linalg.norm(true, axis=0) + 1e-12)

    print("\nModel loaded:", os.path.expanduser(model_path))
    print("Dataset:", data["source"])
    print("samples:", len(true))
    print("\nEval result")
    print(f"  wt_rmse_um: {rmse_um[0]:.6e}")
    print(f"  lt_rmse_um: {rmse_um[1]:.6e}")
    print(f"  wt_mae_um: {mae_um[0]:.6e}")
    print(f"  lt_mae_um: {mae_um[1]:.6e}")
    print(f"  wt_relative_l2: {rel_l2[0]:.6e}")
    print(f"  lt_relative_l2: {rel_l2[1]:.6e}")

    show_result(true, pred, err_um, checkpoint.get("history", {}), data)
    return {"true": true, "pred": pred, "err_um": err_um, "checkpoint": checkpoint}


def show_result(true, pred, err_um, history, data):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    if history and "train" in history:
        axes[0, 0].semilogy(history["train"], label="train")
        axes[0, 0].semilogy(history.get("val", []), label="val")
        axes[0, 0].legend()
    axes[0, 0].set_title("Training loss")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].scatter(true[:, 0] * 1e6, pred[:, 0] * 1e6, s=6, alpha=0.4, label="w_t")
    axes[0, 1].scatter(true[:, 1] * 1e6, pred[:, 1] * 1e6, s=6, alpha=0.4, label="l_t")
    lo = min((true * 1e6).min(), (pred * 1e6).min())
    hi = max((true * 1e6).max(), (pred * 1e6).max())
    axes[0, 1].plot([lo, hi], [lo, hi], "k--", linewidth=1)
    axes[0, 1].set_title("True vs predicted")
    axes[0, 1].set_xlabel("true um")
    axes[0, 1].set_ylabel("pred um")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].hist(err_um[:, 0], bins=60, alpha=0.65, label="w_t error")
    axes[1, 0].hist(err_um[:, 1], bins=60, alpha=0.65, label="l_t error")
    axes[1, 0].set_title("Error distribution")
    axes[1, 0].set_xlabel("error um")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    if data["freq_list"]:
        sample_idx = 0
        axes[1, 1].plot(data["freq_list"][sample_idx], data["response_list"][sample_idx])
        axes[1, 1].set_title("Example frequency response")
        axes[1, 1].set_xlabel("frequency")
        axes[1, 1].set_ylabel("m_c")
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].axis("off")
        axes[1, 1].text(
            0.5,
            0.5,
            "Compact npz contains features and labels only",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )

    plt.tight_layout()
    plt.show()


eval_result = evaluate()
