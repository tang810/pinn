"""
Notebook-friendly train file for MEMS cantilever wt/lt inverse problem.

Upload wt_lt_small_compact.npz as one public dataset asset, fill DATA_HASH and
MODEL_HASH, then run this whole file/cell. The model is saved to:
~/minio/Resource/{MODEL_HASH}/trained_model_mems_wtlt.pt
"""

import os
import sys
import time
import types
import random

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = "e335bf92f88043abb610e81f6ac23b5b"
MODEL_HASH = "e335bf92f88043abb610e81f6ac23b5b"
DATA_FILE_PATH = "e335bf92f88043abb610e81f6ac23b5b"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/minio/Resource/{MODEL_HASH}/trained_model_mems_wtlt.pt")

COMPACT_FILE_NAME = "wt_lt_small_compact.npz"
H5_FILE_NAME = "wt_lt_big.h5"
TRAIN_EPOCHS = 300
PRINT_EVERY = 50
SEED = 42

BATCH_SIZE = 128
LR = 1e-3
HIDDEN_DIM = 64
HIDDEN_LAYERS = 2
Y_SCALE_FACTOR = 1e8


# =========================
# 2. Path and data helpers
# =========================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    # Some HDF5 object columns were pickled with newer numpy paths.
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


def minmax_fit(values):
    values = np.asarray(values, dtype=np.float32)
    return {"min": values.min(axis=0), "max": values.max(axis=0)}


def minmax_transform(values, scaler):
    values = np.asarray(values, dtype=np.float32)
    return (values - scaler["min"]) / (scaler["max"] - scaler["min"] + 1e-12)


def minmax_inverse(values, scaler):
    values = np.asarray(values, dtype=np.float32)
    return values * (scaler["max"] - scaler["min"] + 1e-12) + scaler["min"]


def standard_fit(values):
    values = np.asarray(values, dtype=np.float32)
    return {"mean": values.mean(axis=0), "std": values.std(axis=0) + 1e-8}


def standard_transform(values, scaler):
    values = np.asarray(values, dtype=np.float32)
    return (values - scaler["mean"]) / scaler["std"]


def load_compact_npz(npz_path):
    data = np.load(npz_path)
    X_raw = np.asarray(data["X_raw"], dtype=np.float32)
    Y_raw = np.asarray(data["Y_raw"], dtype=np.float32)
    x_scaler = standard_fit(X_raw)
    y_scaler = minmax_fit(Y_raw)
    X = standard_transform(X_raw, x_scaler)
    Y = minmax_transform(Y_raw, y_scaler)
    return {
        "source_path": npz_path,
        "X_raw": X_raw,
        "Y_raw": Y_raw,
        "X": X.astype(np.float32),
        "Y": Y.astype(np.float32),
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
    }


def load_h5_dataset(h5_path):
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

        feat = [
            float(omega.min()), float(omega.max()), float(omega.std()), float(omega.mean()),
            float(y_scaled.min()), float(y_scaled.max()), float(y_scaled.std()), float(y_scaled.mean()),
        ]
        features.append(feat)
        targets.append([float(row["w_t"]), float(row["l_t"])])
        freq_list.append(freq)
        response_list.append(mc)

    X_raw = np.asarray(features, dtype=np.float32)
    Y_raw = np.asarray(targets, dtype=np.float32)
    x_scaler = standard_fit(X_raw)
    y_scaler = minmax_fit(Y_raw)
    X = standard_transform(X_raw, x_scaler)
    Y = minmax_transform(Y_raw, y_scaler)

    return {
        "source_path": h5_path,
        "X_raw": X_raw,
        "Y_raw": Y_raw,
        "X": X.astype(np.float32),
        "Y": Y.astype(np.float32),
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "freq_list": freq_list,
        "response_list": response_list,
    }


def load_dataset(dataset_path):
    suffix = os.path.splitext(dataset_path)[1].lower()
    if suffix == ".npz":
        return load_compact_npz(dataset_path)
    if suffix in (".h5", ".hdf5"):
        return load_h5_dataset(dataset_path)
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


def train_test_split_indices(n, seed=42, test_ratio=0.2):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = max(1, int(n * test_ratio))
    return idx[n_test:], idx[:n_test]


# =========================
# 4. Train and visualize
# =========================

def train(data_path=DATA_PATH, model_path=MODEL_PATH):
    set_seed(SEED)
    device = get_device()
    print("Using device:", device)

    dataset_path = resolve_dataset_path(data_path)
    data = load_dataset(dataset_path)
    train_idx, val_idx = train_test_split_indices(len(data["X"]), SEED)

    X_train = torch.tensor(data["X"][train_idx], dtype=torch.float32, device=device)
    Y_train = torch.tensor(data["Y"][train_idx], dtype=torch.float32, device=device)
    X_val = torch.tensor(data["X"][val_idx], dtype=torch.float32, device=device)
    Y_val = torch.tensor(data["Y"][val_idx], dtype=torch.float32, device=device)

    model = WTLTInverseMLP(hidden_dim=HIDDEN_DIM, hidden_layers=HIDDEN_LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    mse = nn.MSELoss()
    history = {"train": [], "val": []}
    t0 = time.time()

    print("Dataset:", data["source_path"])
    print("samples:", len(data["X"]), "train:", len(train_idx), "val:", len(val_idx))
    print("\nStart training")

    for epoch in range(1, TRAIN_EPOCHS + 1):
        model.train()
        perm = torch.randperm(X_train.shape[0], device=device)
        batch_losses = []
        for start in range(0, X_train.shape[0], BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            pred = model(X_train[idx])
            loss = mse(pred, Y_train[idx])
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_loss = mse(model(X_val), Y_val).item()
        history["train"].append(float(np.mean(batch_losses)))
        history["val"].append(float(val_loss))

        if epoch == 1 or epoch % PRINT_EVERY == 0 or epoch == TRAIN_EPOCHS:
            print(f"Epoch {epoch:5d}/{TRAIN_EPOCHS} | train={history['train'][-1]:.6e} | val={val_loss:.6e}")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "hidden_dim": HIDDEN_DIM,
        "hidden_layers": HIDDEN_LAYERS,
        "x_scaler": data["x_scaler"],
        "y_scaler": data["y_scaler"],
        "history": history,
        "dataset_file": os.path.basename(data["source_path"]),
    }
    model_path = os.path.abspath(os.path.expanduser(model_path))
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(checkpoint, model_path)

    print("\nTraining finished")
    print(f"elapsed: {time.time() - t0:.1f}s")
    print("final val loss:", history["val"][-1])
    print("model saved to:", model_path)

    show_result(model, data, val_idx, history, device)
    return checkpoint, model


def predict_physical(model, X_norm, y_scaler, device):
    model.eval()
    with torch.no_grad():
        pred_norm = model(torch.tensor(X_norm, dtype=torch.float32, device=device)).cpu().numpy()
    return minmax_inverse(pred_norm, y_scaler)


def show_result(model, data, val_idx, history, device):
    pred = predict_physical(model, data["X"][val_idx], data["y_scaler"], device)
    true = data["Y_raw"][val_idx]
    wt_err = (pred[:, 0] - true[:, 0]) * 1e6
    lt_err = (pred[:, 1] - true[:, 1]) * 1e6

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].semilogy(history["train"], label="train")
    axes[0].semilogy(history["val"], label="val")
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(true[:, 0] * 1e6, pred[:, 0] * 1e6, s=8, alpha=0.5, label="w_t")
    axes[1].scatter(true[:, 1] * 1e6, pred[:, 1] * 1e6, s=8, alpha=0.5, label="l_t")
    lo = min((true * 1e6).min(), (pred * 1e6).min())
    hi = max((true * 1e6).max(), (pred * 1e6).max())
    axes[1].plot([lo, hi], [lo, hi], "k--", linewidth=1)
    axes[1].set_title("True vs predicted geometry")
    axes[1].set_xlabel("true um")
    axes[1].set_ylabel("pred um")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].hist(wt_err, bins=40, alpha=0.65, label="w_t error")
    axes[2].hist(lt_err, bins=40, alpha=0.65, label="l_t error")
    axes[2].set_title("Prediction error")
    axes[2].set_xlabel("error um")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


checkpoint, model = train()
