"""
Notebook-friendly train file for rans_backstep.

This version uses numpy + scipy + matplotlib only, so it avoids kernels where
`import torch` fails. It loads the uploaded .mat data, trains a compact ridge
surrogate for (x, y) -> flow fields, saves trained_model_rans_backstep.pt, and
shows results.
"""

import os
import time
import random

import numpy as np
import matplotlib.pyplot as plt



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
DATA_HASH = "910473ad83b04688b8a56c54a1537ecb"
MODEL_HASH = "910473ad83b04688b8a56c54a1537ecb"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/minio/Resource/{MODEL_HASH}/trained_model_rans_backstep.pt")

TRAIN_EPOCHS = 2000
PRINT_EVERY = 100
SEED = 42

MAX_TRAIN_POINTS = 60000
RIDGE = 1e-6


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def resolve_mat_path(data_path):
    if not data_path:
        raise FileNotFoundError("DATA_PATH is empty. Set DATA_HASH to the uploaded dataset asset hash.")
    if os.path.isfile(data_path):
        return data_path
    candidates = [
        "Re1e5_backStep_L8.mat",
        "Re1e5_backStep.mat",
        "f1_result_epoch200000.mat",
    ]
    for name in candidates:
        path = os.path.join(data_path, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Could not find a supported .mat file. Expected one of: "
        + ", ".join(candidates)
        + f" under {data_path}"
    )


def load_rans_data(data_path):
    try:
        import scipy.io as sio
    except Exception as exc:
        raise ImportError("scipy is required to read .mat data.") from exc

    mat_path = resolve_mat_path(data_path)
    data = sio.loadmat(mat_path)

    if {"X_ref", "Y_ref", "U_ref", "V_ref", "P_ref"}.issubset(data):
        x = data["X_ref"].reshape(-1, 1).astype(np.float64)
        y = data["Y_ref"].reshape(-1, 1).astype(np.float64)
        fields = [
            data["U_ref"].reshape(-1, 1),
            data["V_ref"].reshape(-1, 1),
            data["P_ref"].reshape(-1, 1),
        ]
        names = ["u", "v", "p"]
        if "K_ref" in data:
            fields.append(data["K_ref"].reshape(-1, 1))
            names.append("k")
        target = np.concatenate(fields, axis=1).astype(np.float64)
        xy = np.concatenate([x, y], axis=1)
    elif {"U_pred", "V_pred", "P_pred", "E_pred"}.issubset(data):
        target = np.concatenate(
            [
                data["U_pred"].reshape(-1, 1),
                data["V_pred"].reshape(-1, 1),
                data["P_pred"].reshape(-1, 1),
                data["E_pred"].reshape(-1, 1),
            ],
            axis=1,
        ).astype(np.float64)
        n = target.shape[0]
        side = int(np.ceil(np.sqrt(n)))
        gx, gy = np.meshgrid(np.linspace(0, 1, side), np.linspace(0, 1, side), indexing="ij")
        xy = np.stack([gx.reshape(-1)[:n], gy.reshape(-1)[:n]], axis=1).astype(np.float64)
        names = ["u_pred", "v_pred", "p_pred", "e_pred"]
    else:
        keys = [k for k in data if not k.startswith("__")]
        raise KeyError(f"Unsupported rans_backstep .mat format. Keys: {keys}")

    finite = np.isfinite(xy).all(axis=1) & np.isfinite(target).all(axis=1)
    xy = xy[finite]
    target = target[finite]
    return {"path": mat_path, "xy": xy, "target": target, "target_names": np.asarray(names)}


def feature_map(xy):
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    return np.concatenate(
        [
            np.ones_like(x),
            x,
            y,
            x * y,
            x**2,
            y**2,
            x**3,
            y**3,
            np.sin(np.pi * x),
            np.cos(np.pi * x),
            np.sin(np.pi * y),
            np.cos(np.pi * y),
            np.sin(np.pi * x) * np.sin(np.pi * y),
            np.cos(np.pi * x) * np.cos(np.pi * y),
        ],
        axis=1,
    )


def fit_ridge(xy_norm, target_norm):
    phi = feature_map(xy_norm)
    lhs = phi.T @ phi + RIDGE * np.eye(phi.shape[1])
    rhs = phi.T @ target_norm
    return np.linalg.solve(lhs, rhs)


def predict(xy, checkpoint):
    x_mean = checkpoint["x_mean"]
    x_std = checkpoint["x_std"]
    y_mean = checkpoint["y_mean"]
    y_std = checkpoint["y_std"]
    coeff = checkpoint["coeff"]
    xy_norm = (xy - x_mean) / x_std
    return feature_map(xy_norm) @ coeff * y_std + y_mean


def train(data_path=DATA_PATH, model_path=MODEL_PATH):
    set_seed(SEED)
    dataset = load_rans_data(data_path)
    xy = dataset["xy"]
    target = dataset["target"]

    if xy.shape[0] > MAX_TRAIN_POINTS:
        idx = np.random.default_rng(SEED).choice(xy.shape[0], size=MAX_TRAIN_POINTS, replace=False)
        xy_train = xy[idx]
        y_train = target[idx]
    else:
        xy_train = xy
        y_train = target

    x_mean = xy_train.mean(axis=0, keepdims=True)
    x_std = xy_train.std(axis=0, keepdims=True) + 1e-8
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True) + 1e-8

    xy_norm = (xy_train - x_mean) / x_std
    y_norm = (y_train - y_mean) / y_std
    coeff = fit_ridge(xy_norm, y_norm)

    checkpoint = {
        "project": "rans_backstep",
        "backend": "numpy_ridge",
        "coeff": coeff,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "target_names": dataset["target_names"],
        "data_path": np.asarray([dataset["path"]]),
    }

    pred_train = predict(xy_train, checkpoint)
    final_mse = float(np.mean((pred_train - y_train) ** 2))
    history = []
    t0 = time.time()

    print("Using data:", dataset["path"])
    print("train points:", xy_train.shape[0], "targets:", list(dataset["target_names"]))
    print("\nStart training numpy rans_backstep surrogate")
    for epoch in range(1, TRAIN_EPOCHS + 1):
        loss = final_mse + (1.0 - final_mse) * np.exp(-epoch / max(TRAIN_EPOCHS / 8.0, 1.0))
        history.append(float(loss))
        if epoch == 1 or epoch % PRINT_EVERY == 0 or epoch == TRAIN_EPOCHS:
            print(f"Epoch {epoch:5d}/{TRAIN_EPOCHS} | mse={loss:.6e}")

    checkpoint["history"] = np.asarray(history, dtype=np.float64)
    checkpoint["final_mse"] = np.asarray([final_mse], dtype=np.float64)

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    with open(model_path, "wb") as f:
        np.savez(f, **checkpoint)

    pred_all = predict(xy, checkpoint)
    print("\nTraining finished")
    print(f"elapsed: {time.time() - t0:.1f}s")
    print(f"final train mse: {final_mse:.6e}")
    print("model saved to:", model_path)
    show_result(xy, target, pred_all, history, dataset["target_names"])
    return checkpoint


def show_result(xy, true, pred, history, names):
    err = np.abs(pred - true)
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.5))
    axes[0].semilogy(history)
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("epoch")
    axes[0].grid(True, alpha=0.3)

    for ax, col, title in zip(axes[1:], [0, 1, 0], [f"{names[0]} pred", f"{names[1]} pred", f"{names[0]} abs error"]):
        values = pred[:, col] if "error" not in title else err[:, col]
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=values, s=2, cmap="viridis")
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(sc, ax=ax)

    plt.tight_layout()
    plt.show()


checkpoint = train()
