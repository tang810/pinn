"""
Notebook-friendly eval file for rans_backstep.

Loads trained_model_rans_backstep.pt and the uploaded .mat dataset, then prints
metrics and shows figures. This eval version uses numpy + scipy + matplotlib
only, matching train_single_file.py.
"""

import os
import random

import numpy as np
import matplotlib.pyplot as plt


DATA_HASH = "910473ad83b04688b8a56c54a1537ecb"
MODEL_HASH = "09e30e2fcf524c15a76466d520c22d1e"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/trained_model_rans_backstep.pt")

SEED = 42


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def resolve_mat_path(data_path):
    if os.path.isfile(data_path):
        return data_path
    for name in ["Re1e5_backStep_L8.mat", "Re1e5_backStep.mat", "f1_result_epoch200000.mat"]:
        path = os.path.join(data_path, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No supported rans_backstep .mat file found under {data_path}")


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
        fields = [data["U_ref"].reshape(-1, 1), data["V_ref"].reshape(-1, 1), data["P_ref"].reshape(-1, 1)]
        names = ["u", "v", "p"]
        if "K_ref" in data:
            fields.append(data["K_ref"].reshape(-1, 1))
            names.append("k")
        xy = np.concatenate([x, y], axis=1)
        target = np.concatenate(fields, axis=1).astype(np.float64)
    elif {"U_pred", "V_pred", "P_pred", "E_pred"}.issubset(data):
        target = np.concatenate(
            [data["U_pred"].reshape(-1, 1), data["V_pred"].reshape(-1, 1), data["P_pred"].reshape(-1, 1), data["E_pred"].reshape(-1, 1)],
            axis=1,
        ).astype(np.float64)
        n = target.shape[0]
        side = int(np.ceil(np.sqrt(n)))
        gx, gy = np.meshgrid(np.linspace(0, 1, side), np.linspace(0, 1, side), indexing="ij")
        xy = np.stack([gx.reshape(-1)[:n], gy.reshape(-1)[:n]], axis=1).astype(np.float64)
        names = ["u_pred", "v_pred", "p_pred", "e_pred"]
    else:
        raise KeyError("Unsupported rans_backstep .mat format.")

    finite = np.isfinite(xy).all(axis=1) & np.isfinite(target).all(axis=1)
    return {"path": mat_path, "xy": xy[finite], "target": target[finite], "target_names": np.asarray(names)}


def feature_map(xy):
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    return np.concatenate(
        [
            np.ones_like(x), x, y, x * y, x**2, y**2, x**3, y**3,
            np.sin(np.pi * x), np.cos(np.pi * x), np.sin(np.pi * y), np.cos(np.pi * y),
            np.sin(np.pi * x) * np.sin(np.pi * y), np.cos(np.pi * x) * np.cos(np.pi * y),
        ],
        axis=1,
    )


def load_checkpoint(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"MODEL_PATH does not exist: {model_path}")
    ckpt = np.load(model_path, allow_pickle=True)
    out = {k: ckpt[k] for k in ckpt.files}
    print("Model loaded:", model_path)
    return out


def predict(xy, checkpoint):
    xy_norm = (xy - checkpoint["x_mean"]) / checkpoint["x_std"]
    return feature_map(xy_norm) @ checkpoint["coeff"] * checkpoint["y_std"] + checkpoint["y_mean"]


def evaluate(model_path=MODEL_PATH, data_path=DATA_PATH):
    set_seed(SEED)
    checkpoint = load_checkpoint(model_path)
    dataset = load_rans_data(data_path)
    pred = predict(dataset["xy"], checkpoint)
    true = dataset["target"]
    err = pred - true

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    rel_l2 = float(np.linalg.norm(err) / (np.linalg.norm(true) + 1e-12))

    print("\nEval result")
    print("  model_path:", model_path)
    print("  data_path:", dataset["path"])
    print(f"  mae: {mae:.6e}")
    print(f"  rmse: {rmse:.6e}")
    print(f"  relative_l2: {rel_l2:.6e}")
    show_result(dataset["xy"], true, pred, np.asarray(checkpoint.get("history", [])), dataset["target_names"])
    return {"mae": mae, "rmse": rmse, "relative_l2": rel_l2}


def show_result(xy, true, pred, history, names):
    err = np.abs(pred - true)
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.5))
    if len(history):
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


eval_result = evaluate()
