"""
Notebook-friendly eval file for elasticity_basis_precompute_online_solver_2d.

Online Jupyter usage:
1. Upload the trained .pt model as a public asset.
2. Upload or keep the .npz elasticity dataset as a public asset.
3. Fill MODEL_HASH and DATA_HASH, then run this whole file/cell.

This file loads the model package and part of the dataset, prints final metrics,
and shows figures with plt.show(). It does not save images.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = "b536492325fa4d2ea77495db59dc626a"
MODEL_HASH = "47a4584b15284b449f680e4e71bdd0de"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/trained_model_elasticity_basis.pt")

PREFERRED_DATA_FILE = "real_data.npz"


# =========================
# 2. Model definition
# =========================

class ElasticPINN(nn.Module):
    def __init__(self, in_dim=3, hidden=64, depth=3, out_dim=2):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 3. Data and model utilities
# =========================

def resolve_npz_path(data_path):
    if not data_path:
        raise FileNotFoundError("DATA_PATH is empty. Fill DATA_HASH first.")

    data_path = os.path.expanduser(data_path)
    candidates = []
    if os.path.isfile(data_path):
        candidates.append(data_path)
    elif os.path.isdir(data_path):
        candidates.append(os.path.join(data_path, PREFERRED_DATA_FILE))
        candidates.append(os.path.join(data_path, "real_data.npz"))
        for f in sorted(os.listdir(data_path)):
            if f.endswith(".npz"):
                candidates.append(os.path.join(data_path, f))

    for path in candidates:
        if os.path.isfile(path):
            return path

    raise FileNotFoundError(
        "Could not find elasticity .npz data. Upload a file like real_data.npz into the data asset folder. "
        f"Current DATA_PATH={data_path!r}"
    )


def load_elasticity_data(npz_path):
    data = np.load(npz_path)
    required = ["x", "y", "q", "ux", "uy"]
    for k in required:
        if k not in data:
            raise KeyError(f"The .npz file must contain {k}. Missing: {k}")
    x = data["x"].astype(np.float32).reshape(-1, 1)
    y = data["y"].astype(np.float32).reshape(-1, 1)
    q = data["q"].astype(np.float32).reshape(-1, 1)
    ux = data["ux"].astype(np.float32).reshape(-1, 1)
    uy = data["uy"].astype(np.float32).reshape(-1, 1)
    inp = np.hstack([x, y, q])
    out = np.hstack([ux, uy])
    return inp, out


def load_model_package(model_path, device):
    model_path = os.path.expanduser(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"MODEL_PATH does not exist: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    required = ["state_dict", "x_mean", "x_std", "y_mean", "y_std"]
    for k in required:
        if k not in checkpoint:
            raise KeyError(f"Checkpoint missing key: {k}")
    model = ElasticPINN(
        in_dim=checkpoint.get("in_dim", 3),
        hidden=checkpoint.get("hidden", 64),
        depth=checkpoint.get("depth", 3),
        out_dim=checkpoint.get("out_dim", 2),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def summarize_against_dataset(model, checkpoint, data_path, device, max_points=5000):
    npz_path = resolve_npz_path(data_path)
    inputs, outputs = load_elasticity_data(npz_path)

    if inputs.shape[0] > max_points:
        idx = np.linspace(0, inputs.shape[0] - 1, max_points).astype(int)
        inputs, outputs = inputs[idx], outputs[idx]

    x_mean = checkpoint["x_mean"]
    x_std = checkpoint["x_std"]
    y_mean = checkpoint["y_mean"]
    y_std = checkpoint["y_std"]
    Xn = (inputs - x_mean) / x_std

    with torch.no_grad():
        pred_n = model(torch.tensor(Xn, dtype=torch.float32, device=device)).cpu().numpy()
    pred = pred_n * y_std + y_mean

    err = pred - outputs
    metrics = {
        "dataset_path": npz_path,
        "eval_points": inputs.shape[0],
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "relative_l2": float(np.linalg.norm(err) / (np.linalg.norm(outputs) + 1e-12)),
        "max_abs_error": float(np.max(np.abs(err))),
    }
    return inputs, outputs, pred, metrics


# =========================
# 4. Visualization
# =========================

def show_eval_result(inputs, outputs, pred, checkpoint):
    history = checkpoint.get("history", [])
    if history:
        plt.figure(figsize=(7, 4))
        plt.plot(history, linewidth=1)
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    plt.figure(figsize=(5, 5))
    plt.scatter(outputs.reshape(-1), pred.reshape(-1), s=6, alpha=0.45)
    lo = min(float(outputs.min()), float(pred.min()))
    hi = max(float(outputs.max()), float(pred.max()))
    plt.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("Prediction Scatter")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    err = (pred - outputs).reshape(-1)
    plt.figure(figsize=(7, 4))
    plt.hist(err, bins=60)
    plt.xlabel("Prediction Error")
    plt.ylabel("Count")
    plt.title("Error Histogram")
    plt.tight_layout()
    plt.show()

    ux_true = outputs[:, 0]
    ux_pred = pred[:, 0]
    x_axis = inputs[:, 0]
    idx = np.argsort(x_axis)[: min(512, len(x_axis))]
    plt.figure(figsize=(8, 4))
    plt.plot(x_axis[idx], ux_true[idx], label="true", linewidth=2)
    plt.plot(x_axis[idx], ux_pred[idx], "--", label="pred", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("u_x")
    plt.title("Displacement u_x Curve")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    uy_true = outputs[:, 1]
    uy_pred = pred[:, 1]
    q_vals = inputs[:, 2]
    idx_q = np.argsort(q_vals)[: min(512, len(q_vals))]
    plt.figure(figsize=(8, 4))
    plt.plot(q_vals[idx_q], uy_true[idx_q], label="true", linewidth=2)
    plt.plot(q_vals[idx_q], uy_pred[idx_q], "--", label="pred", linewidth=2)
    plt.xlabel("q")
    plt.ylabel("u_y")
    plt.title("Displacement u_y Curve")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================
# 5. Eval entry
# =========================

def evaluate(model_path=MODEL_PATH, data_path=DATA_PATH, max_points=5000):
    device = get_device()
    model, checkpoint = load_model_package(model_path, device)
    inputs, outputs, pred, metrics = summarize_against_dataset(
        model, checkpoint, data_path, device, max_points=max_points
    )

    print("Eval result")
    print("model_path:", model_path)
    print("data_path:", metrics["dataset_path"])
    print("eval_points:", metrics["eval_points"])
    for key in ("mae", "rmse", "relative_l2", "max_abs_error"):
        print(f"  {key}: {metrics[key]:.6e}")

    show_eval_result(inputs, outputs, pred, checkpoint)
    return {"inputs": inputs, "outputs": outputs, "pred": pred, "metrics": metrics}


eval_result = evaluate()
