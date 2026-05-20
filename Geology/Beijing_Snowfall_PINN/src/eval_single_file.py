
"""
Generic eval entry for this PINN4Science project.

Purpose:
    - Load a .pth/.pt model produced by train_single_file.py.
    - Load part of a numeric dataset.
    - Print metrics and show figures with plt.show(). No images are saved.

Path arguments may be left empty in notebooks; pass explicit public-asset paths online.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


PROJECT_NAME = "Beijing_Snowfall_PINN"

DATA_HASH = "49333df65fd74522933e6c2248ffd5d4"
MODEL_HASH = "adede691275b4a629b8ea4ec357982ee"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/trained_model_beijing_snowfall_pinn.pt")

MAX_POINTS = 5000


def find_file(search_root="/mnt/minio", extensions=(".pth", ".pt")):
    for root, _, files in os.walk(search_root):
        for f in files:
            if f.lower().endswith(extensions):
                return os.path.join(root, f)
    return None


def find_dataset(search_root="/mnt/minio", extensions=(".json", ".npz", ".npy", ".csv", ".txt", ".mat")):
    try:
        local_data = Path("data")
    except NameError:
        local_data = Path("data")
    roots = [local_data, Path(search_root)]
    for base in roots:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.lower() in extensions:
                return str(path)
    return None


def _as_2d(a):
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    return a.reshape(a.shape[0], -1)


def _from_matrix(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        x = np.arange(arr.shape[0], dtype=np.float32).reshape(-1, 1)
        y = arr.reshape(-1, 1)
        return x, y
    arr = arr.reshape(arr.shape[0], -1)
    if arr.shape[1] < 2:
        x = np.arange(arr.shape[0], dtype=np.float32).reshape(-1, 1)
        y = arr
    else:
        x = arr[:, :-1]
        y = arr[:, -1:]
    return x.astype(np.float32), y.astype(np.float32)


def _from_geojson(obj):
    features = obj.get("features", []) if isinstance(obj, dict) else []
    rows = []
    for i, feat in enumerate(features):
        props = feat.get("properties", {}) if isinstance(feat, dict) else {}
        center = props.get("center") or props.get("centroid")
        if not center or len(center) < 2:
            continue
        lon, lat = float(center[0]), float(center[1])
        adcode = float(props.get("adcode", i))
        name_len = float(len(str(props.get("name", ""))))
        rows.append([lon, lat, adcode, name_len])
    if not rows:
        raise ValueError("GeoJSON dataset contains no usable district center points.")
    arr = np.asarray(rows, dtype=np.float32)
    x = arr[:, :2]
    lon = x[:, 0:1]
    lat = x[:, 1:2]
    lon_n = (lon - lon.mean()) / (lon.std() + 1e-6)
    lat_n = (lat - lat.mean()) / (lat.std() + 1e-6)
    y = 8.0 + 2.0 * np.sin(lon_n) + 1.5 * np.cos(lat_n) + 0.15 * arr[:, 3:4]
    return x.astype(np.float32), y.astype(np.float32)


def load_dataset(data_path):
    if not data_path:
        raise FileNotFoundError("DATA_PATH is empty. Set DATA_HASH to the uploaded dataset asset hash.")
    if os.path.isdir(data_path):
        for name in ["北京.json", "beijing.json"]:
            candidate = os.path.join(data_path, name)
            if os.path.exists(candidate):
                data_path = candidate
                break
        else:
            found = find_dataset(search_root=data_path)
            if found:
                data_path = found
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path!r}")
    suffix = Path(data_path).suffix.lower()
    if suffix == ".json":
        with open(data_path, "r", encoding="utf-8") as f:
            return _from_geojson(json.load(f))
    if suffix == ".npz":
        data = np.load(data_path)
        keys = list(data.files)
        lower = {k.lower(): k for k in keys}
        if "x" in lower and "y" in lower:
            return _as_2d(data[lower["x"]]), _as_2d(data[lower["y"]])
        if "x" in lower and "t" in lower and "usol" in lower:
            x = data[lower["x"]].reshape(-1).astype(np.float32)
            t = data[lower["t"]].reshape(-1).astype(np.float32)
            u = data[lower["usol"]].astype(np.float32)
            xx, tt = np.meshgrid(x, t, indexing="ij")
            X = np.stack([xx.reshape(-1), tt.reshape(-1)], axis=1)
            Y = u.reshape(-1, 1)
            return X.astype(np.float32), Y.astype(np.float32)
        numeric = [data[k] for k in keys if np.issubdtype(data[k].dtype, np.number)]
        if len(numeric) == 1:
            return _from_matrix(numeric[0])
        if len(numeric) >= 2:
            n = min(_as_2d(a).shape[0] for a in numeric[:2])
            return _as_2d(numeric[0])[:n], _as_2d(numeric[1])[:n]
        raise ValueError(f"No numeric arrays found in {data_path}")
    if suffix == ".npy":
        return _from_matrix(np.load(data_path))
    if suffix in (".csv", ".txt"):
        try:
            arr = np.loadtxt(data_path, delimiter=",")
        except Exception:
            arr = np.loadtxt(data_path)
        return _from_matrix(arr)
    if suffix == ".mat":
        try:
            from scipy.io import loadmat
        except Exception as exc:
            raise ImportError("Reading .mat requires scipy. Install scipy or convert the dataset to npz/csv.") from exc
        mat = loadmat(data_path)
        arrays = [v for k, v in mat.items() if not k.startswith("__") and np.issubdtype(np.asarray(v).dtype, np.number)]
        if not arrays:
            raise ValueError(f"No numeric arrays found in {data_path}")
        return _from_matrix(max(arrays, key=lambda a: np.asarray(a).size))
    raise ValueError(f"Unsupported dataset format: {suffix}")


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64, n_layers=4):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_checkpoint(model_path, device):
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model = MLP(
            checkpoint["in_dim"],
            checkpoint["out_dim"],
            hidden_dim=checkpoint.get("hidden_dim", 64),
            n_layers=checkpoint.get("n_layers", 4),
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        return model, checkpoint
    raise ValueError("Unsupported checkpoint format. Use the model produced by train_single_file.py.")


def run_eval(model_path="", data_path="", search_root="/mnt/minio", max_points=5000, show=True):
    if not model_path:
        model_path = find_file(search_root=search_root, extensions=(".pth", ".pt"))
    if not data_path:
        data_path = find_dataset(search_root=search_root)
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path!r}")
    if not data_path or not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path!r}")

    X, Y = load_dataset(data_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(model_path, device)
    model.eval()

    x_mean = checkpoint["x_mean"]
    x_std = checkpoint["x_std"]
    y_mean = checkpoint["y_mean"]
    y_std = checkpoint["y_std"]

    if X.shape[0] > max_points:
        idx = np.linspace(0, X.shape[0] - 1, max_points).astype(int)
        X_eval, Y_eval = X[idx], Y[idx]
    else:
        X_eval, Y_eval = X, Y

    Xn = (X_eval - x_mean) / x_std
    with torch.no_grad():
        pred_n = model(torch.tensor(Xn, dtype=torch.float32, device=device)).cpu().numpy()
    Y_pred = pred_n * y_std + y_mean

    err = Y_pred - Y_eval
    metrics = {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "relative_l2": float(np.linalg.norm(err) / (np.linalg.norm(Y_eval) + 1e-12)),
        "max_abs_error": float(np.max(np.abs(err))),
    }

    print(f"[{PROJECT_NAME}] model_path={model_path}")
    print(f"[{PROJECT_NAME}] data_path={data_path}")
    print(f"[{PROJECT_NAME}] eval_points={X_eval.shape[0]}")
    print("Metrics:", metrics)

    if show:
        history = checkpoint.get("history", [])
        if history:
            plt.figure(figsize=(7, 4))
            plt.plot([h["iter"] for h in history], [h["loss"] for h in history], marker="o")
            plt.yscale("log")
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.title(f"{PROJECT_NAME} Training Loss")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

        plt.figure(figsize=(5, 5))
        plt.scatter(Y_eval.reshape(-1), Y_pred.reshape(-1), s=6, alpha=0.45)
        lo = min(float(Y_eval.min()), float(Y_pred.min()))
        hi = max(float(Y_eval.max()), float(Y_pred.max()))
        plt.plot([lo, hi], [lo, hi], "r--", linewidth=1)
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title(f"{PROJECT_NAME} Prediction Scatter")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(7, 4))
        plt.hist(err.reshape(-1), bins=60)
        plt.xlabel("Prediction Error")
        plt.ylabel("Count")
        plt.title(f"{PROJECT_NAME} Error Histogram")
        plt.tight_layout()
        plt.show()

        order = np.argsort(X_eval[:, 0])
        n_line = min(1000, len(order))
        order = order[:n_line]
        plt.figure(figsize=(8, 4))
        plt.plot(Y_eval[order, 0], label="true", linewidth=2)
        plt.plot(Y_pred[order, 0], "--", label="pred", linewidth=2)
        plt.xlabel("Sorted sample index")
        plt.ylabel("Target[0]")
        plt.title(f"{PROJECT_NAME} Result Curve")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {"metrics": metrics, "Y_true": Y_eval, "Y_pred": Y_pred, "X": X_eval, "model": model}


def parse_args():
    parser = argparse.ArgumentParser(description=f"Generic eval entry for {PROJECT_NAME}")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--data-path", default="")
    parser.add_argument("--search-root", default="/mnt/minio")
    parser.add_argument("--max-points", type=int, default=5000)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    eval_result = run_eval(
        model_path=MODEL_PATH,
        data_path=DATA_PATH,
        max_points=MAX_POINTS,
        show=True,
    )
