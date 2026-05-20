
"""
Generic train entry for this PINN4Science project.

Purpose:
    - Load a numeric dataset from data_path.
    - Train a small PyTorch regression model.
    - Save a .pth model file for later public-asset upload.

Path arguments may be left empty in notebooks; pass explicit public-asset paths online.
Supported dataset formats: .json, .npz, .npy, .csv, .txt, .mat (if scipy is installed).
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


PROJECT_NAME = "Beijing_Snowfall_PINN"

DATA_HASH = "49333df65fd74522933e6c2248ffd5d4"
MODEL_HASH = "49333df65fd74522933e6c2248ffd5d4"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/minio/Resource/{MODEL_HASH}/trained_model_beijing_snowfall_pinn.pt")

N_ITERS = 1000
PRINT_EVERY = 100
BATCH_SIZE = 4096
HIDDEN_DIM = 64
N_LAYERS = 4
LR = 1e-3
SEED = 42


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
    # Deterministic snowfall-like target from district coordinates. The uploaded
    # JSON provides geography, while this target keeps the example self-contained.
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
            n = min(np.asarray(a).reshape(np.asarray(a).shape[0], -1).shape[0] for a in numeric[:2])
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


def run_train(
    data_path="",
    model_path="",
    search_root="/mnt/minio",
    n_iters=1000,
    print_every=100,
    batch_size=4096,
    hidden_dim=64,
    n_layers=4,
    lr=1e-3,
    seed=42,
):
    set_seed(seed)
    if not data_path:
        data_path = find_dataset(search_root=search_root)
    if not model_path:
        model_path = "trained_model.pth"

    X, Y = load_dataset(data_path)
    x_mean, x_std = X.mean(axis=0, keepdims=True), X.std(axis=0, keepdims=True) + 1e-8
    y_mean, y_std = Y.mean(axis=0, keepdims=True), Y.std(axis=0, keepdims=True) + 1e-8
    Xn = (X - x_mean) / x_std
    Yn = (Y - y_mean) / y_std

    device = get_device()
    model = MLP(Xn.shape[1], Yn.shape[1], hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    print(f"[{PROJECT_NAME}] device={device}")
    print(f"[{PROJECT_NAME}] data_path={data_path}")
    print(f"[{PROJECT_NAME}] model_path={model_path}")
    print(f"[{PROJECT_NAME}] X={X.shape}, Y={Y.shape}")

    n = Xn.shape[0]
    for it in range(1, n_iters + 1):
        idx = np.random.choice(n, size=min(batch_size, n), replace=False)
        xb = torch.tensor(Xn[idx], dtype=torch.float32, device=device)
        yb = torch.tensor(Yn[idx], dtype=torch.float32, device=device)
        pred = model(xb)
        loss = torch.mean((pred - yb) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it == 1 or it % print_every == 0 or it == n_iters:
            row = {"iter": it, "loss": float(loss.detach().cpu())}
            history.append(row)
            print(f"Iter {it:5d}/{n_iters} | loss={row['loss']:.6e}")

    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    checkpoint = {
        "project": PROJECT_NAME,
        "state_dict": model.state_dict(),
        "in_dim": Xn.shape[1],
        "out_dim": Yn.shape[1],
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "history": history,
        "data_path": data_path,
    }
    torch.save(checkpoint, model_path)
    print(f"Saved model: {model_path}")
    return {"model": model, "model_path": model_path, "data_path": data_path, "history": history}


def parse_args():
    parser = argparse.ArgumentParser(description=f"Generic train entry for {PROJECT_NAME}")
    parser.add_argument("--data-path", default="")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--search-root", default="/mnt/minio")
    parser.add_argument("--n-iters", type=int, default=1000)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


if __name__ == "__main__":
    train_result = run_train(
        data_path=DATA_PATH,
        model_path=MODEL_PATH,
        n_iters=N_ITERS,
        print_every=PRINT_EVERY,
        batch_size=BATCH_SIZE,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        lr=LR,
        seed=SEED,
    )
