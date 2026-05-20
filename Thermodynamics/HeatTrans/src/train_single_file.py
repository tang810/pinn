
"""
Generic train entry for this PINN4Science project.

Purpose:
    - Load a numeric dataset from data_path.
    - Train a small PyTorch regression model.
    - Save a .pth model file for later public-asset upload.

Path arguments may be left empty in notebooks; pass explicit public-asset paths online.
Supported dataset formats: .npz, .npy, .csv, .txt, .mat (if scipy is installed).
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


PROJECT_NAME = "HeatTrans"
DATA_FILE_NAME = "data_7246.txt"


def here():
    try:
        return Path(__file__).resolve()
    except NameError:
        return Path.cwd()


def default_data_path():
    base = here()
    candidates = [
        Path.cwd() / "Thermodynamics" / "Multiphysics" / "data" / DATA_FILE_NAME,
        base.parents[2] / "Multiphysics" / "data" / DATA_FILE_NAME if len(base.parents) > 2 else Path(),
        Path("/mnt/minio/userdk8e2v7l/Thermodynamics/Multiphysics/data") / DATA_FILE_NAME,
        Path("/mnt/minio/userdk8e2v7l/public/Resource") / DATA_FILE_NAME,
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return str(candidates[0])


def default_model_path():
    base = here()
    if len(base.parents) > 1:
        return str(base.parents[1] / "model" / "trained_model_heattrans.pt")
    return str(Path("model") / "trained_model_heattrans.pt")


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


def find_dataset(search_root="/mnt/minio", extensions=(".npz", ".npy", ".csv", ".txt", ".mat")):
    shared = default_data_path()
    if os.path.exists(shared):
        return shared
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


def load_dataset(data_path):
    if not data_path or not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path!r}")
    suffix = Path(data_path).suffix.lower()

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
    data_path=None,
    model_path=None,
    search_root="/mnt/minio",
    n_iters=200,
    print_every=25,
    batch_size=512,
    hidden_dim=32,
    n_layers=2,
    lr=1e-3,
    seed=42,
):
    set_seed(seed)
    if not data_path:
        data_path = default_data_path()
    if not model_path:
        model_path = default_model_path()

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
    parser.add_argument("--n-iters", type=int, default=200)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_known_args()[0]


if __name__ == "__main__":
    args = parse_args()
    run_train(**vars(args))
