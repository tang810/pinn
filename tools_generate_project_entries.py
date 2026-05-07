from pathlib import Path


TRAIN_TEMPLATE = r'''
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


PROJECT_NAME = Path(__file__).resolve().parent.name


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_dataset(search_root="/mnt/minio", extensions=(".npz", ".npy", ".csv", ".txt", ".mat")):
    local_data = Path(__file__).resolve().parent / "data"
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
    args = parse_args()
    run_train(**vars(args))
'''


EVAL_TEMPLATE = r'''
"""
Generic eval entry for this PINN4Science project.

Purpose:
    - Load a .pth/.pt model produced by train_single_file.py.
    - Load part of a numeric dataset.
    - Print metrics and show figures with plt.show(). No images are saved.

Path arguments may be left empty in notebooks; pass explicit public-asset paths online.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


PROJECT_NAME = Path(__file__).resolve().parent.name


def find_file(search_root="/mnt/minio", extensions=(".pth", ".pt")):
    for root, _, files in os.walk(search_root):
        for f in files:
            if f.lower().endswith(extensions):
                return os.path.join(root, f)
    return None


def find_dataset(search_root="/mnt/minio", extensions=(".npz", ".npy", ".csv", ".txt", ".mat")):
    local_data = Path(__file__).resolve().parent / "data"
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
    args = parse_args()
    run_eval(
        model_path=args.model_path,
        data_path=args.data_path,
        search_root=args.search_root,
        max_points=args.max_points,
        show=not args.no_show,
    )
'''


def project_dirs(root: Path):
    dirs = set()
    excluded = {".conda", ".venv", "venv", "__pycache__", "site-packages", "Lib", "lib"}
    for file_json in root.rglob("file.json"):
        if excluded.intersection(file_json.parts):
            continue
        dirs.add(file_json.parent)
    for pattern in ("main.py", "*_main.py", "*PINN_main.py", "run.py"):
        for path in root.rglob(pattern):
            parts = set(path.parts)
            if "src" in parts or "legacy" in parts or excluded.intersection(parts):
                continue
            dirs.add(path.parent)
    return sorted(dirs)


def main():
    root = Path.cwd()
    created = []
    for directory in project_dirs(root):
        train_path = directory / "train_single_file.py"
        eval_path = directory / "eval_single_file.py"
        if not train_path.exists():
            train_path.write_text(TRAIN_TEMPLATE, encoding="utf-8")
            created.append(train_path)
        if not eval_path.exists():
            eval_path.write_text(EVAL_TEMPLATE, encoding="utf-8")
            created.append(eval_path)
    print(f"Created {len(created)} files")
    for path in created:
        print(path.relative_to(root))


if __name__ == "__main__":
    main()
