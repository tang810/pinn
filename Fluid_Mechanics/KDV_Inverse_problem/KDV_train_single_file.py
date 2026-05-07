"""
Train-only single file for Fluid_Mechanics/KDV_Inverse_problem.

Inputs:
    - heat_eq_data.npz dataset with keys x, t, usol

Output:
    - trained_model.pth

Notebook usage:
    import KDV_train_single_file as train_file
    result = train_file.run_train(
        data_path="/mnt/minio/.../heat_eq_data.npz",
        model_path="/tmp/trained_model.pth",
        n_iters=1000,
    )

This script saves only the model file. It does not save images.
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn


def find_file(root="/mnt/minio", filename="heat_eq_data.npz"):
    for current_root, _, files in os.walk(root):
        if filename in files:
            return os.path.join(current_root, filename)
    return None


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeepXDEFNN(nn.Module):
    """PyTorch replica of dde.maps.FNN([2] + [20] * 3 + [1], "tanh", ...)."""

    def __init__(self, layer_sizes=(2, 20, 20, 20, 1)):
        super().__init__()
        self.linears = nn.ModuleList(
            [nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]
        )
        self.activation = torch.tanh

    def forward(self, x):
        for layer in self.linears[:-1]:
            x = self.activation(layer(x))
        return self.linears[-1](x)


def load_data(data_path):
    data = np.load(data_path)
    required = {"x", "t", "usol"}
    missing = required.difference(data.files)
    if missing:
        raise KeyError(f"Dataset missing keys: {sorted(missing)}. Found keys: {data.files}")

    x = data["x"].reshape(-1).astype(np.float32)
    t = data["t"].reshape(-1).astype(np.float32)
    u = data["usol"].astype(np.float32)
    if u.shape != (len(x), len(t)):
        raise ValueError(f"Expected usol shape {(len(x), len(t))}, got {u.shape}")

    x_grid, t_grid = np.meshgrid(x, t, indexing="ij")
    xt = np.stack([x_grid.reshape(-1), t_grid.reshape(-1)], axis=1).astype(np.float32)
    y = u.reshape(-1, 1).astype(np.float32)
    return xt, y, {"x": x, "t": t, "u": u}


def sample_batch(xt, y, batch_size, device):
    idx = np.random.choice(xt.shape[0], size=min(batch_size, xt.shape[0]), replace=False)
    xb = torch.tensor(xt[idx], dtype=torch.float32, device=device)
    yb = torch.tensor(y[idx], dtype=torch.float32, device=device)
    return xb, yb


def run_train(
    data_path=None,
    model_path="trained_model.pth",
    n_iters=1000,
    print_every=100,
    batch_size=4096,
    lr=1e-3,
    seed=42,
):
    set_seed(seed)
    if data_path is None:
        data_path = find_file("/mnt/minio", "heat_eq_data.npz")
    if data_path is None or not os.path.exists(data_path):
        raise FileNotFoundError(f"Cannot find dataset: {data_path}")

    device = get_device()
    xt, y, meta = load_data(data_path)
    model = DeepXDEFNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    print("Using device:", device)
    print("Using data:", data_path)
    print("Model will be saved to:", model_path)
    print("Dataset:", xt.shape, y.shape)

    for it in range(1, n_iters + 1):
        xb, yb = sample_batch(xt, y, batch_size, device)
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
    try:
        torch.save(model.state_dict(), model_path)
    except OSError as exc:
        raise OSError(
            f"Failed to save model to {model_path!r}. "
            "Please choose a writable path such as './trained_model.pth' or '/tmp/trained_model.pth'."
        ) from exc
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to save model to {model_path!r}. "
            "Please choose a writable path such as './trained_model.pth' or '/tmp/trained_model.pth'."
        ) from exc
    print("Saved model:", model_path)

    return {
        "model": model,
        "model_path": model_path,
        "data_path": data_path,
        "history": history,
        "meta": meta,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train KDV FNN model")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--model-path", type=str, default="trained_model.pth")
    parser.add_argument("--n-iters", type=int, default=1000)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_train(
        data_path=args.data_path,
        model_path=args.model_path,
        n_iters=args.n_iters,
        print_every=args.print_every,
        batch_size=args.batch_size,
        lr=args.lr,
    )
