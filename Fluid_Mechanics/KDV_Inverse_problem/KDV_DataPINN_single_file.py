"""
Single-file PyTorch data-PINN for the KdV inverse/data-fitting example.

Dataset format:
    npz file with keys:
        x:    shape (Nx, 1) or (Nx,)
        t:    shape (Nt, 1) or (Nt,)
        usol: shape (Nx, Nt)

Notebook usage:
    import sys
    sys.path.insert(0, "/path/to/KDV_Inverse_problem")
    import KDV_DataPINN_single_file as kdv

    result = kdv.train(
        data_path="/path/to/heat_eq_data.npz",
        n_iters=1000,
        print_every=100,
    )

No model file, image, or result file is written unless you explicitly add it.
"""

from dataclasses import dataclass
import argparse
import random

import numpy as np
import torch
import torch.nn as nn


@dataclass
class TrainConfig:
    data_path: str = "data/heat_eq_data.npz"
    n_iters: int = 1000
    print_every: int = 100
    n_data: int = 2048
    n_pde: int = 1024
    hidden_dim: int = 64
    n_layers: int = 4
    lr: float = 1e-3
    seed: int = 42
    w_data: float = 10.0
    w_pde: float = 1.0


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_kdv_npz(data_path):
    data = np.load(data_path)
    x = data["x"].reshape(-1).astype(np.float32)
    t = data["t"].reshape(-1).astype(np.float32)
    u = data["usol"].astype(np.float32)
    if u.shape != (len(x), len(t)):
        raise ValueError(
            f"Expected usol shape {(len(x), len(t))}, got {u.shape}. "
            "The npz should contain x, t, usol."
        )

    X_grid, T_grid = np.meshgrid(x, t, indexing="ij")
    X_raw = np.stack([X_grid.reshape(-1), T_grid.reshape(-1)], axis=1)
    y_raw = u.reshape(-1, 1)

    # Normalize inputs to roughly [-1, 1] for easier training.
    x_min, x_max = float(x.min()), float(x.max())
    t_min, t_max = float(t.min()), float(t.max())
    u_mean, u_std = float(y_raw.mean()), float(y_raw.std() + 1e-8)

    X = X_raw.copy()
    X[:, 0] = 2.0 * (X[:, 0] - x_min) / (x_max - x_min) - 1.0
    X[:, 1] = 2.0 * (X[:, 1] - t_min) / (t_max - t_min) - 1.0
    y = (y_raw - u_mean) / u_std

    meta = {
        "x": x,
        "t": t,
        "u": u,
        "x_min": x_min,
        "x_max": x_max,
        "t_min": t_min,
        "t_max": t_max,
        "u_mean": u_mean,
        "u_std": u_std,
    }
    return X.astype(np.float32), y.astype(np.float32), meta


class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=64, n_layers=4):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class KDVDataPINN(nn.Module):
    def __init__(self, hidden_dim=64, n_layers=4):
        super().__init__()
        self.net = MLP(2, hidden_dim, n_layers)
        self.c0_raw = nn.Parameter(torch.tensor(0.0))
        self.c1_raw = nn.Parameter(torch.tensor(0.0))
        self.c2_raw = nn.Parameter(torch.tensor(-2.0))

    def forward(self, xt):
        return self.net(xt)

    def coefficients(self):
        c0 = self.c0_raw
        c1 = self.c1_raw
        # Positive dispersion coefficient, initialized small.
        c2 = torch.nn.functional.softplus(self.c2_raw)
        return c0, c1, c2


def sample_batch(X, y, n, device):
    idx = np.random.choice(X.shape[0], size=min(n, X.shape[0]), replace=False)
    xb = torch.tensor(X[idx], dtype=torch.float32, device=device)
    yb = torch.tensor(y[idx], dtype=torch.float32, device=device)
    return xb, yb


def pde_residual(model, xt):
    xt = xt.clone().detach().requires_grad_(True)
    u = model(xt)
    grad_u = torch.autograd.grad(
        u,
        xt,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True,
    )[0]
    u_x = grad_u[:, 0:1]
    u_t = grad_u[:, 1:2]
    grad_u_x = torch.autograd.grad(
        u_x,
        xt,
        grad_outputs=torch.ones_like(u_x),
        retain_graph=True,
        create_graph=True,
    )[0]
    u_xx = grad_u_x[:, 0:1]
    grad_u_xx = torch.autograd.grad(
        u_xx,
        xt,
        grad_outputs=torch.ones_like(u_xx),
        retain_graph=True,
        create_graph=True,
    )[0]
    u_xxx = grad_u_xx[:, 0:1]
    c0, c1, c2 = model.coefficients()
    return u_t + c0 * u_x + c1 * u * u_x + c2 * u_xxx


def train(
    data_path,
    n_iters=1000,
    print_every=100,
    n_data=2048,
    n_pde=1024,
    hidden_dim=64,
    n_layers=4,
    lr=1e-3,
    seed=42,
    w_data=10.0,
    w_pde=1.0,
):
    set_seed(seed)
    device = get_device()
    X, y, meta = load_kdv_npz(data_path)

    model = KDVDataPINN(hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    print(f"Using device: {device}")
    print(f"Loaded data: X={X.shape}, y={y.shape}, data_path={data_path}")

    for it in range(1, n_iters + 1):
        xb, yb = sample_batch(X, y, n_data, device)
        xf, _ = sample_batch(X, y, n_pde, device)

        pred = model(xb)
        loss_data = torch.mean((pred - yb) ** 2)
        res = pde_residual(model, xf)
        loss_pde = torch.mean(res ** 2)
        loss = w_data * loss_data + w_pde * loss_pde

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it == 1 or it % print_every == 0 or it == n_iters:
            c0, c1, c2 = [float(v.detach().cpu()) for v in model.coefficients()]
            row = {
                "iter": it,
                "loss": float(loss.detach().cpu()),
                "data": float(loss_data.detach().cpu()),
                "pde": float(loss_pde.detach().cpu()),
                "c0": c0,
                "c1": c1,
                "c2": c2,
            }
            history.append(row)
            print(
                f"Iter {it:5d}/{n_iters} | loss={row['loss']:.3e} | "
                f"data={row['data']:.3e} | pde={row['pde']:.3e} | "
                f"c0={c0:.3e} c1={c1:.3e} c2={c2:.3e}"
            )

    return {
        "model": model,
        "history": history,
        "meta": meta,
        "device": device,
        "X_norm": X,
        "y_norm": y,
    }


def predict_grid(result, nx=None, nt=None):
    model = result["model"]
    meta = result["meta"]
    device = result["device"]
    x = meta["x"] if nx is None else np.linspace(meta["x_min"], meta["x_max"], nx, dtype=np.float32)
    t = meta["t"] if nt is None else np.linspace(meta["t_min"], meta["t_max"], nt, dtype=np.float32)
    X_grid, T_grid = np.meshgrid(x, t, indexing="ij")

    xt = np.stack([X_grid.reshape(-1), T_grid.reshape(-1)], axis=1).astype(np.float32)
    xt_norm = xt.copy()
    xt_norm[:, 0] = 2.0 * (xt_norm[:, 0] - meta["x_min"]) / (meta["x_max"] - meta["x_min"]) - 1.0
    xt_norm[:, 1] = 2.0 * (xt_norm[:, 1] - meta["t_min"]) / (meta["t_max"] - meta["t_min"]) - 1.0

    model.eval()
    with torch.no_grad():
        y_norm = model(torch.tensor(xt_norm, dtype=torch.float32, device=device)).cpu().numpy()
    U_pred = y_norm.reshape(len(x), len(t)) * meta["u_std"] + meta["u_mean"]
    return X_grid, T_grid, U_pred


def evaluate(result):
    _, _, U_pred = predict_grid(result)
    U_true = result["meta"]["u"]
    rel_l2 = np.linalg.norm(U_pred - U_true) / (np.linalg.norm(U_true) + 1e-12)
    mae = np.mean(np.abs(U_pred - U_true))
    return {"relative_l2": float(rel_l2), "mae": float(mae)}


def parse_args():
    parser = argparse.ArgumentParser(description="Single-file KdV data-PINN trainer")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--n-iters", type=int, default=1000)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--n-data", type=int, default=2048)
    parser.add_argument("--n-pde", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = train(
        data_path=args.data_path,
        n_iters=args.n_iters,
        print_every=args.print_every,
        n_data=args.n_data,
        n_pde=args.n_pde,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        lr=args.lr,
    )
    print("Evaluation:", evaluate(result))
