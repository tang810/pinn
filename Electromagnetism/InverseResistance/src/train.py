import os
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.PINN import MLP
from src.loss import data_loss, pde_loss
from src.physics import phi_forward_toy_2d, phi_forward_toy_3d

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = str(BASE_DIR / "model")
RESULT_DIR = str(BASE_DIR / "results")
DATA_DIR = str(BASE_DIR / "data")


def configure_paths(model_dir: str = "", result_dir: str = "", data_dir: str = "") -> None:
    global MODEL_DIR, RESULT_DIR, DATA_DIR
    if model_dir:
        MODEL_DIR = model_dir
    if result_dir:
        RESULT_DIR = result_dir
    if data_dir:
        DATA_DIR = data_dir
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)


def _load_obs_from_file(filename: str, fallback: torch.Tensor, device: torch.device) -> torch.Tensor:
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return fallback
    try:
        data = np.load(path, allow_pickle=True)
        if isinstance(data, np.lib.npyio.NpzFile):
            if "d_obs" in data.files:
                val = data["d_obs"]
            elif len(data.files) > 0:
                val = data[data.files[0]]
            else:
                return fallback
        else:
            val = data
        val = np.asarray(val).astype(np.float32).reshape(-1)
        if val.size == 0:
            return fallback
        return torch.tensor([[float(val[0])]], dtype=torch.float32, device=device)
    except Exception:
        return fallback


def _build_obs_2d(data_type: str, M: torch.Tensor, N: torch.Tensor, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        simul = phi_forward_toy_2d(M.unsqueeze(0)) - phi_forward_toy_2d(N.unsqueeze(0))
    if data_type == "simul":
        return simul
    if data_type == "real":
        return _load_obs_from_file("abmn_2d_obs.npz", simul, device)
    return simul


def _build_obs_3d(data_type: str, M: torch.Tensor, N: torch.Tensor, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        simul = phi_forward_toy_3d(M.unsqueeze(0)) - phi_forward_toy_3d(N.unsqueeze(0))
    if data_type == "simul":
        return simul
    if data_type == "real":
        return _load_obs_from_file("abmn_3d_obs.npz", simul, device)
    return simul


def _train_common(
    dim: int, device: torch.device, data_type: str, epochs: int, print_every: int
) -> Tuple[MLP, MLP]:
    if dim == 2:
        Nf = 6000
        x = torch.rand(Nf, 2, device=device)
        x[:, 0] *= 50.0
        x[:, 1] *= 30.0
        A = torch.tensor([10.0, 0.0], device=device)
        B = torch.tensor([40.0, 0.0], device=device)
        M = torch.tensor([22.0, 0.0], device=device)
        N = torch.tensor([28.0, 0.0], device=device)
        d_obs = _build_obs_2d(data_type, M, N, device)
    else:
        Nf = 8000
        x = torch.rand(Nf, 3, device=device)
        x[:, 0] *= 20.0
        x[:, 1] *= 20.0
        x[:, 2] *= 10.0
        A = torch.tensor([5.0, 10.0, 0.0], device=device)
        B = torch.tensor([15.0, 10.0, 0.0], device=device)
        M = torch.tensor([9.0, 10.0, 0.0], device=device)
        N = torch.tensor([11.0, 10.0, 0.0], device=device)
        d_obs = _build_obs_3d(data_type, M, N, device)

    x.requires_grad_(True)
    net_phi = MLP(dim, 1).to(device)
    net_logrho = MLP(dim, 1).to(device)
    opt = torch.optim.Adam(list(net_phi.parameters()) + list(net_logrho.parameters()), lr=1e-3)

    for ep in range(epochs):
        opt.zero_grad()
        loss_p = pde_loss(net_phi, net_logrho, x, A, B)
        loss_d = data_loss(net_phi, M, N, d_obs)
        loss = loss_p + loss_d
        loss.backward()
        opt.step()
        if ep % print_every == 0:
            print(f"[{dim}D][TRAIN] Epoch {ep:5d} | PDE {loss_p.item():.2e} | Data {loss_d.item():.2e}")

    return net_phi, net_logrho


def train_2d(device: torch.device, data_type: str = "simul", epochs: int = 3000) -> None:
    print(f"[TRAIN] 2D PINN | data_type={data_type}")
    net_phi, net_logrho = _train_common(2, device, data_type, epochs=epochs, print_every=max(epochs // 6, 1))
    torch.save(net_phi.state_dict(), os.path.join(MODEL_DIR, "phi_2d.pth"))
    torch.save(net_logrho.state_dict(), os.path.join(MODEL_DIR, "rho_2d.pth"))
    _plot_rho_2d(net_logrho, device, tag="train")


def train_3d(device: torch.device, data_type: str = "simul", epochs: int = 5000) -> None:
    print(f"[TRAIN] 3D PINN | data_type={data_type}")
    net_phi, net_logrho = _train_common(3, device, data_type, epochs=epochs, print_every=max(epochs // 5, 1))
    torch.save(net_phi.state_dict(), os.path.join(MODEL_DIR, "phi_3d.pth"))
    torch.save(net_logrho.state_dict(), os.path.join(MODEL_DIR, "rho_3d.pth"))
    _plot_rho_3d(net_logrho, device, tag="train")


def quick_2d(device: torch.device, data_type: str = "simul") -> None:
    print(f"[QUICK] 2D PINN | data_type={data_type}")
    path = os.path.join(MODEL_DIR, "rho_2d.pth")
    if not os.path.exists(path):
        print("[QUICK] 2D model not found, running warmup training...")
        train_2d(device=device, data_type=data_type, epochs=120)
    net_logrho = MLP(2, 1).to(device)
    net_logrho.load_state_dict(torch.load(path, map_location=device))
    net_logrho.eval()
    _plot_rho_2d(net_logrho, device, tag="quick")


def quick_3d(device: torch.device, data_type: str = "simul") -> None:
    print(f"[QUICK] 3D PINN | data_type={data_type}")
    path = os.path.join(MODEL_DIR, "rho_3d.pth")
    if not os.path.exists(path):
        print("[QUICK] 3D model not found, running warmup training...")
        train_3d(device=device, data_type=data_type, epochs=160)
    net_logrho = MLP(3, 1).to(device)
    net_logrho.load_state_dict(torch.load(path, map_location=device))
    net_logrho.eval()
    _plot_rho_3d(net_logrho, device, tag="quick")


def _plot_rho_2d(net_logrho: MLP, device: torch.device, tag: str = "") -> None:
    X, Z = np.meshgrid(np.linspace(0, 50, 120), np.linspace(0, 30, 80))
    pts = torch.tensor(np.c_[X.ravel(), Z.ravel()], dtype=torch.float32, device=device)
    with torch.no_grad():
        rho = torch.exp(net_logrho(pts)).cpu().numpy().reshape(Z.shape)
    plt.figure(figsize=(8, 4))
    plt.imshow(rho, extent=[0, 50, 30, 0], cmap="jet")
    plt.colorbar(label="rho (ohm m)")
    plt.title(f"2D ABMN PINN Inversion ({tag})")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f"rho_2d_{tag}.png"), dpi=200)
    plt.close()
    print(f"[2D][{tag.upper()}] Result saved")


def _plot_rho_3d(net_logrho: MLP, device: torch.device, tag: str = "") -> None:
    X, Y = np.meshgrid(np.linspace(0, 20, 60), np.linspace(0, 20, 60))
    Z = np.ones_like(X) * 5.0
    pts = torch.tensor(np.c_[X.ravel(), Y.ravel(), Z.ravel()], dtype=torch.float32, device=device)
    with torch.no_grad():
        rho = torch.exp(net_logrho(pts)).cpu().numpy().reshape(X.shape)
    plt.figure(figsize=(5, 4))
    plt.imshow(rho, extent=[0, 20, 20, 0], cmap="jet")
    plt.colorbar(label="rho (ohm m)")
    plt.title(f"3D ABMN PINN Inversion ({tag}, z=5m)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f"rho_3d_{tag}.png"), dpi=200)
    plt.close()
    print(f"[3D][{tag.upper()}] Result saved")
