#!/usr/bin/env python3
import argparse
import os
import random

import numpy as np
import torch

from src.loss import compute_loss
from src.network import PINN
from src.utils import generate_and_save_data, load_data
from src.visualize import save_loss_plot


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(base_dir: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)


def maybe_prepare_data(data_mode: str, data_dir: str, x_pde_n: int, x_boundary_n: int) -> None:
    required = ["x_pde.npy", "x_dir.npy", "u_dir.npy", "x_neu.npy", "n_neu.npy", "t_neu.npy"]
    has_all = all(os.path.exists(os.path.join(data_dir, f)) for f in required)

    if data_mode == "simul":
        generate_and_save_data(x_pde_n=x_pde_n, x_boundary_n=x_boundary_n, data_dir=data_dir)
        return

    if not has_all:
        print("[warn] real data files missing. Falling back to simulated data generation.")
        generate_and_save_data(x_pde_n=x_pde_n, x_boundary_n=x_boundary_n, data_dir=data_dir)


def run(mode, data, data_path, epochs, lr, x_pde_n, x_boundary_n, save_every, show_plots, device_arg):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = resolve_path(base_dir, data_path)
    model_dir = resolve_path(base_dir, "model")
    results_dir = resolve_path(base_dir, "results")

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device(device_arg)
    lambda_, mu = 1.0, 1.0

    maybe_prepare_data(data, data_dir, x_pde_n, x_boundary_n)
    x_pde, x_dir, u_dir, x_neu, n_neu, t_neu = load_data(data_dir=data_dir)

    x_pde = x_pde.to(device).requires_grad_(True)
    x_dir = x_dir.to(device).requires_grad_(True)
    u_dir = u_dir.to(device)
    x_neu = x_neu.to(device).requires_grad_(True)
    n_neu = n_neu.to(device)
    t_neu = t_neu.to(device)

    model = PINN().to(device)

    if mode == "quick":
        model_path = os.path.join(model_dir, "pinn_final.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[quick] model file not found: {model_path}. Please run --mode train first.")

        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        print(f"[quick] Loaded model from {model_path}")

        loss, l_pde, l_dir, l_neu = compute_loss(model, x_pde, x_dir, x_neu, n_neu, lambda_, mu)
        print(
            f"[quick] Loss | Total={loss.item():.3e} | "
            f"PDE={l_pde.item():.3e} | Dir={l_dir.item():.3e} | Neu={l_neu.item():.3e}"
        )
        save_loss_plot([loss.item()], file_path=os.path.join(results_dir, "quick_loss.png"))
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []

    for epoch in range(epochs + 1):
        model.train()
        loss, l_pde, l_dir, l_neu = compute_loss(model, x_pde, x_dir, x_neu, n_neu, lambda_, mu)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        if epoch % save_every == 0:
            print(
                f"Epoch {epoch}/{epochs} | Loss={loss.item():.3e} | "
                f"PDE={l_pde.item():.3e} | Dir={l_dir.item():.3e} | Neu={l_neu.item():.3e}"
            )
            save_loss_plot(train_losses, file_path=os.path.join(results_dir, f"loss_epoch_{epoch}.png"))

            if show_plots:
                import matplotlib.pyplot as plt

                plt.figure()
                plt.plot(train_losses)
                plt.title(f"Loss at epoch {epoch}")
                plt.grid(True)
                plt.show()

    torch.save(model.state_dict(), os.path.join(model_dir, "pinn_final.pt"))
    save_loss_plot(train_losses, file_path=os.path.join(results_dir, "loss_curve_final.png"))
    print("[main] Training finished.")


def main():
    parser = argparse.ArgumentParser(description="Navier-Cauchy PINN Training / Quick Mode")
    parser.add_argument("--mode", choices=["train", "quick"], default="quick")
    parser.add_argument("--data", choices=["simul", "real"], default="simul")
    parser.add_argument("--data_path", type=str, default="data", help="Data directory path")
    parser.add_argument("--show_plots", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--x_pde_n", type=int, default=200)
    parser.add_argument("--x_boundary_n", type=int, default=240)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA unavailable. Falling back to CPU.")
        args.device = "cpu"

    set_seed(42)

    print("=" * 60)
    print(f"mode: {args.mode}")
    print(f"data: {args.data}")
    print(f"data_path: {args.data_path}")
    print(f"device: {args.device}")
    print("=" * 60)

    run(
        mode=args.mode,
        data=args.data,
        data_path=args.data_path,
        epochs=args.epochs,
        lr=args.lr,
        x_pde_n=args.x_pde_n,
        x_boundary_n=args.x_boundary_n,
        save_every=args.save_every,
        show_plots=args.show_plots,
        device_arg=args.device,
    )


if __name__ == "__main__":
    main()