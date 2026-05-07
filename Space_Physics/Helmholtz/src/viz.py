import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def save_loss_curve(loss_hist, save_path):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.semilogy(loss_hist["total"], label="total")
    plt.semilogy(loss_hist["pde"], label="pde")
    plt.semilogy(loss_hist["bc"], label="bc")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Helmholtz PINN Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_prediction_plot(model, save_path):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    x = torch.linspace(-1, 1, 400).reshape(-1, 1)
    model.eval()
    with torch.no_grad():
        u_pred = model(x).cpu().numpy().flatten()

    x_np = x.numpy().flatten()
    # Reference-like shape for visualization only.
    u_ref = np.sin(float(model.k.detach().cpu()) * x_np) / np.sin(float(model.k.detach().cpu()))

    plt.figure(figsize=(8, 4))
    plt.plot(x_np, u_pred, label="PINN prediction")
    plt.plot(x_np, u_ref, "--", label="reference")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("1D Helmholtz Solution")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
