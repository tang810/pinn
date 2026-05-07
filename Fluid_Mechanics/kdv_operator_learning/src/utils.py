import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_results(data, pred=None, mode="train", output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    x = data["x"][0]
    ref = data["ref"][0]

    plt.figure(figsize=(8, 5))
    plt.plot(x, ref, label="Reference", color="black", linewidth=2)

    if pred is not None:
        plt.plot(x, pred[0], "--", label="Prediction", color="red", linewidth=1.5)

    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title(f"KdV Solution ({mode})")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(output_dir, f"{mode}_result.png")
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_loss_curve(loss_history, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, label="Training Loss", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("PINN Training Loss Curve")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(save_path, dpi=300)
    plt.close()


def gradients(y, x):
    grad_outputs = torch.ones_like(y)
    return torch.autograd.grad(y, x, grad_outputs=grad_outputs, create_graph=True)[0]


def ref_sol(rho, h1, a, X):
    h2 = 1
    g = 9.81
    h = h2 / h1
    c0 = np.sqrt((g * h2 * (1 - rho)) / (h + rho))
    c1 = -1.5 * c0 * ((rho - h**2) / (rho * h2 + h * h2))
    c2 = (1 / 6) * c0 * ((rho * h2**2 * h1 + h1**2 * h2) / (rho * h1 + h2))
    lam = np.sqrt(12 * c2 / (a * c1))
    sol = a / (np.cosh(X / lam) ** 2)
    return sol