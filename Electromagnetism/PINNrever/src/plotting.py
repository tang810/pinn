import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_two_panel(points_xy: np.ndarray, left_values: np.ndarray, right_values: np.ndarray, left_title: str, right_title: str, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    x = points_xy[:, 0]
    y = points_xy[:, 1]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(x, y, c=left_values, cmap="viridis", s=8)
    plt.colorbar()
    plt.title(left_title)

    plt.subplot(1, 2, 2)
    plt.scatter(x, y, c=right_values, cmap="viridis", s=8)
    plt.colorbar()
    plt.title(right_title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_loss_curve(loss_hist, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(loss_hist)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
