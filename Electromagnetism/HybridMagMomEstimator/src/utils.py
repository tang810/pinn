import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_str):
    if "cuda" in device_str and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def resolve_history_path(args):
    candidates = []
    if args.history_path:
        candidates.append(args.history_path)
    candidates.extend(
        [
            os.path.join(args.output_dir, "magnetic_moments_history.npy"),
            os.path.join(os.path.dirname(args.output_dir), "data", "magnetic_moments_history.npy"),
        ]
    )
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return ""


def plot_moment_comparison(pred, truth, save_path):
    idx_mx = np.arange(0, 48, 3)
    idx_my = np.arange(1, 48, 3)
    idx_mz = np.arange(2, 48, 3)
    n = np.arange(0, 16, 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].plot(n, pred[idx_mx], color="blue", label="Pred")
    axes[0].plot(n, truth[idx_mx], color="orange", label="True")
    axes[0].set_xlabel("n")
    axes[0].set_ylabel("mx")
    axes[0].grid(True, linestyle="--", alpha=0.7)
    axes[0].legend()

    axes[1].plot(n, pred[idx_my], color="green", label="Pred")
    axes[1].plot(n, truth[idx_my], color="orange", label="True")
    axes[1].set_xlabel("n")
    axes[1].set_ylabel("my")
    axes[1].grid(True, linestyle="--", alpha=0.7)
    axes[1].legend()

    axes[2].plot(n, pred[idx_mz], color="red", label="Pred")
    axes[2].plot(n, truth[idx_mz], color="orange", label="True")
    axes[2].set_xlabel("n")
    axes[2].set_ylabel("mz")
    axes[2].grid(True, linestyle="--", alpha=0.7)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_metrics(metrics, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
