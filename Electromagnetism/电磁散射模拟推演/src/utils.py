import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import ticker


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def select_device(device_str):
    if "cuda" in device_str and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def resolve_checkpoint(args):
    candidates = []
    if getattr(args, "ckpt", ""):
        candidates.append(args.ckpt)
    candidates.extend(
        [
            os.path.join(args.model_dir, "final_model.pt"),
            os.path.join(args.output_dir, "final_model.pt"),
            os.path.join(os.path.dirname(args.output_dir), "model", "final_model.pt"),
            os.path.join(os.path.dirname(args.output_dir), "results", "final_model.pt"),
        ]
    )
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return ""


def plot_complex_field(xx, yy, re_field, im_field, save_path, title_left, title_right, radius=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plt.subplots_adjust(wspace=0.3)

    im1 = ax1.imshow(
        re_field,
        extent=[xx.min(), xx.max(), yy.min(), yy.max()],
        origin="lower",
        cmap="jet",
    )
    ax1.set_title(title_left)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    cbar1 = fig.colorbar(im1, ax=ax1, format=ticker.ScalarFormatter(useMathText=True))
    cbar1.formatter.set_powerlimits((-2, 2))
    cbar1.update_ticks()

    if radius is not None:
        circle = plt.Circle((0, 0), float(radius), color="r", fill=False, linestyle="--", linewidth=2)
        ax1.add_patch(circle)

    im2 = ax2.imshow(
        im_field,
        extent=[xx.min(), xx.max(), yy.min(), yy.max()],
        origin="lower",
        cmap="jet",
    )
    ax2.set_title(title_right)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    cbar2 = fig.colorbar(im2, ax=ax2, format=ticker.ScalarFormatter(useMathText=True))
    cbar2.formatter.set_powerlimits((-2, 2))
    cbar2.update_ticks()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
