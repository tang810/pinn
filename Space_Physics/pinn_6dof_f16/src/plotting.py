from __future__ import annotations
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

def plot_loss(loss_hist: np.ndarray, out_path: Optional[str] = None) -> None:
    plt.figure(figsize=(6,3))
    plt.plot(loss_hist)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss (physics residuals)")
    plt.grid(True)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200)
    plt.close()

def plot_coeff_compare(t: np.ndarray, coeff_true: np.ndarray, coeff_pred: np.ndarray,
                       out_path: Optional[str] = None) -> None:
    names = ["CX", "CY", "CZ", "Cl", "Cm", "Cn"]
    plt.figure(figsize=(10, 10))
    for i, name in enumerate(names, start=1):
        plt.subplot(3, 2, i)
        plt.plot(t, coeff_true[:, i-1], label=f"{name} true")
        plt.plot(t, coeff_pred[:, i-1], "--", label=f"{name} PINN")
        plt.grid(True)
        plt.ylabel(name)
        if i >= 5:
            plt.xlabel("time [s]")
        plt.legend(fontsize=8)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200)
    plt.close()
