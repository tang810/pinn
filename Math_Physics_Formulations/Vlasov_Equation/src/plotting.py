import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_velocity_slice(v, truth, pred, out_path):
    plt.figure(figsize=(8, 5))
    plt.plot(v, truth, label="truth")
    plt.plot(v, pred, label="prediction")
    plt.xlim(-5, 5)
    plt.ylim(-0.05, 0.6)
    plt.xlabel("Velocity")
    plt.ylabel("Normalized Velocity Distribution Function")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_metrics(truth, pred, out_path):
    mae = float(np.mean(np.abs(truth - pred)))
    rmse = float(np.sqrt(np.mean((truth - pred) ** 2)))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"mae={mae:.8e}\n")
        f.write(f"rmse={rmse:.8e}\n")