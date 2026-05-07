import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_quick_result(x, y_true, y_pred, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y_true, label="truth", linewidth=2)
    plt.plot(x, y_pred, "--", label="prediction", linewidth=1.8)
    plt.xlabel("x")
    plt.ylabel("u_x")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def compute_metrics(y_true, y_pred):
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    return {"mae": mae, "rmse": rmse}