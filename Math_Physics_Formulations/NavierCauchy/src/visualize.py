import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_loss_plot(history, file_path="results/loss_curve.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(history, label="Total Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()
    print(f"[results] Loss curve saved to {file_path}")