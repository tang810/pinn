# src/visualize.py
import os
import matplotlib.pyplot as plt

def plot_history(history, save_path=None, show=True):
    """
    history: dict with keys ['total', 'pde', 'bc']
    save_path: str or None, e.g. 'results/quick_loss.png'
    show: bool
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].semilogy(history["total"], 'b-')
    axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')

    axes[1].semilogy(history["pde"], 'r-')
    axes[1].set_title('PDE Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')

    axes[2].semilogy(history["bc"], 'g-')
    axes[2].set_title('Boundary Loss')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()
