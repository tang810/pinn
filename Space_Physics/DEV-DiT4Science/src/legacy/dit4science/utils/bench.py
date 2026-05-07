import numpy as np
import torch
import matplotlib.pyplot as plt

# Helper function to calculate L2 Error
def get_error(pred, truth, ord=2):
    """
    Compute the relative error of two input. input are supposed to have
    shape of (channel, height, width). Other shape might work but please
    be careful and aware of what you are doing.
    
    pred: The prediction output by model.
    truth: The ground truth against which error are measured.
    ord: the order of norm.
    """
    pred, truth = [vec.flatten() for vec in [pred, truth]]
    nume = torch.linalg.norm((pred - truth), ord=ord)
    deno = torch.linalg.norm(truth, ord=ord)
    return nume / deno

# Helper functions to plot
def plot_error_curve(error_p, error_ux, error_uy, epochs):
    """Plot three relative error against epoch in one figure

    Args:
        error_p (List of float)
        error_ux (List of float)
        error_uy (List of float)
    """
    fig, ax = plt.subplots(figsize=(10,2.5))
    ax.plot(epochs, error_p, label='$P$', color="blue")
    ax.plot(epochs, error_ux, label='$U_x$', color="green")
    ax.plot(epochs, error_uy, label='$U_y$', color="yellow")
    ax.yaxis.grid(True, alpha=0.3)
    ax.legend()

    return fig, ax