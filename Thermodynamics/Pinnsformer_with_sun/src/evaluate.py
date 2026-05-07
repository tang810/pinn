import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm
from . import data_loader
from .model import PINNsformer
import os

def _create_animation(data_path, T_pred, save_path):
    data = pd.read_csv(data_path, delimiter=' ', header=None)
    x, y, z, t, T_true = (data.iloc[:, i] for i in range(5))
    unique_times = np.unique(t)

    fig = plt.figure(figsize=(16, 7))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    vmin, vmax = min(T_true.min(), T_pred.min()), max(T_true.max(), T_pred.max())

    scat1 = ax1.scatter([],[],[], c=[], cmap='viridis', vmin=vmin, vmax=vmax, s=100)
    scat2 = ax2.scatter([],[],[], c=[], cmap='viridis', vmin=vmin, vmax=vmax, s=100)

    fig.colorbar(scat1, ax=ax1, shrink=0.6, aspect=10, label='Temperature (K)')
    fig.colorbar(scat2, ax=ax2, shrink=0.6, aspect=10, label='Temperature (K)')
    
    def update(frame):
        time_point = unique_times[frame]
        mask = np.isclose(t, time_point)

        scat1._offsets3d = (x[mask], y[mask], z[mask])
        scat1.set_array(T_true[mask])
        ax1.set_title(f'Ground Truth, Time = {time_point:.2f}s')

        scat2._offsets3d = (x[mask], y[mask], z[mask])
        scat2.set_array(T_pred[mask])
        ax2.set_title(f'Prediction, Time = {time_point:.2f}s')
        
        return scat1, scat2

    for ax in [ax1, ax2]:
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)

    anim = FuncAnimation(fig, update, frames=tqdm(range(len(unique_times)), desc="Creating animation"), interval=100, blit=False)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    anim.save(save_path, writer=PillowWriter(fps=10))
    plt.close(fig)

def _create_error_plots(T_star_true, T_pred, save_dir):
    """
    Creates and saves static error analysis plots.
    1. Prediction vs. True values scatter plot.
    2. Error histogram.
    """
    os.makedirs(save_dir, exist_ok=True)
    error = T_pred - T_star_true
    
    # 1. Prediction vs. True scatter plot
    plt.figure(figsize=(8, 8))
    min_val = min(T_star_true.min(), T_pred.min())
    max_val = max(T_star_true.max(), T_pred.max())
    plt.scatter(T_star_true, T_pred, alpha=0.1, label='Predictions')
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (y=x)')
    plt.xlabel('True Temperature (K)')
    plt.ylabel('Predicted Temperature (K)')
    plt.title('Prediction vs. True Value')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    pred_vs_true_path = os.path.join(save_dir, 'prediction_vs_true.png')
    plt.savefig(pred_vs_true_path)
    plt.close()
    print(f"Prediction vs. True plot saved to {pred_vs_true_path}")

    # 2. Error histogram
    plt.figure(figsize=(10, 6))
    plt.hist(error, bins=50, density=True)
    plt.xlabel('Prediction Error (Predicted - True) (K)')
    plt.ylabel('Probability Density')
    plt.title('Histogram of Prediction Errors')
    plt.grid(True)
    plt.tight_layout()
    error_hist_path = os.path.join(save_dir, 'error_histogram.png')
    plt.savefig(error_hist_path)
    plt.close()
    print(f"Error histogram saved to {error_hist_path}")


def _create_error_animation(data_path, T_pred, save_path):
    """
    Creates a 3D animation of the absolute error over time.
    """
    data = pd.read_csv(data_path, delimiter=' ', header=None)
    x, y, z, t, T_true = (data.iloc[:, i] for i in range(5))
    unique_times = np.unique(t)
    
    abs_error = np.abs(T_pred - T_true.to_numpy())

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    vmin, vmax = np.min(abs_error), np.max(abs_error)

    scat = ax.scatter([],[],[], c=[], cmap='Reds', vmin=vmin, vmax=vmax, s=100)
    fig.colorbar(scat, ax=ax, shrink=0.6, aspect=10, label='Absolute Error (K)')
    
    def update(frame):
        time_point = unique_times[frame]
        mask = np.isclose(t, time_point)

        scat._offsets3d = (x[mask], y[mask], z[mask])
        scat.set_array(abs_error[mask])
        ax.set_title(f'Absolute Error, Time = {time_point:.2f}s')
        
        return scat,

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)

    anim = FuncAnimation(fig, update, frames=tqdm(range(len(unique_times)), desc="Creating error animation"), interval=100, blit=False)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    anim.save(save_path, writer=PillowWriter(fps=10))
    plt.close(fig)

def run(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    BATCH_SIZE = 4096

    x_star, y_star, z_star, t_star, T_star_true = data_loader.get_evaluation_data(args)

    model = PINNsformer(
        d_out=args.d_out, d_hidden=args.d_hidden, d_model=args.d_model,
        N=args.n_layers, heads=args.n_heads
    ).to(device)
    model.load_state_dict(torch.load(args.model_save_path, map_location=device))
    model.eval()
    print(f"Model loaded from {args.model_save_path}")

    print(f"Starting evaluation with batch size {BATCH_SIZE}...")
    num_samples = x_star.shape[0]
    T_pred_list = []
    
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, BATCH_SIZE), desc="Evaluating"):
            start_idx = i
            end_idx = min(i + BATCH_SIZE, num_samples)
            
            x_batch = x_star[start_idx:end_idx].unsqueeze(1)
            y_batch = y_star[start_idx:end_idx].unsqueeze(1)
            z_batch = z_star[start_idx:end_idx].unsqueeze(1)
            t_batch = t_star[start_idx:end_idx].unsqueeze(1)
            
            pred_batch_physical = model(x_batch, y_batch, z_batch, t_batch).squeeze(1).cpu().numpy()
            
            T_pred_list.append(pred_batch_physical)

    T_pred = np.concatenate(T_pred_list, axis=0).flatten()

    print("Prediction complete. De-normalization is not needed.")

    T_star_true_np = T_star_true.flatten()
    
    rl2_error = np.linalg.norm(T_star_true_np - T_pred) / np.linalg.norm(T_star_true_np)
    
    print(f'\nRelative L2 error: {rl2_error:.6f}')

    print("Generating comparison animation...")
    _create_animation(args.data_path, T_pred, args.animation_save_path)
    print(f"Animation saved to {args.animation_save_path}")

    print("\nGenerating error analysis plots...")
    results_dir = os.path.dirname(args.animation_save_path)
    
    _create_error_plots(T_star_true_np, T_pred, results_dir)

    error_animation_path = args.animation_save_path.replace('.gif', '_error.gif')
    _create_error_animation(args.data_path, T_pred, error_animation_path)
    print(f"Error animation saved to {error_animation_path}")