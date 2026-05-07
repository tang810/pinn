import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .model import PINNsformer

def _compute_von_mises_stress(T, E, alpha, nu, T0):
    """
    Simplified thermal stress -> von Mises approximation:
    sigma_th = E*alpha*(T - T0)/(1-2*nu)
    von_mises ~ |sigma_th|
    """
    sigma_th = E * alpha * (T - T0) / (1 - 2 * nu)
    return np.abs(sigma_th)

def _save_error_analysis_txt(output_dir, metrics: dict):
    path = os.path.join(output_dir, "error_analysis.txt")
    with open(path, "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    print(f"Error analysis saved to {path}")

def _plot_error_distributions(output_dir, abs_err, rel_err_pct):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    axs[0, 0].hist(abs_err, bins=50)
    axs[0, 0].set_title("Absolute Error Distribution")
    axs[0, 0].set_xlabel("Absolute Error (K)")
    axs[0, 0].set_ylabel("Frequency")

    axs[0, 1].hist(rel_err_pct, bins=50)
    axs[0, 1].set_title("Relative Error Distribution")
    axs[0, 1].set_xlabel("Relative Error (%)")
    axs[0, 1].set_ylabel("Frequency")

    axs[1, 0].scatter(np.arange(len(abs_err)), abs_err, s=5, c='red', alpha=0.5)
    axs[1, 0].set_title("Absolute Error vs Data Point")
    axs[1, 0].set_xlabel("Data Point Index")
    axs[1, 0].set_ylabel("Absolute Error (K)")

    sorted_idx = np.argsort(abs_err)
    axs[1, 1].scatter(abs_err[sorted_idx], sorted_idx, s=5, c='purple', alpha=0.5)
    axs[1, 1].set_title("Sorted Absolute Errors")
    axs[1, 1].set_xlabel("Absolute Error (K)")
    axs[1, 1].set_ylabel("Data Point Index")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "error_distribution.png")
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Error distribution plot saved to {save_path}")

def _plot_prediction_vs_truth(output_dir, T_true, T_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(T_true, T_pred, s=10, alpha=0.6, label="Predictions")
    minv = min(np.min(T_true), np.min(T_pred))
    maxv = max(np.max(T_true), np.max(T_pred))
    plt.plot([minv, maxv], [minv, maxv], 'r--', label="Perfect Prediction")
    plt.title("Prediction vs Ground Truth")
    plt.xlabel("Ground Truth Temperature (K)")
    plt.ylabel("Predicted Temperature (K)")
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(output_dir, "prediction_vs_truth.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Prediction vs truth plot saved to {save_path}")

def _plot_temperature_vs_stress(output_dir, T_pred, stress):
    plt.figure(figsize=(10, 6))
    plt.scatter(T_pred, stress, s=10, alpha=0.6)
    plt.title("Temperature-Stress Relationship (Physics-driven)")
    plt.xlabel("Temperature (K)")
    plt.ylabel("von Mises Stress (Pa)")
    plt.grid(True)
    save_path = os.path.join(output_dir, "temperature_vs_stress.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Temperature vs stress plot saved to {save_path}")

def _create_animation(output_dir, x, y, z, t, T_true, T_pred, time_points):
    # Calculate global limits
    vmin_true, vmax_true = float(np.min(T_true)), float(np.max(T_true))
    vmin_pred, vmax_pred = float(np.min(T_pred)), float(np.max(T_pred))

    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    zmin, zmax = float(np.min(z)), float(np.max(z))

    frames = []
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # Draw colorbars once
    sc_true_dummy = ax1.scatter(x[:1], y[:1], z[:1], c=T_true[:1], cmap='viridis', vmin=vmin_true, vmax=vmax_true, alpha=1.0)
    sc_pred_dummy = ax2.scatter(x[:1], y[:1], z[:1], c=T_pred[:1], cmap='viridis', vmin=vmin_pred, vmax=vmax_pred, alpha=1.0)
    fig.colorbar(sc_true_dummy, ax=ax1, shrink=0.72, pad=0.08, label='Temperature (K)')
    fig.colorbar(sc_pred_dummy, ax=ax2, shrink=0.72, pad=0.08, label='Temperature (K)')

    for tp in time_points:
        ax1.clear()
        ax2.clear()
        mask = np.isclose(t, tp)
        if not np.any(mask):
            continue

        ax1.set_title(f"Ground Truth, t={tp:.2f}")
        ax1.scatter(x[mask], y[mask], z[mask], c=T_true[mask], cmap='viridis', s=80, alpha=0.3, vmin=vmin_true, vmax=vmax_true)

        ax2.set_title(f"PINNs Prediction, t={tp:.2f}")
        ax2.scatter(x[mask], y[mask], z[mask], c=T_pred[mask], cmap='viridis', s=80, alpha=0.3, vmin=vmin_pred, vmax=vmax_pred)

        ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
        ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")

        ax1.set_xlim(xmin, xmax); ax1.set_ylim(ymin, ymax); ax1.set_zlim(zmin, zmax)
        ax2.set_xlim(xmin, xmax); ax2.set_ylim(ymin, ymax); ax2.set_zlim(zmin, zmax)

        plt.subplots_adjust(left=0.05, right=0.95, wspace=0.3)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        frames.append(frame)

    if len(frames) == 0:
        plt.close(fig)
        raise RuntimeError("No valid frames collected for animation.gif")

    save_path = os.path.join(output_dir, "animation.gif")
    imageio.mimsave(save_path, frames, fps=2)
    plt.close(fig)
    print(f"Temperature animation saved to {save_path}")

def _create_stress_animation(output_dir, x, y, z, t, stress, time_points):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    frames = []

    vmin = float(np.min(stress))
    vmax = float(np.max(stress))
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    zmin, zmax = float(np.min(z)), float(np.max(z))

    # Draw colorbar once
    sc_dummy = ax.scatter(x[:1], y[:1], z[:1], c=stress[:1], cmap='inferno', vmin=vmin, vmax=vmax, alpha=1.0)
    fig.colorbar(sc_dummy, ax=ax, shrink=0.72, pad=0.08, label='von Mises Stress (Pa)')

    for tp in time_points:
        ax.clear()
        mask = np.isclose(t, tp)
        if not np.any(mask):
            continue

        ax.set_title(f"Thermal Stress, t={tp:.2f}")
        ax.scatter(x[mask], y[mask], z[mask], c=stress[mask], cmap='inferno', s=100, alpha=0.4, vmin=vmin, vmax=vmax)

        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_zlim(zmin, zmax)
        fig.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        frames.append(frame)

    if len(frames) == 0:
        plt.close(fig)
        raise RuntimeError("No valid frames collected for animation_stress.gif")

    save_path = os.path.join(output_dir, "animation_stress.gif")
    imageio.mimsave(save_path, frames, fps=2)
    plt.close(fig)
    print(f"Stress animation saved to {save_path}")

def run(args):
    _dev = getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu")
    if "cuda" in _dev and not torch.cuda.is_available():
        _dev = "cpu"
    device = torch.device(_dev)
    print(f"Using device: {device}")

    # Load evaluation data
    data = pd.read_csv(args.data_path, delimiter=' ', header=None)
    x = data.iloc[:, 0].values
    y = data.iloc[:, 1].values
    z = data.iloc[:, 2].values
    t = data.iloc[:, 3].values
    T_true = data.iloc[:, 4].values

    # Prepare input tensor
    # ------------------ Normalization ------------------
    x_in, y_in, z_in, t_in = x.copy(), y.copy(), z.copy(), t.copy()
    if getattr(args, "input_norm", "none") == "box01":
        bp1 = np.array(getattr(args, 'box_p1', [-1,-1,-1]))
        bp2 = np.array(getattr(args, 'box_p2', [1,1,1]))
        x_in = np.clip((x_in - bp1[0]) / (bp2[0] - bp1[0] + 1e-12), 0, 1)
        y_in = np.clip((y_in - bp1[1]) / (bp2[1] - bp1[1] + 1e-12), 0, 1)
        z_in = np.clip((z_in - bp1[2]) / (bp2[2] - bp1[2] + 1e-12), 0, 1)

    elif getattr(args, "input_norm", "none") == "01tobox":
        bp1 = np.array(getattr(args, 'box_p1', [-1,-1,-1]))
        bp2 = np.array(getattr(args, 'box_p2', [1,1,1]))
        x_in = x_in * (bp2[0] - bp1[0]) + bp1[0]
        y_in = y_in * (bp2[1] - bp1[1]) + bp1[1]
        z_in = z_in * (bp2[2] - bp1[2]) + bp1[2]

    if getattr(args, "t_norm", "none") == "minmax01":
        tmin = getattr(args, 'tmin', None)
        tmax = getattr(args, 'tmax', None)
        if tmin is None: tmin = float(np.min(t_in))
        if tmax is None: tmax = float(np.max(t_in))
        t_in = np.clip((t_in - tmin) / (tmax - tmin + 1e-12), 0, 1)
    # ---------------------------------------------------

    inputs = torch.tensor(np.stack([x_in, y_in, z_in, t_in], axis=1), dtype=torch.float32).to(device)
    inputs = inputs.unsqueeze(1).repeat(1, args.time_steps, 1)

    # Add step_size to the time dimension for subsequent sequence steps
    for i in range(args.time_steps):
        inputs[:, i, 3] = inputs[:, i, 3] + i * args.step_size

    # Load model using the correct hidden dims
    model = PINNsformer(d_model=64, d_hidden=512, N=args.N, heads=args.heads).to(device)
    model_path = getattr(args, "ckpt", "model/pinnsformer_withsun.pt")
    fallback_model_path = "model/pinnsformer_withsun.pt"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except RuntimeError as e:
        if model_path != fallback_model_path and os.path.exists(fallback_model_path):
            print(f"[warn] failed to load ckpt={model_path}: {e}")
            print(f"[info] retry with fallback ckpt={fallback_model_path}")
            model.load_state_dict(torch.load(fallback_model_path, map_location=device))
            model_path = fallback_model_path
        else:
            raise
    model.eval()

    # Predict（分批推理，避免大数据集 OOM）
    BATCH = 4096
    results = []
    with torch.no_grad():
        for i in range(0, inputs.shape[0], BATCH):
            batch = inputs[i:i+BATCH].to(device)
            out = model(batch)[:, 0, 0].cpu()
            results.append(out)
    T_pred = torch.cat(results, dim=0).numpy()

    # Denorm / Calibrate
    Tmin, Tmax = float(np.min(T_true)), float(np.max(T_true))
    if getattr(args, 'pred_denorm', False):
        T_pred = T_pred * (Tmax - Tmin) + Tmin
    
    if getattr(args, 'pred_calib', 'none') in ('global', 'per_frame'):
        vx = float(np.var(T_pred)) + 1e-12
        a = float(np.cov(T_pred, T_true, bias=True)[0, 1] / vx)
        b = float(np.mean(T_true) - a * np.mean(T_pred))
        T_pred = a * T_pred + b

    # Compute stress
    stress = _compute_von_mises_stress(T_pred, args.E, args.alpha, args.nu, args.T0)

    # Error metrics
    abs_err = np.abs(T_true - T_pred)
    rel_err_pct = abs_err / (np.abs(T_true) + 1e-8) * 100.0

    mae = mean_absolute_error(T_true, T_pred)
    rmse = np.sqrt(mean_squared_error(T_true, T_pred))
    max_abs_err = np.max(abs_err)
    mape = np.mean(rel_err_pct)
    r2 = r2_score(T_true, T_pred)

    # Relative L2 error
    rel_l2 = np.linalg.norm(T_pred - T_true) / (np.linalg.norm(T_true) + 1e-12)

    metrics = {
        "MAE (K)": mae,
        "RMSE (K)": rmse,
        "Max Absolute Error (K)": max_abs_err,
        "MAPE (%)": mape,
        "R2 Score": r2,
        "Relative L2 Error": rel_l2,
        "T_true_min": float(np.min(T_true)),
        "T_true_max": float(np.max(T_true)),
        "T_pred_min": float(np.min(T_pred)),
        "T_pred_max": float(np.max(T_pred)),
    }

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    _save_error_analysis_txt(args.output_dir, metrics)
    _plot_error_distributions(args.output_dir, abs_err, rel_err_pct)
    _plot_prediction_vs_truth(args.output_dir, T_true, T_pred)
    _plot_temperature_vs_stress(args.output_dir, T_pred, stress)

    # Animations
    time_points = np.unique(t)
    _create_animation(args.output_dir, x, y, z, t, T_true, T_pred, time_points)
    _create_stress_animation(args.output_dir, x, y, z, t, stress, time_points)

    print("=" * 50)
    print("Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print("=" * 50)
