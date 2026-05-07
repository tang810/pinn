import os
import time
import numpy as np
import tensorflow as tf

from .data_utils import build_sensor_training_data, build_bc_ic_points
from .model import PhysicsInformedNN
from .plotting import plot_results, save_matrix_csv


tf1 = tf.compat.v1


def _summary_text(settings, raw_data, n_sensors, n_time_steps, X_u_train, sensor_xyz, selected_times,
                  ax_val, aperp_val, err_ax, err_aperp, error_u,
                  ax_val_noisy, aperp_val_noisy, err_ax_noisy, err_aperp_noisy, error_u_noisy):
    sensor_lines = "\n".join([
        f"  Sensor {i+1:02d}: x={xyz[0]:.8f}, y={xyz[1]:.8f}, z={xyz[2]:.8f}" for i, xyz in enumerate(sensor_xyz)
    ])

    return (
        "\n=======================================================\n"
        "          FINAL RESULTS SUMMARY - SURFACE + BC         \n"
        "=======================================================\n"
        f"Data file                  : {settings.data_path}\n"
        f"Original total rows        : {raw_data.shape[0]}\n"
        f"Sensor count               : {n_sensors}\n"
        f"Time steps per sensor      : {n_time_steps}\n"
        f"Observation points total   : {X_u_train.shape[0]}\n"
        f"Collocation points N_f     : {settings.n_f}\n"
        f"BC points per face         : {settings.n_bc_per_face}\n"
        f"IC points                  : {settings.n_ic}\n"
        "-------------------------------------------------------\n"
        "Sensor coordinates:\n"
        f"{sensor_lines}\n"
        "-------------------------------------------------------\n"
        f"Selected time range        : {selected_times[0]:.4f} to {selected_times[-1]:.4f}\n"
        "-------------------------------------------------------\n"
        "[Noiseless Data]\n"
        f"  Identified Alpha_x       : {ax_val:.8e} (Error: {err_ax:.4f}%)\n"
        f"  Identified Alpha_perp    : {aperp_val:.8e} (Error: {err_aperp:.4f}%)\n"
        f"  Temperature Error (L2)   : {error_u:.5e}\n"
        "-------------------------------------------------------\n"
        f"[{settings.noise_level*100:.0f}% Noisy Data]\n"
        f"  Identified Alpha_x       : {ax_val_noisy:.8e} (Error: {err_ax_noisy:.4f}%)\n"
        f"  Identified Alpha_perp    : {aperp_val_noisy:.8e} (Error: {err_aperp_noisy:.4f}%)\n"
        f"  Temperature Error (L2)   : {error_u_noisy:.5e}\n"
        "=======================================================\n"
    )


def run_experiment(raw_data, n_sensors, n_time_steps, case_tag, settings, paths):
    case_seed = settings.random_seed + n_sensors * 1000 + n_time_steps
    np.random.seed(case_seed)
    tf1.set_random_seed(case_seed)

    X_star = raw_data[:, 0:4].astype(np.float32)
    u_star = raw_data[:, 4:5].astype(np.float32)

    u_min, u_max = np.min(u_star), np.max(u_star)
    u_scale = u_max - u_min
    lb = X_star.min(0).astype(np.float32)
    ub = X_star.max(0).astype(np.float32)

    X_u_train, u_sensor, sensor_xyz, selected_times = build_sensor_training_data(
        raw_data,
        n_sensors=n_sensors,
        n_time_steps=n_time_steps,
        surface_tol=settings.surface_tol,
        seed=settings.random_seed,
    )

    u_train = ((u_sensor - u_min) / u_scale).astype(np.float32)

    rng_f = np.random.RandomState(case_seed)
    X_f_train = (lb + (ub - lb) * rng_f.rand(settings.n_f, 4)).astype(np.float32)
    X_f_train = np.vstack((X_f_train, X_u_train)).astype(np.float32)

    X_bc, X_ic = build_bc_ic_points(
        raw_data, lb, ub,
        n_bc_per_face=settings.n_bc_per_face,
        n_ic=settings.n_ic,
        surface_tol=settings.surface_tol,
        seed=case_seed,
    )

    model = PhysicsInformedNN(X_u_train, u_train, X_f_train, X_bc, X_ic, settings, lb, ub, u_min, u_scale)
    model.train(settings.adam_iters, use_lbfgs=settings.use_lbfgs)
    ax_val, aperp_val = model.get_alphas()

    err_ax = float(np.abs(ax_val - settings.true_alpha_x) / settings.true_alpha_x * 100)
    err_aperp = float(np.abs(aperp_val - settings.true_alpha_perp) / settings.true_alpha_perp * 100)

    u_pred_norm, _ = model.predict(X_star.astype(np.float32))
    u_pred = u_pred_norm * u_scale + u_min
    error_u = float(np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2))

    u_train_noisy = (
        u_train + settings.noise_level * np.std(u_train) * np.random.randn(u_train.shape[0], u_train.shape[1])
    ).astype(np.float32)

    model_noisy = PhysicsInformedNN(X_u_train, u_train_noisy, X_f_train, X_bc, X_ic, settings, lb, ub, u_min, u_scale)
    model_noisy.train(settings.adam_iters, use_lbfgs=settings.use_lbfgs)
    ax_noisy, aperp_noisy = model_noisy.get_alphas()

    err_ax_noisy = float(np.abs(ax_noisy - settings.true_alpha_x) / settings.true_alpha_x * 100)
    err_aperp_noisy = float(np.abs(aperp_noisy - settings.true_alpha_perp) / settings.true_alpha_perp * 100)

    u_pred_noisy_norm, _ = model_noisy.predict(X_star.astype(np.float32))
    u_pred_noisy = u_pred_noisy_norm * u_scale + u_min
    error_u_noisy = float(np.linalg.norm(u_star - u_pred_noisy, 2) / np.linalg.norm(u_star, 2))

    os.makedirs(paths.result_dir, exist_ok=True)
    os.makedirs(paths.model_dir, exist_ok=True)

    plot_results(X_star, u_star, u_pred, case_tag, out_dir=paths.result_dir)

    summary = _summary_text(
        settings, raw_data, n_sensors, n_time_steps, X_u_train, sensor_xyz, selected_times,
        ax_val, aperp_val, err_ax, err_aperp, error_u,
        ax_noisy, aperp_noisy, err_ax_noisy, err_aperp_noisy, error_u_noisy,
    )

    summary_path = os.path.join(paths.result_dir, f"results_summary_{case_tag}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    np.savez(
        os.path.join(paths.model_dir, f"heattrans_{case_tag}.npz"),
        alpha_x=ax_val,
        alpha_perp=aperp_val,
        alpha_x_noisy=ax_noisy,
        alpha_perp_noisy=aperp_noisy,
    )

    return {
        "n_sensors": n_sensors,
        "n_time_steps": n_time_steps,
        "ax": ax_val,
        "aperp": aperp_val,
        "err_ax": err_ax,
        "err_aperp": err_aperp,
        "error_u": error_u,
        "ax_noisy": ax_noisy,
        "aperp_noisy": aperp_noisy,
        "err_ax_noisy": err_ax_noisy,
        "err_aperp_noisy": err_aperp_noisy,
        "error_u_noisy": error_u_noisy,
    }


def run_all_cases(raw_data, settings, paths):
    if settings.run_full_matrix:
        cases = [(s, t) for s in settings.sensor_counts for t in settings.time_step_counts]
    else:
        cases = [(settings.default_n_sensors, settings.default_n_time_steps)]

    all_results = []
    for n_sensors, n_times in cases:
        case_tag = f"surface_bc_sensor_{n_sensors}_time_{n_times}"
        start_time = time.time()
        row = run_experiment(raw_data, n_sensors, n_times, case_tag, settings, paths)
        row["elapsed_sec"] = time.time() - start_time
        all_results.append(row)

    if len(all_results) > 1:
        csv_path = os.path.join(paths.result_dir, "results_summary_surface_bc_matrix.csv")
        save_matrix_csv(all_results, csv_path)

    return all_results
