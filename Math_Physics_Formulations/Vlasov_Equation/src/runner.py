import os

import numpy as np
import torch

from .data_utils import (
    build_prediction_grid,
    build_training_set,
    generate_sim_data,
    load_real_data,
    save_data_info,
)
from .model import PhysicsInformedNN
from .plotting import plot_velocity_slice, save_metrics


def _load_data(data_mode, data_dir, output_dir):
    if data_mode == "real":
        try:
            data = load_real_data(data_dir)
            save_data_info(data, os.path.join(output_dir, "data_info_real.json"), "real")
            return data
        except Exception as e:
            print(f"[warn] failed to load real data: {e}. fallback to simul.")

    data = generate_sim_data()
    save_data_info(data, os.path.join(output_dir, "data_info_simul.json"), "simul")
    return data


def run_train(args):
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    data = _load_data(args.data, args.data_dir, args.results_dir)
    X_train, E_train, f_train = build_training_set(data, n_u=args.n_u, seed=args.seed)

    model = PhysicsInformedNN(X_train, E_train, f_train, args.layers, args.device)
    model.train(args.epochs, model_dir=args.model_dir, log_every=args.log_every, save_every=args.save_every)


def run_quick(args):
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    data = _load_data(args.data, args.data_dir, args.results_dir)
    X_train, E_train, f_train = build_training_set(data, n_u=min(args.n_u, 8000), seed=args.seed)

    model = PhysicsInformedNN(X_train, E_train, f_train, args.layers, args.device)
    model_path = os.path.join(args.model_dir, "PINN_final.pth")

    if os.path.exists(model_path):
        model.load_model(model_path)
        print(f"[quick] loaded model: {model_path}")
    else:
        print("[quick] model missing, running short warmup training...")
        model.train(args.quick_epochs, model_dir=args.model_dir, log_every=50, save_every=max(args.quick_epochs, 1))

    T, X, V, T_star, X_star, V_star = build_prediction_grid(data, args.device)
    f_pred, _ = model.predict(T_star, X_star, V_star)
    f_pred = np.reshape(f_pred, data["f"].shape)

    t_idx = data["f"].shape[0] // 2
    x_idx = data["f"].shape[1] // 2
    truth = data["f"][t_idx, x_idx, :]
    pred = f_pred[t_idx, x_idx, :]
    v_axis = data["v"]

    fig_path = os.path.join(args.results_dir, "quick_velocity_distribution.png")
    metrics_path = os.path.join(args.results_dir, "quick_metrics.txt")

    plot_velocity_slice(v_axis, truth, pred, fig_path)
    save_metrics(truth, pred, metrics_path)
    print(f"[quick] figure saved: {fig_path}")
    print(f"[quick] metrics saved: {metrics_path}")