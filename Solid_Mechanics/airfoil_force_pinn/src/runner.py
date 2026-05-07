import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .SM_data import DataLoader
from .pinn_solver import PysicsInformedNeuralNetwork


def _pick_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_pinn(net_params: Optional[str], model_dir: str, result_dir: str) -> PysicsInformedNeuralNetwork:
    E = 100
    nu = 0.3
    N_HLayer = 6
    return PysicsInformedNeuralNetwork(
        E=E,
        nu=nu,
        layers=N_HLayer,
        bc_weight=10,
        eq_weight=1,
        net_params=net_params,
        model_dir=model_dir,
        results_dir=result_dir,
    )


def _prepare_data(data_path: str):
    dataloader = DataLoader(path=".", N_f=10000, N_b=1000)
    boundary_data = dataloader.loading_boundary_data(data_path)
    training_data = dataloader.loading_training_data(data_path)
    eval_data = dataloader.loading_evaluate_data(data_path)
    return boundary_data, training_data, eval_data


def run_train(args) -> None:
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")

    pinn = _build_pinn(net_params=None, model_dir=args.model_dir, result_dir=args.result_dir)
    boundary_data, training_data, eval_data = _prepare_data(args.data_path)
    pinn.set_boundary_data(X=boundary_data)
    pinn.set_eq_training_data(X=training_data)
    x_star, y_star, z_star, u_star, v_star, w_star = eval_data

    pinn.train(num_epoch=args.train_epochs, lr=args.lr)
    pinn.test(x_star, y_star, z_star, u_star, v_star, w_star, loop=0, num_epoch=args.train_epochs)
    pinn.evaluate(x_star, y_star, z_star, u_star, v_star, w_star, plot_name="result_train.png")
    ckpt = "airfoil_disp_pinn_best.pt"
    pinn.save(ckpt, directory=args.model_dir)
    print(f"[train] model saved to {Path(args.model_dir) / ckpt}")


def run_quick(args) -> None:
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")

    ckpt_path = Path(args.model_dir) / "airfoil_disp_pinn_best.pt"
    net_params = str(ckpt_path) if ckpt_path.exists() else None

    pinn = _build_pinn(net_params=net_params, model_dir=args.model_dir, result_dir=args.result_dir)
    boundary_data, training_data, eval_data = _prepare_data(args.data_path)
    pinn.set_boundary_data(X=boundary_data)
    pinn.set_eq_training_data(X=training_data)
    x_star, y_star, z_star, u_star, v_star, w_star = eval_data

    if net_params is None:
        print("[quick] checkpoint not found, running short warmup training...")
        pinn.train(num_epoch=args.quick_epochs, lr=args.lr)
        pinn.save("airfoil_disp_pinn_best.pt", directory=args.model_dir)

    pinn.evaluate(x_star, y_star, z_star, u_star, v_star, w_star, plot_name="result_quick.png")
    print(f"[quick] result saved to {Path(args.result_dir) / 'result_quick.png'}")
