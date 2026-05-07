import argparse
import os

import numpy as np

from src.config import Paths, build_train_settings, build_quick_settings
from src.data_utils import load_data


def build_parser():
    parser = argparse.ArgumentParser(description="HeatTrans PINN runner")
    parser.add_argument("--mode", choices=["train", "quick"], default="quick")
    parser.add_argument("--data_path", type=str, default="data/data_7246.txt")
    parser.add_argument("--simulation", choices=["simul", "load"], default="simul")
    parser.add_argument("--result_dir", type=str, default="result")
    parser.add_argument("--model_dir", type=str, default="model")
    parser.add_argument("--run_full_matrix", action="store_true")
    parser.add_argument("--adam_iters", type=int, default=None)
    parser.add_argument("--no_lbfgs", action="store_true")
    return parser


def main(
    mode="quick",
    data_path="data/data_7246.txt",
    simulation="simul",
    result_dir="result",
    model_dir="model",
    run_full_matrix=False,
    adam_iters=None,
    use_lbfgs=True,
):
    try:
        import tensorflow as tf

        tf.compat.v1.disable_eager_execution()
    except ModuleNotFoundError:
        print("TensorFlow is required for HeatTrans.")
        print("Please install it first, for example:")
        print("  python -m pip install tensorflow==2.13.1")
        return

    from src.runner import run_all_cases

    settings = build_train_settings() if mode == "train" else build_quick_settings()
    settings.data_path = data_path
    if run_full_matrix:
        settings.run_full_matrix = True
    if adam_iters is not None:
        settings.adam_iters = adam_iters
    if not use_lbfgs:
        settings.use_lbfgs = False

    paths = Paths(model_dir=model_dir, result_dir=result_dir)
    os.makedirs(paths.model_dir, exist_ok=True)
    os.makedirs(paths.result_dir, exist_ok=True)

    print("=" * 60)
    print(f"mode: {mode}")
    print(f"data_path: {settings.data_path}")
    print(f"simulation: {simulation}")
    print(f"run_full_matrix: {settings.run_full_matrix}")
    print(f"adam_iters: {settings.adam_iters}")
    print(f"use_lbfgs: {settings.use_lbfgs}")
    print(f"result_dir: {paths.result_dir}")
    print(f"model_dir: {paths.model_dir}")
    print("=" * 60)

    raw_data = load_data(settings.data_path)
    results = run_all_cases(raw_data, settings, paths)

    np.save(os.path.join(paths.result_dir, f"{mode}_summary.npy"), np.array(results, dtype=object))
    print(f"Completed {len(results)} case(s).")


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(
        mode=args.mode,
        data_path=args.data_path,
        simulation=args.simulation,
        result_dir=args.result_dir,
        model_dir=args.model_dir,
        run_full_matrix=args.run_full_matrix,
        adam_iters=args.adam_iters,
        use_lbfgs=not args.no_lbfgs,
    )
