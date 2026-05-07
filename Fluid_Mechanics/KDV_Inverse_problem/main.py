import os

import numpy as np

from src.config import parse_args
from src.data_utils import load_user_data, validate_user_data
from src.runner import run_pinn_training_mode, run_quick_inference


def _prepare_user_data(args):
    if args.data == "real":
        if not os.path.exists(args.data_path):
            print(f"[warn] data_path not found: {args.data_path}, fallback to simul")
            return None
        data = load_user_data(args.data_path)
        if isinstance(data, np.ndarray) and validate_user_data(data):
            return {"X": data[:, :2], "Y": data[:, 2:3]}
        print("[warn] invalid real data format, fallback to simul")
    return None


def main():
    args = parse_args()
    print("=" * 60)
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("=" * 60)

    user_data = _prepare_user_data(args)

    if args.mode == "train":
        result = run_pinn_training_mode(
            epochs=args.epochs,
            lr=args.lr,
            user_data=user_data,
            model_dir=args.model_dir,
            result_dir=args.result_dir,
        )
        print("[train] finished:", result)
    elif args.mode == "quick":
        result = run_quick_inference(
            model_dir=args.model_dir,
            result_dir=args.result_dir,
            user_data=user_data,
        )
        if result is None:
            print("[quick] no model found, run train first")
        else:
            print("[quick] finished")


if __name__ == "__main__":
    main()

