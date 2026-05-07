import argparse
import os
from types import SimpleNamespace

import torch

from src import config
from src import evaluate
from src import train


def _build_runtime_args(mode: str, data: str) -> SimpleNamespace:
    cfg = {
        k: v
        for k, v in vars(config).items()
        if not k.startswith("_") and not callable(v)
    }
    args = SimpleNamespace(**cfg)
    args.mode = mode

    # Reserved data shortcut interface.
    if data == "simul":
        sim_path = os.path.join("data", "simul_data.txt")
        args.data_path = sim_path if os.path.exists(sim_path) else config.data_path
    elif data:
        args.data_path = data

    return args


def main(mode: str, data: str) -> None:
    args = _build_runtime_args(mode=mode, data=data)

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    print("=" * 60)
    print("Arguments:")
    print(f"mode: {args.mode}")
    print(f"data_path: {args.data_path}")
    print("=" * 60)

    if args.mode == "train":
        train.run(args)
    elif args.mode in ("eval", "quick"):
        evaluate.run(args)
    else:
        raise ValueError("Invalid mode. Choose one of: train, eval, quick")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval", "quick"],
        help="Execution mode",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="",
        help="Shortcut or path for data (e.g., simul)",
    )
    parsed = parser.parse_args()
    main(mode=parsed.mode, data=parsed.data)
