import argparse
import os
import random

import numpy as np
import torch

from src.runner import run_quick, run_train


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve(base_dir: str, p: str) -> str:
    if os.path.isabs(p):
        return p
    return os.path.join(base_dir, p)


def main():
    parser = argparse.ArgumentParser(description="Vlasov Equation PINN")
    parser.add_argument("--mode", choices=["train", "quick"], default="quick")
    parser.add_argument("--data", choices=["simul", "real"], default="simul")
    parser.add_argument("--data_path", type=str, default="data", help="Data directory for real mode")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--quick_epochs", type=int, default=200)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=20000)
    parser.add_argument("--n_u", type=int, default=70000)

    parser.add_argument("--model_dir", type=str, default="model")
    parser.add_argument("--results_dir", type=str, default="results")

    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA unavailable, fallback to CPU")
        args.device = "cpu"
    args.device = torch.device(args.device)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    args.data_dir = resolve(base_dir, args.data_path)
    args.model_dir = resolve(base_dir, args.model_dir)
    args.results_dir = resolve(base_dir, args.results_dir)

    args.layers = [3, 50, 50, 50, 50, 50, 50, 50, 50, 2]

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    set_seed(args.seed)

    print("=" * 60)
    print(f"mode: {args.mode}")
    print(f"data: {args.data}")
    print(f"data_dir: {args.data_dir}")
    print(f"device: {args.device}")
    print(f"model_dir: {args.model_dir}")
    print(f"results_dir: {args.results_dir}")
    print("=" * 60)

    if args.mode == "train":
        run_train(args)
    else:
        run_quick(args)


if __name__ == "__main__":
    main()