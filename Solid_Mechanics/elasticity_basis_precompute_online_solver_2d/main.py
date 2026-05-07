import argparse
import os
import random

import numpy as np
import torch

from src.data_utils import save_sample_real_data
from src.runner import quick_mode, train_mode


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve(base, p):
    if os.path.isabs(p):
        return p
    return os.path.join(base, p)


def main():
    parser = argparse.ArgumentParser(description="2D Elasticity Basis Precompute + Online Solver")
    parser.add_argument("--mode", choices=["train", "quick"], default="quick")
    parser.add_argument("--data", choices=["simul", "real"], default="simul")
    parser.add_argument("--data_path", type=str, default="data/real_data.npz")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--quick_epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--n_samples", type=int, default=4096)

    parser.add_argument("--model_dir", type=str, default="model")
    parser.add_argument("--results_dir", type=str, default="results")

    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA unavailable, fallback to cpu")
        args.device = "cpu"
    args.device = torch.device(args.device)

    base = os.path.dirname(os.path.abspath(__file__))
    args.data_path = resolve(base, args.data_path)
    args.model_dir = resolve(base, args.model_dir)
    args.results_dir = resolve(base, args.results_dir)

    os.makedirs(os.path.dirname(args.data_path), exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    if not os.path.exists(args.data_path):
        save_sample_real_data(args.data_path)

    set_seed(args.seed)

    print("=" * 60)
    print(f"mode: {args.mode}")
    print(f"data: {args.data}")
    print(f"data_path: {args.data_path}")
    print(f"device: {args.device}")
    print(f"model_dir: {args.model_dir}")
    print(f"results_dir: {args.results_dir}")
    print("=" * 60)

    if args.mode == "train":
        train_mode(args)
    else:
        quick_mode(args)


if __name__ == "__main__":
    main()