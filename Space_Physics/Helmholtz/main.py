import argparse

import os

from src.runner import run_quick, run_train


def parse_args():
    parser = argparse.ArgumentParser(description="Helmholtz PINN case")
    parser.add_argument("--mode", choices=["train", "quick"], default="quick")
    parser.add_argument("--data", choices=["simul", "load"], default="simul")
    parser.add_argument("--data_path", type=str, default="data/simul_placeholder.txt")
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--k", type=float, default=3.0)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_domain", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--quick_epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log_every", type=int, default=200)

    parser.add_argument("--model_dir", type=str, default="model")
    parser.add_argument("--result_dir", type=str, default="results")
    parser.add_argument("--model_name", type=str, default="helmholtz_pinn.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.data_path):
        args.data_path = os.path.join(base_dir, args.data_path)
    if not os.path.isabs(args.model_dir):
        args.model_dir = os.path.join(base_dir, args.model_dir)
    if not os.path.isabs(args.result_dir):
        args.result_dir = os.path.join(base_dir, args.result_dir)

    print("=" * 60)
    print(f"mode: {args.mode}")
    print(f"data: {args.data}")
    print(f"data_path: {args.data_path}")
    print(f"model_dir: {args.model_dir}")
    print(f"result_dir: {args.result_dir}")
    print("=" * 60)

    if args.data == "load":
        print("[info] load mode placeholder: this case currently uses simulation workflow.")

    if args.mode == "train":
        run_train(args)
    elif args.mode == "quick":
        run_quick(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
