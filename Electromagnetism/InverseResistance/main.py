import argparse
import os

import matplotlib
import torch

from src.train import configure_paths, quick_2d, quick_3d, train_2d, train_3d

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
matplotlib.use("Agg")


def _pick_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="InverseResistance standardized runner")
    parser.add_argument("--mode", type=str, default="quick", choices=["quick", "train"])
    parser.add_argument("--data", type=str, default="simul", choices=["simul", "real"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--data_path", type=str, default="", help="optional data file path")
    parser.add_argument("--model_dir", type=str, default="model")
    parser.add_argument("--result_dir", type=str, default="results")
    parser.add_argument("--train_epochs_2d", type=int, default=3000)
    parser.add_argument("--train_epochs_3d", type=int, default=5000)
    args = parser.parse_args()

    data_dir = "data"
    if args.data_path:
        data_dir = os.path.dirname(args.data_path) or "data"
    configure_paths(model_dir=args.model_dir, result_dir=args.result_dir, data_dir=data_dir)
    device = _pick_device(args.device)

    print("=" * 60)
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print(f"resolved_device: {device}")
    print("=" * 60)

    if args.mode == "train":
        train_2d(device=device, data_type=args.data, epochs=args.train_epochs_2d)
        train_3d(device=device, data_type=args.data, epochs=args.train_epochs_3d)
    elif args.mode == "quick":
        quick_2d(device=device, data_type=args.data)
        quick_3d(device=device, data_type=args.data)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()

