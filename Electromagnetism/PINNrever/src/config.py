import argparse
import os


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PINNrever unified runner")
    parser.add_argument("--mode", choices=["train", "quick"], default="quick")
    parser.add_argument(
        "--data",
        choices=["simul", "real"],
        default="simul",
        help="Data source flag",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/real_points.npy",
        help="Path to real point data (.npy, shape N x 2). Used when --data real",
    )
    parser.add_argument("--task", choices=["scalar", "recover"], default="recover")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--num_points", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=50)
    parser.add_argument("--num_layers", type=int, default=3)

    parser.add_argument("--train_epochs", type=int, default=5000)
    parser.add_argument("--quick_epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_every", type=int, default=100)

    parser.add_argument("--lambda_pde", type=float, default=1.1)
    parser.add_argument("--lambda_bc", type=float, default=1.0)
    parser.add_argument("--lambda_data", type=float, default=1.0)

    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--model_dir", type=str, default="model")

    return parser


def resolve_device(device_arg: str):
    import torch

    if device_arg.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_arg)
    return torch.device("cpu")


def ensure_dirs(output_dir: str, model_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)


def resolve_data_path(data_path: str) -> str:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    path = os.path.expanduser(data_path)
    if os.path.isabs(path):
        return path
    return os.path.join(project_root, path)
