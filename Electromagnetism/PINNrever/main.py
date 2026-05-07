import os
import random

import numpy as np
import torch

from src.config import build_parser, resolve_device, ensure_dirs, resolve_data_path
from src.data_utils import generate_points_from_source
from src.models import PINN, CoefNet
from src.trainer import train_scalar, train_recover
from src.inference import run_quick_scalar, run_quick_recover


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_models(args, device):
    model = PINN(2, args.hidden_dim, 1, args.num_layers).to(device)
    coef_net = CoefNet(2, args.hidden_dim, 1, args.num_layers).to(device)
    tau = torch.tensor(0.1, device=device, requires_grad=True)
    return model, coef_net, tau


def print_args(args):
    print("=" * 60)
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("=" * 60)


def main():
    parser = build_parser()
    args = parser.parse_args()

    args.data_path = resolve_data_path(args.data_path)
    ensure_dirs(args.output_dir, args.model_dir)
    device = resolve_device(args.device)
    set_seed(args.seed)
    print_args(args)
    print(f"Using device: {device}")

    points = generate_points_from_source(args.data, args.num_points, device, args.data_path)
    model, coef_net, tau = build_models(args, device)

    if args.mode == "train":
        if args.task == "scalar":
            train_scalar(model, tau, points, args)
            print(f"[train] saved model to {os.path.join(args.model_dir, 'scalar_model.pt')}")
        else:
            train_recover(model, coef_net, points, args)
            print(f"[train] saved model to {os.path.join(args.model_dir, 'recover_model.pt')}")
        return

    if args.task == "scalar":
        run_quick_scalar(model, tau, points, args)
        print(f"[quick] result saved to {os.path.join(args.output_dir, 'quick_scalar.png')}")
    else:
        run_quick_recover(model, coef_net, points, args)
        print(f"[quick] result saved to {os.path.join(args.output_dir, 'quick_recover.png')}")


if __name__ == "__main__":
    main()
