import os
import random

import numpy as np
import torch

from src.config import build_parser, get_config, resolve_device
from src.runner import run


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = build_parser().parse_args()
    cfg = get_config(args)
    device = resolve_device(args.device)
    project_root = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(cfg.output_dir):
        cfg.output_dir = os.path.join(project_root, cfg.output_dir)
    if not os.path.isabs(cfg.model_dir):
        cfg.model_dir = os.path.join(project_root, cfg.model_dir)
    if not os.path.isabs(cfg.data_path):
        cfg.data_path = os.path.join(project_root, cfg.data_path)

    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.model_dir, exist_ok=True)

    set_seed(cfg.seed)

    print("=" * 60)
    print(f"mode: {cfg.mode}")
    print(f"data: {cfg.data}")
    print(f"data_path: {cfg.data_path}")
    print(f"device: {device}")
    print(f"seed: {cfg.seed}")
    print(f"output_dir: {cfg.output_dir}")
    print(f"model_dir: {cfg.model_dir}")
    print("=" * 60)

    run(cfg, device)


if __name__ == "__main__":
    main()
