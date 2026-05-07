import argparse
import os
import random

import numpy as np
import torch

from src.data_utils import (
    extract_data_parameters,
    generate_virtual_data,
    load_user_data,
    save_data_info,
    validate_data,
)
from src.runner import (
    build_dataloader,
    export_all_results,
    export_training_results,
    run_pinn_training,
    run_quick_inference,
)
from src.models import KdV_PINN
from src.utils import plot_results


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(base_dir: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)


def prepare_data(args, output_dir):
    if args.data == "real":
        if args.data_path:
            raw_path = resolve_path(os.path.dirname(os.path.abspath(__file__)), args.data_path)
            data = load_user_data(raw_path)
            if data and validate_data(data):
                n_samples, n_x = extract_data_parameters(data, args.n_samples, args.n_x)
                print(f"Using real data: {raw_path}")
                return data, "real", n_samples, n_x
            print("[warn] real data invalid. Falling back to simulated data.")
        else:
            print("[warn] --data real set without --data_path. Falling back to simulated data.")

    data = generate_virtual_data(n_samples=args.n_samples, n_x=args.n_x)
    save_data_info(data, os.path.join(output_dir, "data_info_simulation.json"), source="simulation")
    return data, "simulation", args.n_samples, args.n_x


def main():
    parser = argparse.ArgumentParser(description="KdV PINN Runner")
    parser.add_argument("--mode", type=str, default="quick", choices=["train", "quick"])
    parser.add_argument("--data", type=str, default="simul", choices=["simul", "real"])
    parser.add_argument("--data_path", type=str, default="", help="Path to real data (.csv/.npz)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_samples", type=int, default=80)
    parser.add_argument("--n_x", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--model_dir", type=str, default="model")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable, switching to CPU.")
        args.device = "cpu"

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = resolve_path(base_dir, args.output_dir)
    model_dir = resolve_path(base_dir, args.model_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    set_seed(42)

    print("=" * 60)
    print(f"mode: {args.mode}")
    print(f"data: {args.data}")
    print(f"data_path: {args.data_path}")
    print(f"device: {args.device}")
    print(f"output_dir: {output_dir}")
    print(f"model_dir: {model_dir}")
    print("=" * 60)

    data, data_source, _, _ = prepare_data(args, output_dir)
    if data_source == "real":
        save_data_info(data, os.path.join(output_dir, "data_info_real.json"), source="real")

    dataloader = build_dataloader(data, device=args.device, batch_size=args.batch_size)

    model = KdV_PINN(n_input=1, n_output=1, n_hidden=50, n_layers=3).to(args.device)

    if args.mode == "train":
        train_ret = run_pinn_training(
            model=model,
            dataloader=dataloader,
            device=args.device,
            model_dir=model_dir,
            epochs=args.epochs,
            lr=args.lr,
        )
        export_training_results(train_ret, output_dir=output_dir)
        pred = run_quick_inference(data, device=args.device, model_dir=model_dir, quick_epochs=0)
        plot_results(data, pred=pred, mode="train", output_dir=output_dir)
        print("train finished")
        return

    pred = run_quick_inference(data, device=args.device, model_dir=model_dir, quick_epochs=30)
    export_all_results(pred, output_dir=output_dir)
    plot_results(data, pred=pred, mode="quick", output_dir=output_dir)
    print("quick finished")


if __name__ == "__main__":
    main()