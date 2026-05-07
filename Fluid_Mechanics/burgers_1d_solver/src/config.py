import argparse
from dataclasses import dataclass


@dataclass
class RunConfig:
    mode: str
    data: str
    data_path: str
    device: str
    seed: int
    output_dir: str
    model_dir: str
    train_final: bool
    nc: int
    n_test: int
    bc_pts: int
    ic_pts: int
    epochs_pinn: int
    epochs_gpt_train: int
    epochs_gpt_test: int
    lr_pinn: float
    lr_gpt: float
    tol: float
    number_of_neurons: int
    test_cases: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="1D Burgers GPT-PINN runner")
    parser.add_argument("--mode", choices=["train", "quick"], default="quick")
    parser.add_argument("--data", choices=["simul", "real"], default="simul")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/real_nu.npy",
        help="Path to viscosity candidates (.npy). Used when --data real",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--model_dir", type=str, default="model")
    parser.add_argument("--train_final", action="store_true")
    return parser


def resolve_device(device_arg: str):
    import torch

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_arg)
    return torch.device("cpu")


def get_config(args) -> RunConfig:
    if args.mode == "train":
        return RunConfig(
            mode=args.mode,
            data=args.data,
            data_path=args.data_path,
            device=args.device,
            seed=args.seed,
            output_dir=args.output_dir,
            model_dir=args.model_dir,
            train_final=args.train_final,
            nc=100,
            n_test=100,
            bc_pts=200,
            ic_pts=200,
            epochs_pinn=60000,
            epochs_gpt_train=2000,
            epochs_gpt_test=5000,
            lr_pinn=0.005,
            lr_gpt=0.02,
            tol=2e-5,
            number_of_neurons=9,
            test_cases=25,
        )

    return RunConfig(
        mode=args.mode,
        data=args.data,
        data_path=args.data_path,
        device=args.device,
        seed=args.seed,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        train_final=False,
        nc=40,
        n_test=40,
        bc_pts=80,
        ic_pts=80,
        epochs_pinn=300,
        epochs_gpt_train=80,
        epochs_gpt_test=200,
        lr_pinn=0.005,
        lr_gpt=0.02,
        tol=1e-4,
        number_of_neurons=2,
        test_cases=3,
    )
