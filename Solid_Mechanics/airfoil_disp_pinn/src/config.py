import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Airfoil displacement PINN standardized runner")
    parser.add_argument("--mode", type=str, default="quick", choices=["quick", "train"])
    parser.add_argument("--data", type=str, default="simul", choices=["simul", "real"])
    parser.add_argument("--data_path", type=str, default="data/data.mat")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--model_dir", type=str, default="model")
    parser.add_argument("--result_dir", type=str, default="results")
    parser.add_argument("--train_epochs", type=int, default=3000)
    parser.add_argument("--quick_epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

