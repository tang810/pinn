import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DEV-DiT4Science standardized runner")
    parser.add_argument("--mode", type=str, default="quick", choices=["quick", "train"])
    parser.add_argument("--data", type=str, default="simul", choices=["simul", "real"], help="data source tag")
    parser.add_argument("--data-path", type=str, default="", help="optional explicit dataset path")
    parser.add_argument("--epochs", type=int, default=300, help="training epochs for train mode")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for toy surrogate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--result-dir", type=str, default="results")
    parser.add_argument("--model-dir", type=str, default="model")
    return parser.parse_args()
