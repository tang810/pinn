import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="KDV inverse problem standardized runner")
    parser.add_argument("--mode", type=str, default="quick", choices=["quick", "train"])
    parser.add_argument("--data", type=str, default="simul", choices=["simul", "real"])
    parser.add_argument("--data_path", type=str, default="data/real_data.npz")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model_dir", type=str, default="model")
    parser.add_argument("--result_dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()

