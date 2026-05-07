import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Mixed convection PINN wrapper")
    parser.add_argument("--mode", choices=["quick", "train"], default="quick")
    parser.add_argument("--data", choices=["simul", "load"], default="simul")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--variant", choices=["mcf", "ev_mcf"], default="mcf")
    parser.add_argument("--re", type=int, default=2000)
    parser.add_argument("--pr", type=float, default=0.71)
    parser.add_argument("--ri", type=float, default=0.1)
    parser.add_argument("--n_f", type=int, default=40000)
    parser.add_argument("--n_b", type=int, default=1000)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--hidden_size", type=int, default=120)
    parser.add_argument("--lam_bcs", type=float, default=10.0)
    parser.add_argument("--lam_equ", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--quick_epochs", type=int, default=200)
    parser.add_argument("--train_epochs", type=int, default=5000)
    parser.add_argument("--model_dir", type=str, default="model")
    parser.add_argument("--result_dir", type=str, default="results")
    return parser.parse_args()
