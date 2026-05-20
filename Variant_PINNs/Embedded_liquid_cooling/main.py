import argparse

from src.runner import run


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Embedded liquid cooling PINN solver")
    parser.add_argument("--mode", default="quick", choices=["quick", "train"])
    parser.add_argument("--data", default="simul", choices=["simul"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--n-points", type=int, default=None)
    return parser


def main():
    args = build_arg_parser().parse_args()
    run(
        mode=args.mode,
        data=args.data,
        epochs=args.epochs,
        n_points=args.n_points,
    )


if __name__ == "__main__":
    main()
