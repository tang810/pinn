import argparse

from src.runner import run


def build_arg_parser():
    parser = argparse.ArgumentParser(description="3D packaging liquid cooling PINN")
    parser.add_argument("--mode", default="quick", choices=["quick", "train"])
    parser.add_argument("--data", default="simul", choices=["simul"])
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--show", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    print("=" * 50)
    print("3D Packaging Liquid Cooling PINN Solver")
    print(f"Mode: {args.mode}")
    print(f"Data: {args.data}")
    print("=" * 50)
    run(mode=args.mode, data=args.data, data_path=args.data_path, show=args.show)
    print("Done.")


if __name__ == "__main__":
    main()
