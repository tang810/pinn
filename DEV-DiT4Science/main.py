import os

from src.config import parse_args
from src.runner import run_quick, run_train


def main() -> None:
    args = parse_args()

    if not args.data_path:
        if args.data == "simul":
            args.data_path = os.path.join("data", "simul_airfoil_flow.npz")
        elif args.data == "real":
            args.data_path = os.path.join("data", "real_airfoil_flow.npz")

    print("=" * 60)
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("=" * 60)

    if args.mode == "train":
        run_train(args)
    elif args.mode == "quick":
        run_quick(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
