from src.config import parse_args
from src.runner import run_quick, run_train


def main():
    args = parse_args()
    print("=" * 60)
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("=" * 60)

    if args.mode == "quick":
        run_quick(args)
    elif args.mode == "train":
        run_train(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
