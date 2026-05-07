import argparse

from src.config import define_arguments, define_default_values
from src.evaluate import run as eval_run
from src.train import train_model


def main():
    parser = argparse.ArgumentParser()
    define_arguments(parser)
    args = parser.parse_args()
    define_default_values(args)

    print("=" * 60)
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("=" * 60)

    if args.mode == "train":
        train_model(args)
    elif args.mode == "quick":
        eval_run(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
