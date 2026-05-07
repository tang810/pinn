import argparse
from pathlib import Path

from src.runner import run_eval_only, run_train_and_test


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Solar Wind Forecast standardized runner")
    parser.add_argument("--mode", type=str, default="quick", choices=["quick", "train"])
    parser.add_argument("--data", type=str, default="simul", choices=["simul", "real"])
    parser.add_argument("--case_root", type=str, default=".")
    parser.add_argument("--n_hours", type=int, default=24 * 240)
    parser.add_argument("--span", type=int, default=24)
    parser.add_argument("--prior", type=int, default=12)
    parser.add_argument("--lam", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--show_plots", action="store_true")
    parser.add_argument("--quick", action="store_true", help="use reduced config for fast smoke run")
    return parser


def args_to_config(args: argparse.Namespace) -> dict:
    cfg = {
        "n_hours": args.n_hours,
        "span": args.span,
        "prior": args.prior,
        "lam": args.lam,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "clip": args.clip,
        "seed": args.seed,
        "hidden": args.hidden,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "show_plots": args.show_plots,
    }
    if args.quick:
        cfg.update(
            {
                "epochs": min(cfg["epochs"], 10),
                "n_hours": min(cfg["n_hours"], 24 * 60),
                "batch_size": min(cfg["batch_size"], 128),
                "hidden": min(cfg["hidden"], 64),
                "num_layers": min(cfg["num_layers"], 2),
            }
        )
    return cfg


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    case_root = Path(args.case_root).resolve()
    cfg = args_to_config(args)

    print("=" * 60)
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("=" * 60)

    ckpt = case_root / "model" / "solar_wind_prediction_best.pt"
    if args.mode == "train":
        run_train_and_test(config=cfg, case_root=case_root)
        return

    if args.mode == "quick":
        # quick mode prefers eval-only; if no checkpoint, do a short warmup training first.
        if ckpt.exists():
            run_eval_only(config=cfg, case_root=case_root)
        else:
            print("[quick] checkpoint not found, running warmup train+test...")
            run_train_and_test(config=cfg, case_root=case_root)
        return

    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
