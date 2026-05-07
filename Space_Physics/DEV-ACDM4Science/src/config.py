import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="DEV-ACDM4Science standardized runner")
    parser.add_argument("--mode", type=str, default="quick", choices=["quick", "train"])
    parser.add_argument("--data", type=str, default="simul", choices=["simul", "real"])
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/simul_acdm.npz",
        help="Path to dataset for real mode, or synthetic output location for simul mode.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--model_dir", type=str, default="model")
    parser.add_argument("--result_dir", type=str, default="results")
    parser.add_argument(
        "--task",
        type=str,
        default="acdm_cylinder",
        choices=["acdm_cylinder", "acdm_foilslice", "ddpm"],
        help="Original script task to launch in train mode.",
    )
    parser.add_argument(
        "--delegate_original",
        action="store_true",
        help="When set in train mode, call original scripts under src/scripts.",
    )
    return parser.parse_args()

