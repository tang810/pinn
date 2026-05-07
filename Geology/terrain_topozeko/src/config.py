import os


def define_arguments(parser):
    parser.add_argument("--mode", type=str, default="quick", choices=["train", "quick"], help="Run mode")
    parser.add_argument(
        "--data",
        type=str,
        default="simul",
        choices=["simul", "real"],
        help="Data source flag: simul for synthetic, real for .mat file",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="",
        help="Optional custom path to .mat file containing BED and SUR (used when --data real)",
    )
    parser.add_argument("--output_dir", type=str, default="results", help="Output figures directory")
    parser.add_argument("--model_dir", type=str, default="model", help="Model/checkpoint directory")
    parser.add_argument("--plot_name", type=str, default="topozeko_plot.png", help="Output figure filename")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda:x")
    parser.add_argument("--d2", action="store_true", help="Also render 2D thickness map")
    parser.add_argument("--save_legacy_name", action="store_true", help="Also save as topozeko_plot.png in output_dir")


def define_default_values(args):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    def _resolve(path):
        path = os.path.expanduser(path)
        if os.path.isabs(path):
            return path
        return os.path.join(project_root, path)

    default_real_data = os.path.join(project_root, "data", "example_data_Morteratsch_25m.mat")
    if args.data == "simul":
        args.data_path = "simul"
    elif args.data_path:
        args.data_path = _resolve(args.data_path)
    else:
        args.data_path = default_real_data

    args.output_dir = _resolve(args.output_dir)
    args.model_dir = _resolve(args.model_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
