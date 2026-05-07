import os


def define_arguments(parser):
    parser.add_argument("--mode", type=str, default="quick", choices=["train", "quick"], help="Run mode")
    parser.add_argument("--data_csv", type=str, default="data/Analytical_value_55K03.csv", help="Input CSV path")
    parser.add_argument("--element", type=str, default="cu", help="Element column name to analyze")
    parser.add_argument("--output_dir", type=str, default="result", help="Result directory")
    parser.add_argument("--model_dir", type=str, default="model", help="Model/checkpoint directory")
    parser.add_argument("--grid_size", type=int, default=100, help="Interpolation grid size")


def define_default_values(args):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    def _resolve(path):
        path = os.path.expanduser(path)
        if os.path.isabs(path):
            return path
        return os.path.join(project_root, path)

    args.data_csv = _resolve(args.data_csv)
    args.output_dir = _resolve(args.output_dir)
    args.model_dir = _resolve(args.model_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
