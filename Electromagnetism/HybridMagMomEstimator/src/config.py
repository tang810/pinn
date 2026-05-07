import os


def define_arguments(parser):
    parser.add_argument("--mode", type=str, default="quick", choices=["train", "quick"], help="Run mode")
    parser.add_argument(
        "--data",
        type=str,
        default="simul",
        choices=["simul", "real"],
        help="Data source flag: simul uses built-in synthetic path, real uses MAT file",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="cpu or cuda:x")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--input_size", type=int, default=7, help="Input features: x,y,z,theta,Bx,By,Bz")
    parser.add_argument("--output_size", type=int, default=48, help="Output magnetic moments")
    parser.add_argument("--hidden_dims", type=int, nargs=3, default=[128, 256, 128], help="MLP hidden dims")

    parser.add_argument("--epochs", type=int, default=5000, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.1, help="Adam learning rate")
    parser.add_argument("--log_every", type=int, default=100, help="Log interval")

    parser.add_argument(
        "--mat_path",
        type=str,
        default="data/Large_ship_model,L=152.96m,W=17.28m,rz0=10m,ry0=0m,theta=0rad,V=3ms,SNR=0dB.mat",
        help="Path to source .mat file (used in real mode)",
    )
    parser.add_argument("--history_path", type=str, default="", help="Optional explicit history .npy path")
    parser.add_argument("--output_dir", type=str, default="results", help="Figures and logs directory")
    parser.add_argument("--model_dir", type=str, default="model", help="Model checkpoint directory")


def define_default_values(args):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    def _resolve(path):
        path = os.path.expanduser(path)
        if os.path.isabs(path):
            return path
        return os.path.join(project_root, path)

    default_mat = _resolve(
        "data/Large_ship_model,L=152.96m,W=17.28m,rz0=10m,ry0=0m,theta=0rad,V=3ms,SNR=0dB.mat"
    )
    default_history = _resolve("data/magnetic_moments_history.npy")

    if args.data == "simul":
        args.mat_path = "simul"
        if args.history_path:
            args.history_path = _resolve(args.history_path)
        else:
            args.history_path = default_history
    else:
        args.mat_path = _resolve(args.mat_path) if args.mat_path else default_mat
        if args.history_path:
            args.history_path = _resolve(args.history_path)

    args.output_dir = _resolve(args.output_dir)
    args.model_dir = _resolve(args.model_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
