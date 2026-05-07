import os


def define_arguments(parser):
    parser.add_argument("--mode", type=str, default="quick", choices=["train", "quick"], help="Run mode")
    parser.add_argument("--task", type=str, default="inverse", choices=["inverse", "forward"], help="Problem type")

    parser.add_argument("--device", type=str, default="cuda:0", help="cpu or cuda:x")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--length", type=float, default=3.0, help="Physical box length")
    parser.add_argument("--dpml", type=float, default=1.0, help="PML thickness")
    parser.add_argument("--k0", type=float, default=2.0944, help="Wave number")
    parser.add_argument("--omega", type=float, default=2.0944, help="Angular frequency")
    parser.add_argument("--eps", type=float, default=4.0, help="Permittivity for forward task")
    parser.add_argument("--target_r_init", type=float, default=0.3, help="Initial inverse radius")
    parser.add_argument("--target_eps_init", type=float, default=2.0, help="Initial inverse permittivity")

    parser.add_argument("--hidden_dim", type=int, default=50, help="Hidden width")
    parser.add_argument("--num_layers", type=int, default=5, help="Hidden depth")

    parser.add_argument("--num_domain", type=int, default=1024, help="Domain points per epoch")
    parser.add_argument("--num_boundary", type=int, default=128, help="Boundary points per epoch")
    parser.add_argument("--train_epochs", type=int, default=10000, help="Adam epochs for inverse task")
    parser.add_argument("--adam_epochs", type=int, default=1000, help="Adam epochs for forward task")
    parser.add_argument("--lbfgs_epochs", type=int, default=100, help="LBFGS epochs for forward task")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--w_boundary", type=float, default=0.1, help="Boundary loss weight for inverse task")
    parser.add_argument("--save_every", type=int, default=1000, help="Checkpoint interval")
    parser.add_argument("--viz_resolution", type=int, default=100, help="Visualization grid size")

    parser.add_argument("--output_dir", type=str, default="result", help="Figures and logs directory")
    parser.add_argument("--model_dir", type=str, default="model", help="Model checkpoint directory")
    parser.add_argument("--viz_dir", type=str, default="result", help="Deprecated, mapped to output_dir")
    parser.add_argument("--ckpt", type=str, default="", help="Checkpoint path for quick mode")


def define_default_values(args):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    def _resolve(path):
        path = os.path.expanduser(path)
        if os.path.isabs(path):
            return path
        return os.path.join(project_root, path)

    args.output_dir = _resolve(args.output_dir)
    args.model_dir = _resolve(args.model_dir)
    args.viz_dir = args.output_dir
    if getattr(args, "ckpt", ""):
        args.ckpt = _resolve(args.ckpt)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
