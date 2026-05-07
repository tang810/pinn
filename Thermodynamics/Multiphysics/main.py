import argparse

from src.config import define_arguments
from src.evaluate import run as eval_run
from src.train import train_model
from src.viz_sym_geom_from_data import run_viz


def main():
    parser = argparse.ArgumentParser()
    define_arguments(parser)
    parser.add_argument("--data", type=str, help="Shortcut or path for data (e.g., simul)")
    args = parser.parse_args()

    if args.data == "simul":
        args.data_path = "data/data_7246.txt"
    elif args.data:
        args.data_path = args.data

    print("=" * 60)
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("=" * 60)

    if args.mode == "train":
        train_model(args)
    elif args.mode == "quick":
        # keep user-selected device; fallback is handled inside evaluate
        if not getattr(args, "ckpt", None) or not __import__("os").path.exists(args.ckpt):
            args.ckpt = "model/pinnsformer_withsun.pt"
        eval_run(args)

        # optional sym visualization (does not block quick result generation)
        args.data_path_gt = args.data_path
        args.out_dir = "results/viz_sym_res"
        args.geometry = "csg"
        args.geometry_backend = "sym"
        args.sym_shape = "box"
        args.box_p1 = [0.0, 0.0, 0.0]
        args.box_p2 = [1.0, 1.0, 1.0]
        args.res_points = 120000
        args.bc_points = 20000
        args.ic_points = 5000
        args.plot_on = "res"
        args.make_stress = True
        args.fps = 6
        args.cbar_mode = "shared"
        args.t_tol = 1e-6
        args.cleanup_frames = True
        args.pred_calib = "none"
        args.pred_denorm = False
        args.input_norm = "none"
        args.t_norm = "none"
        args.tmin = None
        args.tmax = None
        args.calib_points = 20000

        try:
            run_viz(args)
        except Exception as e:
            print(f"[warn] run_viz skipped: {e}")
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
