import argparse
import os
import time

from src.data_loader import DataLoader
from src.pinn_solver import PysicsInformedNeuralNetwork as PINN_Standard
from src.pinn_solver_ev import PysicsInformedNeuralNetwork as PINN_EV


def resolve_paths(args):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data_dir or os.path.join(base_dir, "data")
    model_dir = args.model_dir or os.path.join(base_dir, "model")
    result_dir = args.result_dir or os.path.join(base_dir, "results")

    if args.data_path:
        eval_filename = args.data_path
        if not os.path.isabs(eval_filename):
            eval_filename = os.path.join(base_dir, eval_filename)
    else:
        eval_filename = os.path.join(data_dir, f"cavity_Re{args.reynolds}_256.mat")

    if args.data == "real" and not args.data_path:
        print("[warn] --data real set without --data_path, using default cavity data file.")

    return data_dir, model_dir, result_dir, eval_filename


def main(args):
    data_dir, model_dir, result_dir, eval_filename = resolve_paths(args)

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    print("Loading data...")
    dataloader = DataLoader(path=data_dir, N_f=args.n_f, N_b=args.n_b)
    x_star, y_star, u_star, v_star = dataloader.loading_evaluate_data(eval_filename)

    if args.model_type == "standard":
        print("Initializing Standard (NSKANet) Model...")

        if args.mode == "train":
            print("Mode: Training")
            pinn = PINN_Standard(
                Re=args.reynolds,
                N_f=args.n_f,
                bc_weight=args.bc_weight,
                eq_weight=args.eq_weight,
                model_dir=model_dir,
                result_dir=result_dir,
            )
            boundary_data = dataloader.loading_boundary_data()
            pinn.set_boundary_data(X=boundary_data)
            training_data = dataloader.loading_training_data()
            pinn.set_eq_training_data(X=training_data)

            start_time = time.time()
            for loop in range(0, 1):
                epoch = 100000
                pinn.train(num_epoch=epoch, lr=1e-3, label=100000)
                pinn.test(x_star, y_star, u_star, v_star, 100000, loop)

                pinn.train(num_epoch=epoch, lr=2e-4, label=200000)
                pinn.test(x_star, y_star, u_star, v_star, 200000, loop)

                pinn.train(num_epoch=epoch, lr=5e-5, label=300000)
                pinn.test(x_star, y_star, u_star, v_star, 300000, loop)

                pinn.train(num_epoch=epoch, lr=5e-5, label=400000)
                pinn.test(x_star, y_star, u_star, v_star, 400000, loop)

                pinn.train(num_epoch=epoch, lr=1e-5, label=500000)
                pinn.test(x_star, y_star, u_star, v_star, 500000, loop)

            train_time = time.time() - start_time
            print(f"Standard Model training finished. Total time: {train_time:.2f} seconds.")
            pinn.evaluate(x_star, y_star, u_star, v_star)

        elif args.mode == "quick":
            print("Mode: quick")
            if not args.model_path:
                print("No specified model path, using default model.")
                args.model_path = os.path.join(model_dir, "KAN_Re2k02_epoch_500000_ev_net.pth")
            print(f"Loading model from: {args.model_path}")
            pinn = PINN_Standard(
                Re=args.reynolds,
                N_f=args.n_f,
                bc_weight=args.bc_weight,
                eq_weight=args.eq_weight,
                net_params=args.model_path,
                model_dir=model_dir,
                result_dir=result_dir,
            )
            pinn.evaluate(x_star, y_star, u_star, v_star)

    elif args.model_type == "ev_enhanced":
        print("Initializing Enhanced Viscosity (ev_NSKANet) Model...")

        if args.mode == "train":
            print("Mode: Training")
            pinn = PINN_EV(
                Re=args.reynolds,
                N_f=args.n_f,
                alpha_evm=args.alpha_evm,
                bc_weight=args.bc_weight,
                eq_weight=args.eq_weight,
                model_dir=model_dir,
                result_dir=result_dir,
            )
            boundary_data = dataloader.loading_boundary_data()
            pinn.set_boundary_data(X=boundary_data)
            training_data = dataloader.loading_training_data()
            pinn.set_eq_training_data(X=training_data)

            start_time = time.time()
            for loop in range(0, 1):
                epoch = 20000
                pinn.set_alpha_evm(0.05)
                pinn.train(num_epoch=epoch, lr=1e-3, label=epoch)
                pinn.test(x_star, y_star, u_star, v_star, epoch, loop)

                pinn.set_alpha_evm(0.03)
                pinn.train(num_epoch=epoch, lr=2e-4, label=2 * epoch)
                pinn.test(x_star, y_star, u_star, v_star, 2 * epoch, loop)

                pinn.set_alpha_evm(0.02)
                pinn.train(num_epoch=epoch, lr=5e-5, label=3 * epoch)
                pinn.test(x_star, y_star, u_star, v_star, 3 * epoch, loop)

                pinn.set_alpha_evm(0.01)
                pinn.train(num_epoch=epoch, lr=5e-5, label=4 * epoch)
                pinn.test(x_star, y_star, u_star, v_star, 4 * epoch, loop)

                pinn.set_alpha_evm(0.005)
                pinn.train(num_epoch=epoch, lr=1e-5, label=5 * epoch)
                pinn.test(x_star, y_star, u_star, v_star, 5 * epoch, loop)

            train_time = time.time() - start_time
            print(f"Enhanced Model training finished. Total time: {train_time:.2f} seconds.")
            pinn.evaluate(x_star, y_star, u_star, v_star)

        elif args.mode == "quick":
            print("Mode: quick")
            if not args.model_path:
                print("No specified model path, using default model.")
                args.model_path = os.path.join(model_dir, "ev_net_checkpoint_epoch_100000.pth")
            print(f"Loading model from: {args.model_path}")
            pinn = PINN_EV(
                Re=args.reynolds,
                N_f=args.n_f,
                alpha_evm=args.alpha_evm,
                bc_weight=args.bc_weight,
                eq_weight=args.eq_weight,
                net_params=args.model_path,
                model_dir=model_dir,
                result_dir=result_dir,
            )
            pinn.evaluate(x_star, y_star, u_star, v_star)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cavity Flow PIKAN Solver")

    parser.add_argument("--mode", type=str, default="quick", choices=["train", "quick"], help="Execution mode")
    parser.add_argument("--data", type=str, default="simul", choices=["simul", "real"], help="Data source")
    parser.add_argument("--model_type", type=str, default="ev_enhanced", choices=["standard", "ev_enhanced"], help="Model type")

    parser.add_argument("--model_path", type=str, default=None, help="Path to a pre-trained model file")
    parser.add_argument("--data_path", type=str, default="", help="Optional explicit evaluation .mat file")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to data directory")
    parser.add_argument("--result_dir", type=str, default=None, help="Path to result directory")
    parser.add_argument("--model_dir", type=str, default=None, help="Path to model directory")

    parser.add_argument("--reynolds", type=int, default=2000, help="Reynolds number")
    parser.add_argument("--n_f", type=int, default=40000, help="Number of collocation points")
    parser.add_argument("--n_b", type=int, default=2500, help="Number of boundary points")
    parser.add_argument("--bc_weight", type=float, default=10.0, help="BC loss weight")
    parser.add_argument("--eq_weight", type=float, default=1.0, help="Equation loss weight")
    parser.add_argument("--alpha_evm", type=float, default=0.03, help="Initial EVM coefficient")

    args = parser.parse_args()
    main(args)