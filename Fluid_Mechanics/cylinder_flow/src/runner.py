import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data_utils import DataLoader
from .pinn_solver import PysicsInformedNeuralNetwork, device as solver_device
from .sim_data import ensure_simul_data


def _setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_data_path(args):
    data_path = Path(args.data_path)
    if args.data == "simul":
        return ensure_simul_data(data_path, seed=args.seed)
    if not data_path.exists():
        raise FileNotFoundError(f"Real data file not found: {data_path}")
    return data_path


def _build_pinn(args, net_params=None, net_params_1=None):
    ckpt_dir = Path(args.model_dir) / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return PysicsInformedNeuralNetwork(
        Re=args.re,
        layers=args.layers,
        layers_1=args.layers_aux,
        hidden_size=args.hidden_size,
        hidden_size_1=args.hidden_size_aux,
        alpha_evm=args.alpha_evm,
        bc_weight=args.lam_bc,
        eq_weight=args.lam_equ,
        ic_weight=args.lam_ic,
        outlet_weight=args.lam_bc,
        checkpoint_path=str(ckpt_dir) + "/",
        net_params=net_params,
        net_params_1=net_params_1,
    )


def _prepare_window_data(dataloader, n_train, n_boundary, time_window, start=1):
    max_idx = len(dataloader.get_t_list()) - 1
    stop = min(start + time_window - 1, max_idx)
    frames = max(1, stop - start + 1)
    if max_idx >= 1 and stop >= start:
        dataloader.set_time_range(time_frames=frames, time_range=[start, stop])
    dataloader.sample_geometry(N_train=n_train, N_b=n_boundary)
    boundary_data, outlet_data = dataloader.get_boundary_data()
    training_data = dataloader.get_training_data()
    return boundary_data, outlet_data, training_data, stop


def _eval_and_save_predictions(pinn, dataloader, result_dir, eval_snap):
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    snap = min(eval_snap, len(dataloader.get_t_list()))
    p_star, u_star, v_star, t_star, x_star, y_star = dataloader.get_evaluate_data(snap=snap)
    pinn.evaluate(t_star, x_star, y_star, u_star, v_star)

    with torch.no_grad():
        tt = torch.tensor(t_star).float().to(solver_device)
        xx = torch.tensor(x_star).float().to(solver_device)
        yy = torch.tensor(y_star).float().to(solver_device)
        u_pred, v_pred, p_pred, _ = pinn.neural_net_u(tt, xx, yy)
    u_pred_np = u_pred.detach().cpu().numpy()
    v_pred_np = v_pred.detach().cpu().numpy()
    p_pred_np = p_pred.detach().cpu().numpy()

    np.savez(
        result_dir / "quick_prediction.npz",
        t=t_star,
        x=x_star,
        y=y_star,
        u_true=u_star,
        v_true=v_star,
        p_true=p_star,
        u_pred=u_pred_np,
        v_pred=v_pred_np,
        p_pred=p_pred_np,
    )
    _save_quick_figures(
        result_dir=result_dir,
        x=x_star,
        y=y_star,
        u_true=u_star,
        v_true=v_star,
        p_true=p_star,
        u_pred=u_pred_np,
        v_pred=v_pred_np,
        p_pred=p_pred_np,
    )


def _save_quick_figures(result_dir, x, y, u_true, v_true, p_true, u_pred, v_pred, p_pred):
    x = x.reshape(-1)
    y = y.reshape(-1)
    u_true = u_true.reshape(-1)
    v_true = v_true.reshape(-1)
    p_true = p_true.reshape(-1)
    u_pred = u_pred.reshape(-1)
    v_pred = v_pred.reshape(-1)
    p_pred = p_pred.reshape(-1)

    speed_true = np.sqrt(u_true**2 + v_true**2)
    speed_pred = np.sqrt(u_pred**2 + v_pred**2)

    def _scatter(ax, val, title):
        sc = ax.scatter(x, y, c=val, s=2, cmap="jet")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        plt.colorbar(sc, ax=ax, shrink=0.8)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=150)
    _scatter(axes[0, 0], speed_true, "Speed True")
    _scatter(axes[0, 1], speed_pred, "Speed Pred")
    _scatter(axes[1, 0], p_true, "Pressure True")
    _scatter(axes[1, 1], p_pred, "Pressure Pred")
    fig.suptitle("Cylinder Flow Quick Result (PINN)", fontsize=14)
    fig.tight_layout()
    fig.savefig(result_dir / "quick_result_uvp.png")
    plt.close(fig)

    err_u = np.linalg.norm(u_true - u_pred) / (np.linalg.norm(u_true) + 1e-12)
    err_v = np.linalg.norm(v_true - v_pred) / (np.linalg.norm(v_true) + 1e-12)
    err_p = np.linalg.norm(p_true - p_pred) / (np.linalg.norm(p_true) + 1e-12)
    fig2, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)
    labels = ["u", "v", "p"]
    vals = [err_u, err_v, err_p]
    ax.bar(labels, vals)
    ax.set_title("Relative L2 Error (Quick)")
    ax.set_ylabel("Relative L2")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3e}", ha="center", va="bottom", fontsize=9)
    fig2.tight_layout()
    fig2.savefig(result_dir / "quick_error_bar.png")
    plt.close(fig2)


def run_quick(args):
    _setup_seed(args.seed)
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    data_path = _resolve_data_path(args)

    main_ckpt = Path(args.model_dir) / "cylinder_flow_quick.pth"
    aux_ckpt = Path(args.model_dir) / "cylinder_flow_quick_evm.pth"
    net_params = str(main_ckpt) if main_ckpt.exists() else None
    net_params_1 = str(aux_ckpt) if aux_ckpt.exists() else None

    dataloader = DataLoader(filename=str(data_path), time_window=args.time_window)
    pinn = _build_pinn(args, net_params=net_params, net_params_1=net_params_1)

    initial_data = dataloader.get_initial_data()
    pinn.set_initial_data(X=initial_data)
    boundary_data, outlet_data, training_data, _ = _prepare_window_data(
        dataloader=dataloader,
        n_train=max(2000, min(args.n_train, 4000)),
        n_boundary=max(256, min(args.n_boundary, 1024)),
        time_window=max(3, min(args.time_window, 10)),
        start=1,
    )
    pinn.set_boundary_data(X=boundary_data)
    pinn.set_outlet_data(X=outlet_data)
    pinn.set_eq_training_data(X=training_data)

    if net_params is None or net_params_1 is None:
        print("[quick] checkpoint not found, running short training...")
        pinn.train(num_epoch=args.quick_epochs, lr=args.lr)
        torch.save(pinn.net.state_dict(), str(main_ckpt))
        torch.save(pinn.net_1.state_dict(), str(aux_ckpt))
    else:
        print(f"[quick] loaded checkpoints: {main_ckpt.name}, {aux_ckpt.name}")

    _eval_and_save_predictions(pinn, dataloader, args.result_dir, args.eval_snap)
    summary = {
        "mode": "quick",
        "data": args.data,
        "data_path": str(data_path),
        "main_checkpoint": str(main_ckpt),
        "aux_checkpoint": str(aux_ckpt),
        "result_file": str(Path(args.result_dir) / "quick_prediction.npz"),
    }
    (Path(args.result_dir) / "quick_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[quick] results saved to {Path(args.result_dir) / 'quick_prediction.npz'}")


def run_train(args):
    _setup_seed(args.seed)
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    data_path = _resolve_data_path(args)

    dataloader = DataLoader(filename=str(data_path), time_window=args.time_window)
    pinn = _build_pinn(args)
    t_list = dataloader.get_t_list()
    max_idx = len(t_list) - 1
    initial_data = dataloader.get_initial_data()

    if args.run_full_windows:
        windows = []
        start = 1
        while start <= max_idx:
            windows.append(start)
            start += args.time_window
    else:
        windows = [1]

    next_ic = initial_data
    for i, start in enumerate(windows):
        pinn.set_initial_data(X=next_ic)
        boundary_data, outlet_data, training_data, stop = _prepare_window_data(
            dataloader=dataloader,
            n_train=args.n_train,
            n_boundary=args.n_boundary,
            time_window=args.time_window,
            start=start,
        )
        pinn.set_boundary_data(X=boundary_data)
        pinn.set_outlet_data(X=outlet_data)
        pinn.set_eq_training_data(X=training_data)
        pinn.train(num_epoch=args.train_epochs, lr=args.lr)
        main_ckpt = Path(args.model_dir) / f"cylinder_flow_train_window_{i}.pth"
        aux_ckpt = Path(args.model_dir) / f"cylinder_flow_train_window_{i}_evm.pth"
        torch.save(pinn.net.state_dict(), str(main_ckpt))
        torch.save(pinn.net_1.state_dict(), str(aux_ckpt))
        print(f"[train] saved {main_ckpt.name}")

        snap = min(args.eval_snap, len(t_list))
        _, u_star, v_star, _, x_star, y_star = dataloader.get_evaluate_data(snap=snap)
        next_ic = pinn.prepare_next_ic(t_list[stop], x_star, y_star)

    _eval_and_save_predictions(pinn, dataloader, args.result_dir, args.eval_snap)
    print("[train] finished")
