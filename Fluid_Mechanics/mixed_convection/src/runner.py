import json
import shutil
from pathlib import Path

from .legacy_bridge import (
    _VARIANT_DIR,
    default_eval_data,
    find_checkpoint,
    import_legacy_modules,
    legacy_import_context,
)


def _build_pinn(psolver, args, net_params):
    pe = args.re * args.pr
    return psolver.PysicsInformedNeuralNetwork(
        Re=args.re,
        Pr=args.pr,
        Pe=pe,
        Ri=args.ri,
        layers=args.layers,
        hidden_size=args.hidden_size,
        N_f=args.n_f,
        bc_weight=args.lam_bcs,
        eq_weight=args.lam_equ,
        net_params=net_params,
        checkpoint_path="./checkpoint/",
    )


def _resolve_data_path(project_root: Path, args, legacy_dir: Path):
    if args.data == "load":
        data_path = Path(args.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"data file not found: {data_path}")
        return data_path
    data_path = default_eval_data(legacy_dir, args.re, args.pr, args.ri, args.variant)
    if not data_path.exists():
        raise FileNotFoundError(f"default simulation data not found: {data_path}")
    return data_path


def _copy_result_plot(legacy_dir: Path, result_dir: Path, mode: str):
    src_plot = legacy_dir / "result_plot_ev.png"
    if src_plot.exists():
        dst_plot = result_dir / f"{mode}_result_plot.png"
        shutil.copy2(src_plot, dst_plot)
        return str(dst_plot)
    return ""


def run_quick(args):
    project_root = Path(__file__).resolve().parents[1]
    result_dir = project_root / args.result_dir
    model_dir = project_root / args.model_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    cavity_data, psolver = import_legacy_modules(project_root, args.variant)

    with legacy_import_context(project_root, args.variant) as legacy_dir:
        ckpt = find_checkpoint(legacy_dir)
        pinn = _build_pinn(psolver, args, str(ckpt) if ckpt else None)
        if ckpt is None:
            dataloader = cavity_data.DataLoader(path="./datasets/", N_f=args.n_f, N_b=args.n_b)
            boundary_data = dataloader.loading_boundary_data()
            pinn.set_boundary_data(X=boundary_data)
            training_data = dataloader.loading_training_data()
            pinn.set_eq_training_data(X=training_data)
            pinn.train(num_epoch=args.quick_epochs, lr=args.lr)
            quick_ckpt = model_dir / "mixed_convection_quick.pth"
            pinn.save(quick_ckpt.name, directory=str(model_dir), N_HLayer=args.layers, N_neu=args.hidden_size, N_f=args.n_f)
            ckpt = quick_ckpt

        data_path = _resolve_data_path(project_root, args, legacy_dir)
        dataloader = cavity_data.DataLoader(path="./datasets/", N_f=args.n_f, N_b=args.n_b)
        x_star, y_star, u_star, v_star, t_star = dataloader.loading_evaluate_data(str(data_path))
        pinn.evaluate(x_star, y_star, u_star, v_star, t_star)

        copied_plot = _copy_result_plot(legacy_dir, result_dir, mode="quick")

    summary = {
        "mode": "quick",
        "variant": args.variant,
        "data": args.data,
        "data_path": str(data_path),
        "checkpoint": str(ckpt) if ckpt else "",
        "result_plot": copied_plot,
    }
    (result_dir / "quick_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[quick] summary saved: {result_dir / 'quick_summary.json'}")


def run_train(args):
    project_root = Path(__file__).resolve().parents[1]
    result_dir = project_root / args.result_dir
    model_dir = project_root / args.model_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    cavity_data, psolver = import_legacy_modules(project_root, args.variant)

    with legacy_import_context(project_root, args.variant) as legacy_dir:
        pinn = _build_pinn(psolver, args, None)
        dataloader = cavity_data.DataLoader(path="./datasets/", N_f=args.n_f, N_b=args.n_b)
        boundary_data = dataloader.loading_boundary_data()
        pinn.set_boundary_data(X=boundary_data)
        training_data = dataloader.loading_training_data()
        pinn.set_eq_training_data(X=training_data)

        pinn.train(num_epoch=args.train_epochs, lr=args.lr)
        train_ckpt = model_dir / "mixed_convection_train.pth"
        pinn.save(train_ckpt.name, directory=str(model_dir), N_HLayer=args.layers, N_neu=args.hidden_size, N_f=args.n_f)

        data_path = _resolve_data_path(project_root, args, legacy_dir)
        x_star, y_star, u_star, v_star, t_star = dataloader.loading_evaluate_data(str(data_path))
        pinn.evaluate(x_star, y_star, u_star, v_star, t_star)
        copied_plot = _copy_result_plot(legacy_dir, result_dir, mode="train")

    summary = {
        "mode": "train",
        "variant": args.variant,
        "data": args.data,
        "data_path": str(data_path),
        "checkpoint": str(train_ckpt),
        "result_plot": copied_plot,
    }
    (result_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[train] summary saved: {result_dir / 'train_summary.json'}")
