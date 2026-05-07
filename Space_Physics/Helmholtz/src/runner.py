import os

import torch

from .data import generate_1d_points
from .losses import compute_losses
from .model import HelmholtzPINN
from .viz import save_loss_curve, save_prediction_plot


def _safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _build_model(args, device):
    model = HelmholtzPINN(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        k=args.k,
    ).to(device)
    return model


def run_train(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _build_model(args, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    hist = {"total": [], "pde": [], "bc": []}

    for epoch in range(args.epochs):
        x_domain, x_boundary, u_boundary = generate_1d_points(args.num_domain)
        x_domain = x_domain.to(device)
        x_boundary = x_boundary.to(device)
        u_boundary = u_boundary.to(device)

        total, pde, bc = compute_losses(model, x_domain, x_boundary, u_boundary)

        optimizer.zero_grad()
        total.backward()
        optimizer.step()

        hist["total"].append(float(total.detach().cpu()))
        hist["pde"].append(float(pde.cpu()))
        hist["bc"].append(float(bc.cpu()))

        if (epoch + 1) % max(1, args.log_every) == 0:
            print(f"[train] epoch={epoch+1}/{args.epochs} total={hist['total'][-1]:.3e}")

    os.makedirs(args.model_dir, exist_ok=True)
    ckpt = {
        "state_dict": model.state_dict(),
        "k": float(model.k.detach().cpu()),
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
    }
    model_path = os.path.join(args.model_dir, args.model_name)
    torch.save(ckpt, model_path)

    os.makedirs(args.result_dir, exist_ok=True)
    save_loss_curve(hist, os.path.join(args.result_dir, "loss_curve.png"))
    save_prediction_plot(model, os.path.join(args.result_dir, "prediction.png"))
    print(f"[train] model saved: {model_path}")


def run_quick(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model_path = os.path.join(args.model_dir, args.model_name)

    if os.path.exists(model_path):
        ckpt = _safe_torch_load(model_path, device)
        hidden_dim = int(ckpt.get("hidden_dim", args.hidden_dim))
        num_layers = int(ckpt.get("num_layers", args.num_layers))
        model = HelmholtzPINN(hidden_dim=hidden_dim, num_layers=num_layers, k=float(ckpt.get("k", args.k))).to(device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"[quick] loaded model: {model_path}")
    else:
        # quick fallback: short training to guarantee runnable workflow.
        model = _build_model(args, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(args.quick_epochs):
            x_domain, x_boundary, u_boundary = generate_1d_points(args.num_domain)
            x_domain = x_domain.to(device)
            x_boundary = x_boundary.to(device)
            u_boundary = u_boundary.to(device)
            total, _, _ = compute_losses(model, x_domain, x_boundary, u_boundary)
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
        os.makedirs(args.model_dir, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "k": float(model.k.detach().cpu()),
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
            },
            model_path,
        )
        print(f"[quick] no checkpoint found, trained quick model and saved: {model_path}")

    os.makedirs(args.result_dir, exist_ok=True)
    save_prediction_plot(model, os.path.join(args.result_dir, "quick_prediction.png"))
    print(f"[quick] result saved: {os.path.join(args.result_dir, 'quick_prediction.png')}")
