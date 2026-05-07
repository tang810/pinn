import os

import numpy as np
import torch

from .plotting import save_two_panel
from .trainer import quick_warmup_scalar, quick_warmup_recover


def run_quick_scalar(model, tau, points, args):
    model_path = os.path.join(args.model_dir, "scalar_model.pt")
    tau_path = os.path.join(args.model_dir, "scalar_tau.pt")

    if os.path.exists(model_path) and os.path.exists(tau_path):
        model.load_state_dict(torch.load(model_path, map_location=points.device))
        tau_state = torch.load(tau_path, map_location="cpu")
        tau = torch.tensor(float(tau_state["tau"]), device=points.device, requires_grad=True)
        print(f"[quick][scalar] loaded checkpoint from {model_path}")
    else:
        print("[quick][scalar] checkpoint not found, running quick warmup...")
        model, tau = quick_warmup_scalar(model, tau, points, args)

    with torch.no_grad():
        u_pred = model(points).detach().cpu().numpy().flatten()
        tau_value = float(tau.detach().cpu().item())

    points_xy = points.detach().cpu().numpy()
    tau_field = np.full_like(u_pred, tau_value)
    save_two_panel(
        points_xy,
        u_pred,
        tau_field,
        "Predicted u",
        f"Predicted tau = {tau_value:.4f}",
        os.path.join(args.output_dir, "quick_scalar.png"),
    )

    with open(os.path.join(args.output_dir, "quick_scalar_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"tau={tau_value:.8f}\n")


def run_quick_recover(model, coef_net, points, args):
    model_path = os.path.join(args.model_dir, "recover_model.pt")
    coef_path = os.path.join(args.model_dir, "recover_coef.pt")

    if os.path.exists(model_path) and os.path.exists(coef_path):
        model.load_state_dict(torch.load(model_path, map_location=points.device))
        coef_net.load_state_dict(torch.load(coef_path, map_location=points.device))
        print(f"[quick][recover] loaded checkpoint from {model_path}")
    else:
        print("[quick][recover] checkpoint not found, running quick warmup...")
        model, coef_net = quick_warmup_recover(model, coef_net, points, args)

    with torch.no_grad():
        u_pred = model(points).detach().cpu().numpy().flatten()
        tau_pred = coef_net(points).detach().cpu().numpy().flatten()

    points_xy = points.detach().cpu().numpy()
    save_two_panel(
        points_xy,
        u_pred,
        tau_pred,
        "Predicted u",
        "Recovered tau(x, y)",
        os.path.join(args.output_dir, "quick_recover.png"),
    )

    with open(os.path.join(args.output_dir, "quick_recover_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"tau_mean={float(np.mean(tau_pred)):.8f}\n")
        f.write(f"tau_std={float(np.std(tau_pred)):.8f}\n")
