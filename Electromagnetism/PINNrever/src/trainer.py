import os

import torch

from .losses import scalar_loss, recover_loss
from .plotting import save_loss_curve


def train_scalar(model, tau, points, args):
    optimizer = torch.optim.Adam(list(model.parameters()) + [tau], lr=args.lr)
    loss_hist = []

    for epoch in range(args.train_epochs):
        optimizer.zero_grad()
        loss, _, _, _ = scalar_loss(
            model,
            tau,
            points,
            lambda_pde=args.lambda_pde,
            lambda_bc=args.lambda_bc,
            lambda_data=args.lambda_data,
        )
        loss.backward()
        optimizer.step()
        loss_hist.append(float(loss.item()))

        if epoch % args.save_every == 0:
            print(f"[train][scalar] epoch={epoch} loss={loss.item():.6e}")

    model_path = os.path.join(args.model_dir, "scalar_model.pt")
    tau_path = os.path.join(args.model_dir, "scalar_tau.pt")
    torch.save(model.state_dict(), model_path)
    torch.save({"tau": float(tau.detach().cpu().item())}, tau_path)

    save_loss_curve(loss_hist, os.path.join(args.output_dir, "train_scalar_loss.png"))
    return model, tau


def train_recover(model, coef_net, points, args):
    optimizer = torch.optim.Adam(list(model.parameters()) + list(coef_net.parameters()), lr=args.lr)
    loss_hist = []

    for epoch in range(args.train_epochs):
        optimizer.zero_grad()
        loss, _, _, _ = recover_loss(
            model,
            coef_net,
            points,
            lambda_pde=args.lambda_pde,
            lambda_bc=args.lambda_bc,
            lambda_data=args.lambda_data,
        )
        loss.backward()
        optimizer.step()
        loss_hist.append(float(loss.item()))

        if epoch % args.save_every == 0:
            print(f"[train][recover] epoch={epoch} loss={loss.item():.6e}")

    model_path = os.path.join(args.model_dir, "recover_model.pt")
    coef_path = os.path.join(args.model_dir, "recover_coef.pt")
    torch.save(model.state_dict(), model_path)
    torch.save(coef_net.state_dict(), coef_path)

    save_loss_curve(loss_hist, os.path.join(args.output_dir, "train_recover_loss.png"))
    return model, coef_net


def quick_warmup_scalar(model, tau, points, args):
    optimizer = torch.optim.Adam(list(model.parameters()) + [tau], lr=args.lr)
    for _ in range(args.quick_epochs):
        optimizer.zero_grad()
        loss, _, _, _ = scalar_loss(
            model,
            tau,
            points,
            lambda_pde=args.lambda_pde,
            lambda_bc=args.lambda_bc,
            lambda_data=args.lambda_data,
        )
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), os.path.join(args.model_dir, "scalar_model.pt"))
    torch.save({"tau": float(tau.detach().cpu().item())}, os.path.join(args.model_dir, "scalar_tau.pt"))
    return model, tau


def quick_warmup_recover(model, coef_net, points, args):
    optimizer = torch.optim.Adam(list(model.parameters()) + list(coef_net.parameters()), lr=args.lr)
    for _ in range(args.quick_epochs):
        optimizer.zero_grad()
        loss, _, _, _ = recover_loss(
            model,
            coef_net,
            points,
            lambda_pde=args.lambda_pde,
            lambda_bc=args.lambda_bc,
            lambda_data=args.lambda_data,
        )
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), os.path.join(args.model_dir, "recover_model.pt"))
    torch.save(coef_net.state_dict(), os.path.join(args.model_dir, "recover_coef.pt"))
    return model, coef_net
