import os

import torch
import torch.optim as optim

from .data_loader import create_visualization_grid, generate_domain_points, generate_inverse_points
from .model import FNN
from .physics import boundary_condition_loss, pde_forward, pde_inverse, sigma0, transform_inverse
from .utils import plot_complex_field, select_device, setup_seed


def _save_checkpoint(path, epoch, model, optimizer, radius=None, eps_param=None, loss=None, task="inverse"):
    payload = {
        "task": task,
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": None if loss is None else float(loss),
    }
    if radius is not None:
        payload["R_param"] = float(radius)
    if eps_param is not None:
        payload["eps_param"] = float(eps_param)
    torch.save(payload, path)


def _viz_and_save(net, vis_points, xx, yy, save_path, radius=None):
    with torch.no_grad():
        vis_outputs = net(vis_points)
        if radius is not None:
            vis_outputs = transform_inverse(vis_points, vis_outputs, radius)
        re_field = vis_outputs[:, 0].reshape(xx.shape).detach().cpu().numpy()
        im_field = vis_outputs[:, 1].reshape(xx.shape).detach().cpu().numpy()
    plot_complex_field(
        xx,
        yy,
        re_field,
        im_field,
        save_path,
        "Real Part of Electric Field",
        "Imaginary Part of Electric Field",
        radius=radius,
    )


def _train_forward(args, device):
    box = torch.tensor(
        [[-args.length / 2.0, -args.length / 2.0], [args.length / 2.0, args.length / 2.0]],
        dtype=torch.float32,
        device=device,
    )
    sigma0_value = sigma0(args.dpml)

    net = FNN(num_layers=args.num_layers, num_nodes=args.hidden_dim).to(device)
    optimizer_adam = optim.Adam(net.parameters(), lr=args.lr)
    domain_points = generate_domain_points(args.num_domain, args.length, args.dpml, device)

    for i in range(args.adam_epochs):
        optimizer_adam.zero_grad()
        outputs = net(domain_points)
        loss_re, loss_im = pde_forward(domain_points, outputs, args.eps, args.k0, box, args.omega, sigma0_value)
        loss = torch.mean(loss_re**2) + torch.mean(loss_im**2)
        loss.backward()
        optimizer_adam.step()
        if i % 100 == 0:
            print(f"Adam Iteration {i}, Loss: {loss.item():.6f}")

    optimizer_lbfgs = torch.optim.LBFGS(net.parameters())

    def closure():
        optimizer_lbfgs.zero_grad()
        outputs = net(domain_points)
        loss_re, loss_im = pde_forward(domain_points, outputs, args.eps, args.k0, box, args.omega, sigma0_value)
        total_loss = torch.mean(loss_re**2) + torch.mean(loss_im**2)
        total_loss.backward()
        return total_loss

    for i in range(args.lbfgs_epochs):
        optimizer_lbfgs.step(closure)
        if i % 10 == 0:
            print(f"L-BFGS Iteration {i}, Loss: {closure().item():.6f}")

    vis_points, xx, yy = create_visualization_grid(args.length, args.dpml, args.viz_resolution, device)
    viz_path = os.path.join(args.output_dir, "pytorchES.png")
    _viz_and_save(net, vis_points, xx, yy, viz_path, radius=None)

    model_path = os.path.join(args.model_dir, "forward_model.pt")
    _save_checkpoint(model_path, args.adam_epochs + args.lbfgs_epochs, net, optimizer_lbfgs, loss=float(closure().item()), task="forward")
    print(f"Forward model saved to {model_path}")


def _train_inverse(args, device):
    setup_seed(args.seed)
    sigma0_value = sigma0(args.dpml)

    vis_points, xx, yy = create_visualization_grid(args.length, args.dpml, args.viz_resolution, device)
    net = FNN(num_layers=args.num_layers, num_nodes=args.hidden_dim).to(device)

    r_param = torch.tensor([args.target_r_init], dtype=torch.float32, device=device, requires_grad=True)
    eps_param = torch.tensor([args.target_eps_init], dtype=torch.float32, device=device, requires_grad=True)

    optimizer = optim.Adam(list(net.parameters()) + [r_param, eps_param], lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1000, threshold=1e-4
    )

    for epoch in range(args.train_epochs):
        domain_points, boundary_points = generate_inverse_points(
            args.num_domain, args.num_boundary, args.length, args.dpml, float(r_param.item()), device
        )

        optimizer.zero_grad()
        outputs = net(domain_points)
        loss_re, loss_im = pde_inverse(
            domain_points,
            outputs,
            radius=r_param,
            eps_pred=eps_param,
            k0=args.k0,
            length=args.length,
            dpml=args.dpml,
            omega=args.omega,
            sigma0_value=sigma0_value,
        )
        pde_loss = torch.mean(loss_re**2) + torch.mean(loss_im**2)
        bc_loss = boundary_condition_loss(net, boundary_points, r_param, args.k0)
        total_loss = pde_loss + args.w_boundary * bc_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss)

        if epoch % 100 == 0:
            print(
                f"Epoch {epoch}, Total Loss: {total_loss.item():.6f}, "
                f"PDE Loss: {pde_loss.item():.6f}, Boundary Loss: {bc_loss.item():.6f}, "
                f"R: {r_param.item():.4f}, eps: {eps_param.item():.4f}"
            )

        if epoch % args.save_every == 0:
            ckpt_name = f"model_epoch_{epoch}.pt"
            ckpt_path = os.path.join(args.model_dir, ckpt_name)
            _save_checkpoint(
                ckpt_path,
                epoch,
                net,
                optimizer,
                radius=r_param.item(),
                eps_param=eps_param.item(),
                loss=total_loss.item(),
                task="inverse",
            )
            viz_path = os.path.join(args.output_dir, f"epoch_{epoch}_Re_Im.png")
            _viz_and_save(net, vis_points, xx, yy, viz_path, radius=r_param.item())

    print(f"Optimized params: R = {r_param.item():.4f}, eps = {eps_param.item():.4f}")
    final_output_ckpt = os.path.join(args.model_dir, "final_model.pt")
    _save_checkpoint(
        final_output_ckpt,
        args.train_epochs - 1,
        net,
        optimizer,
        radius=r_param.item(),
        eps_param=eps_param.item(),
        loss=total_loss.item(),
        task="inverse",
    )
    final_viz = os.path.join(args.output_dir, "final_Re_Im.png")
    _viz_and_save(net, vis_points, xx, yy, final_viz, radius=r_param.item())
    print(f"Inverse model saved to {final_output_ckpt}")


def train_model(args):
    device = select_device(args.device)
    print(f"Using device: {device}")
    if args.task == "forward":
        _train_forward(args, device)
    else:
        _train_inverse(args, device)
