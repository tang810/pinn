import torch

from .data_utils import boundary_condition, exact_solution, select_boundary_points


def _grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]


def scalar_loss(model, tau: torch.Tensor, points: torch.Tensor, lambda_pde=1.1, lambda_bc=1.0, lambda_data=1.0):
    points = points.clone().detach().requires_grad_(True)
    u = model(points)

    grad_u = _grad(u, points)
    grad_ux = grad_u[:, 0:1]
    grad_uy = grad_u[:, 1:2]

    grad_uxx = _grad(grad_ux * tau, points)[:, 0:1]
    grad_uyy = _grad(grad_uy * tau, points)[:, 1:2]

    res = grad_uxx + grad_uyy + u
    loss_f = torch.mean(res ** 2)

    boundary_points = select_boundary_points(points)
    pred_bc = model(boundary_points)
    true_bc = boundary_condition(boundary_points)
    loss_bc = torch.mean((pred_bc - true_bc) ** 2)

    u_data = exact_solution(points)
    loss_data = torch.mean((u - u_data) ** 2)

    total = lambda_pde * loss_f + lambda_bc * loss_bc + lambda_data * loss_data
    return total, loss_f, loss_bc, loss_data


def recover_loss(model, coef_net, points: torch.Tensor, lambda_pde=1.1, lambda_bc=1.0, lambda_data=1.0):
    points = points.clone().detach().requires_grad_(True)
    u = model(points)
    tau = coef_net(points)

    grad_u = _grad(u, points)
    grad_ux = grad_u[:, 0:1]
    grad_uy = grad_u[:, 1:2]

    grad_uxx = _grad(grad_ux * tau, points)[:, 0:1]
    grad_uyy = _grad(grad_uy * tau, points)[:, 1:2]

    res = grad_uxx + grad_uyy + u
    loss_f = torch.mean(res ** 2)

    boundary_points = select_boundary_points(points)
    pred_bc = model(boundary_points)
    true_bc = boundary_condition(boundary_points)
    loss_bc = torch.mean((pred_bc - true_bc) ** 2)

    u_data = exact_solution(points)
    loss_data = torch.mean((u - u_data) ** 2)

    total = lambda_pde * loss_f + lambda_bc * loss_bc + lambda_data * loss_data
    return total, loss_f, loss_bc, loss_data
