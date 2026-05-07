import torch


def compute_losses(model, x_domain, x_boundary, u_boundary):
    x_domain = x_domain.clone().detach().requires_grad_(True)
    u = model(x_domain)

    du_dx = torch.autograd.grad(
        u,
        x_domain,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]
    d2u_dx2 = torch.autograd.grad(
        du_dx,
        x_domain,
        grad_outputs=torch.ones_like(du_dx),
        create_graph=True,
        retain_graph=True,
    )[0]

    pde_res = d2u_dx2 + (model.k ** 2) * u
    loss_pde = torch.mean(pde_res ** 2)

    u_b_pred = model(x_boundary)
    loss_bc = torch.mean((u_b_pred - u_boundary) ** 2)

    total = loss_pde + loss_bc
    return total, loss_pde.detach(), loss_bc.detach()
