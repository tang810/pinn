import torch
from .physics import body_force, dirichlet_bc, neumann_bc

d = 2

def compute_strain_stress_div_sigma(model, x, lambda_val, mu_val):
    x = x.clone().detach().requires_grad_(True).to(next(model.parameters()).device)
    u = model(x)

    grads = []
    for i in range(d):
        grad_u_i = torch.autograd.grad(u[:, i].sum(), x, create_graph=True)[0]
        grads.append(grad_u_i)
    grad_u = torch.stack(grads, dim=1)

    epsilon = 0.5 * (grad_u + grad_u.transpose(1, 2))
    trace_epsilon = epsilon[:, 0, 0] + epsilon[:, 1, 1]
    identity = torch.eye(d, device=x.device).unsqueeze(0).expand(x.size(0), -1, -1)
    sigma = lambda_val * trace_epsilon[:, None, None] * identity + 2.0 * mu_val * epsilon

    div_sigma = torch.zeros(x.size(0), d, device=x.device)
    for j in range(d):
        grad_sigma_j = torch.autograd.grad(sigma[:, j, :].sum(), x, create_graph=True)[0]
        div_sigma[:, j] = grad_sigma_j.sum(dim=1)

    return u, epsilon, sigma, div_sigma

def compute_loss(model, x_pde, x_dir, x_neu, n_neu, lambda_val, mu_val):
    u_pde, _, _, div_sigma_pde = compute_strain_stress_div_sigma(model, x_pde, lambda_val, mu_val)
    b = body_force(x_pde.to(div_sigma_pde.device))
    l_pde = torch.mean((div_sigma_pde + b) ** 2)

    u_dir_pred, _, _, _ = compute_strain_stress_div_sigma(model, x_dir, lambda_val, mu_val)
    u_dir_true = dirichlet_bc(x_dir)
    l_dir = torch.mean((u_dir_pred - u_dir_true) ** 2)

    _, _, sigma_neu, _ = compute_strain_stress_div_sigma(model, x_neu, lambda_val, mu_val)
    traction_pred = torch.einsum("nij,nj->ni", sigma_neu, n_neu)
    traction_true = neumann_bc(x_neu, n_neu)
    l_neu = torch.mean((traction_pred - traction_true) ** 2)

    loss = l_pde + 10.0 * l_dir + 5.0 * l_neu
    return loss, l_pde, l_dir, l_neu
