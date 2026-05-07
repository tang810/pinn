# -*- coding: utf-8 -*-
"""
对数似然（B-PINN 的数据项 + PDE 项）
"""
import torch

def model_loss(data_dict, fmodel, params_unflattened, tau_likes, gradients, params_single=None):
    """
    返回：
        ll: 对数似然（标量）
        outputs: [pred_u, pred_f]
    说明：
        - u 数据似然：Normal(y_u | û(x_u), like_std_u^2)
        - PDE 似然：  Normal(y_f | 0.01*Δû(x_f) + α*tanh(û(x_f)), like_std_f^2)
        - α = exp(θ) 保证正值
    """
    xu, yu = data_dict['x_u'], data_dict['y_u']
    xf, yf = data_dict['x_f'], data_dict['y_f']

    # u 数据似然（确保返回标量）
    pred_u = fmodel[0](xu, params=params_unflattened[0])
    ll = -0.5 * tau_likes[0] * ((pred_u - yu) ** 2).sum()

    # PDE 似然（需要 Δu）
    xf = xf.detach().requires_grad_(True)
    u  = fmodel[0](xf, params=params_unflattened[0])

    grad_u = torch.autograd.grad(outputs=u, inputs=xf,
                                 grad_outputs=torch.ones_like(u),
                                 create_graph=True, retain_graph=True)[0]
    ux, uy, uz = grad_u[:, 0:1], grad_u[:, 1:2], grad_u[:, 2:3]
    uxx = torch.autograd.grad(ux, xf, torch.ones_like(ux), create_graph=True, retain_graph=True)[0][:, 0:1]
    uyy = torch.autograd.grad(uy, xf, torch.ones_like(uy), create_graph=True, retain_graph=True)[0][:, 1:2]
    uzz = torch.autograd.grad(uz, xf, torch.ones_like(uz), create_graph=True, retain_graph=True)[0][:, 2:3]

    alpha  = torch.exp(params_single[0])
    pred_f = 0.01 * (uxx + uyy + uzz) + alpha * torch.tanh(u)

    ll = ll - 0.5 * tau_likes[1] * ((pred_f - yf) ** 2).sum()

    # —— 强制把 ll 变成 0-维标量，避免 autograd 抛出 “only for scalar outputs” —— #
    if ll.ndim != 0:
        ll = ll.reshape(-1).sum()

    return ll, [pred_u.detach(), pred_f.detach()]
