import torch
from src.physics import Ez_inc, contrast_chi


def pde_loss(model, xy, eps, r, fi, ai, theta, k):
    """
    Helmholtz PDE residual (strict scattering form)
    """
    xy.requires_grad_(True)
    out = model(xy, fi)

    Es_re = out[:, 2 * ai:2 * ai + 1]
    Es_im = out[:, 2 * ai + 1:2 * ai + 2]

    Ei_re, Ei_im = Ez_inc(xy, theta, k)
    chi = contrast_chi(xy, eps, r)

    # 计算梯度
    grad_re = torch.autograd.grad(Es_re.sum(), xy, create_graph=True)[0]
    grad_im = torch.autograd.grad(Es_im.sum(), xy, create_graph=True)[0]

    # 计算拉普拉斯算子
    lap_re = (
        torch.autograd.grad(grad_re[:, 0].sum(), xy, create_graph=True)[0][:, 0:1] +
        torch.autograd.grad(grad_re[:, 1].sum(), xy, create_graph=True)[0][:, 1:2]
    )
    lap_im = (
        torch.autograd.grad(grad_im[:, 0].sum(), xy, create_graph=True)[0][:, 0:1] +
        torch.autograd.grad(grad_im[:, 1].sum(), xy, create_graph=True)[0][:, 1:2]
    )

    # 散射场方程: (∇² + k²)Es = -k²χ(Ei + Es)
    res_re = lap_re + k**2 * Es_re + k**2 * chi * (Ei_re + Es_re)
    res_im = lap_im + k**2 * Es_im + k**2 * chi * (Ei_im + Es_im)

    return torch.mean(res_re**2 + res_im**2)


def data_loss(model, xy, Eobs_re, Eobs_im, fi, ai, theta, k):
    """
    Data loss (简化修复版，避免梯度错误)
    """
    out = model(xy, fi)

    Es_re = out[:, 2 * ai:2 * ai + 1]
    Es_im = out[:, 2 * ai + 1:2 * ai + 2]

    Ei_re, Ei_im = Ez_inc(xy, theta, k)

    Et_re = Ei_re + Es_re
    Et_im = Ei_im + Es_im

    # 简单的MSE损失
    mse_loss = torch.mean((Et_re - Eobs_re)**2 + (Et_im - Eobs_im)**2)
    
    # 添加振幅损失，提高对r的敏感度
    amp_obs = torch.sqrt(Eobs_re**2 + Eobs_im**2)
    amp_pred = torch.sqrt(Et_re**2 + Et_im**2)
    amp_loss = torch.mean((amp_pred - amp_obs)**2)
    
    # 总数据损失
    return 0.8 * mse_loss + 0.2 * amp_loss


def radius_constraint_loss(r, target_r):
    """
    半径约束损失
    """
    # 软约束：鼓励r接近目标值
    return 0.05 * (r - target_r)**2