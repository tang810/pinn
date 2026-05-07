import numpy as np
import torch


def sigma0(dpml):
    return -np.log(1e-20) / (4 * dpml**3 / 3)


def pml_forward(inputs, box, omega, sigma0_value):
    def sigma(x, a, b):
        d1 = a - x
        d2 = x - b
        s1 = sigma0_value * d1**2 * torch.where(d1 > 0, torch.ones_like(d1), torch.zeros_like(d1))
        s2 = sigma0_value * d2**2 * torch.where(d2 > 0, torch.ones_like(d2), torch.zeros_like(d2))
        return s1 + s2

    def dsigma(x, a, b):
        d1 = a - x
        d2 = x - b
        s1 = 2 * sigma0_value * d1 * torch.where(d1 > 0, torch.ones_like(d1), torch.zeros_like(d1))
        s2 = 2 * sigma0_value * d2 * torch.where(d2 > 0, torch.ones_like(d2), torch.zeros_like(d2))
        return -s1 + s2

    sigma_x = sigma(inputs[:, :1], box[0][0], box[1][0])
    ab1 = 1 / (1 + 1j / omega * sigma_x) ** 2
    a1, b1 = ab1.real, ab1.imag
    dsigma_x = dsigma(inputs[:, :1], box[0][0], box[1][0])
    ab2 = -1j / omega * dsigma_x * ab1 / (1 + 1j / omega * sigma_x)
    a2, b2 = ab2.real, ab2.imag

    sigma_y = sigma(inputs[:, 1:2], box[0][1], box[1][1])
    ab3 = 1 / (1 + 1j / omega * sigma_y) ** 2
    a3, b3 = ab3.real, ab3.imag
    dsigma_y = dsigma(inputs[:, 1:2], box[0][1], box[1][1])
    ab4 = -1j / omega * dsigma_y * ab3 / (1 + 1j / omega * sigma_y)
    a4, b4 = ab4.real, ab4.imag

    return a1, b1, a2, b2, a3, b3, a4, b4


def pml_inverse(inputs, length, dpml, omega, sigma0_value):
    x = inputs[:, 0:1]
    y = inputs[:, 1:2]
    dpml_t = torch.tensor(dpml, dtype=inputs.dtype, device=inputs.device)
    x_dist = torch.minimum(length / 2 - torch.abs(x), dpml_t)
    y_dist = torch.minimum(length / 2 - torch.abs(y), dpml_t)
    sigma_x = sigma0_value * (1 - x_dist / dpml_t) ** 3
    sigma_y = sigma0_value * (1 - y_dist / dpml_t) ** 3
    a1 = 1 / (1 + sigma_x / omega) ** 2
    b1 = sigma_x / omega / (1 + sigma_x / omega) ** 2
    a2 = 1 / (1 + sigma_y / omega) ** 2
    b2 = sigma_y / omega / (1 + sigma_y / omega) ** 2
    return a1, b1, a2, b2


def transform_inverse(inputs, outputs, radius):
    dist_square = inputs[:, 0:1] ** 2 + inputs[:, 1:2] ** 2
    return (dist_square - radius**2) * outputs


def _first_second_derivatives(inputs, outputs):
    dre_dx = torch.autograd.grad(
        outputs[:, 0:1], inputs, grad_outputs=torch.ones_like(outputs[:, 0:1]), create_graph=True
    )[0]
    dre_xx = torch.autograd.grad(
        dre_dx[:, 0:1], inputs, grad_outputs=torch.ones_like(dre_dx[:, 0:1]), create_graph=True
    )[0][:, 0:1]
    dre_yy = torch.autograd.grad(
        dre_dx[:, 1:2], inputs, grad_outputs=torch.ones_like(dre_dx[:, 1:2]), create_graph=True
    )[0][:, 1:2]

    dim_dx = torch.autograd.grad(
        outputs[:, 1:2], inputs, grad_outputs=torch.ones_like(outputs[:, 1:2]), create_graph=True
    )[0]
    dim_xx = torch.autograd.grad(
        dim_dx[:, 0:1], inputs, grad_outputs=torch.ones_like(dim_dx[:, 0:1]), create_graph=True
    )[0][:, 0:1]
    dim_yy = torch.autograd.grad(
        dim_dx[:, 1:2], inputs, grad_outputs=torch.ones_like(dim_dx[:, 1:2]), create_graph=True
    )[0][:, 1:2]
    return dre_dx, dre_xx, dre_yy, dim_dx, dim_xx, dim_yy


def pde_forward(inputs, outputs, eps, k0, box, omega, sigma0_value):
    a1, b1, a2, b2, a3, b3, a4, b4 = pml_forward(inputs, box, omega, sigma0_value)

    dre_dx, dre_xx, dre_yy, dim_dx, dim_xx, dim_yy = _first_second_derivatives(inputs, outputs)
    dre_x = dre_dx[:, 0:1]
    dre_y = dre_dx[:, 1:2]
    dim_x = dim_dx[:, 0:1]
    dim_y = dim_dx[:, 1:2]

    re_e = outputs[:, 0:1]
    im_e = outputs[:, 1:2]

    loss_re = (
        a1 * dre_xx
        + a2 * dre_x
        + a3 * dre_yy
        + a4 * dre_y
        - (b1 * dim_xx + b2 * dim_x + b3 * dim_yy + b4 * dim_y)
        + eps * k0**2 * re_e
    )
    loss_im = (
        a1 * dim_xx
        + a2 * dim_x
        + a3 * dim_yy
        + a4 * dim_y
        + (b1 * dre_xx + b2 * dre_x + b3 * dre_yy + b4 * dre_y)
        + eps * k0**2 * im_e
    )
    return loss_re, loss_im


def pde_inverse(inputs, outputs, radius, eps_pred, k0, length, dpml, omega, sigma0_value):
    transformed = transform_inverse(inputs, outputs, radius)
    re_e = transformed[:, 0:1]
    im_e = transformed[:, 1:2]

    dre_dx = torch.autograd.grad(re_e, inputs, torch.ones_like(re_e), create_graph=True)[0]
    dre_xx = torch.autograd.grad(dre_dx[:, 0:1], inputs, torch.ones_like(dre_dx[:, 0:1]), create_graph=True)[0][:, 0:1]
    dre_yy = torch.autograd.grad(dre_dx[:, 1:2], inputs, torch.ones_like(dre_dx[:, 1:2]), create_graph=True)[0][:, 1:2]

    dim_dx = torch.autograd.grad(im_e, inputs, torch.ones_like(im_e), create_graph=True)[0]
    dim_xx = torch.autograd.grad(dim_dx[:, 0:1], inputs, torch.ones_like(dim_dx[:, 0:1]), create_graph=True)[0][:, 0:1]
    dim_yy = torch.autograd.grad(dim_dx[:, 1:2], inputs, torch.ones_like(dim_dx[:, 1:2]), create_graph=True)[0][:, 1:2]

    a1, _, a2, _ = pml_inverse(inputs, length, dpml, omega, sigma0_value)
    loss_re = a1 * dre_xx + a2 * dre_yy + eps_pred * k0**2 * re_e
    loss_im = a1 * dim_xx + a2 * dim_yy + eps_pred * k0**2 * im_e
    return loss_re, loss_im


def boundary_condition_loss(net, boundary_points, radius, k0):
    outputs = net(boundary_points)
    field = transform_inverse(boundary_points, outputs, radius)
    re_e = field[:, 0:1]
    im_e = field[:, 1:2]
    normals = boundary_points / torch.norm(boundary_points, dim=1, keepdim=True)

    grad_re = torch.autograd.grad(re_e, boundary_points, torch.ones_like(re_e), create_graph=True)[0]
    grad_im = torch.autograd.grad(im_e, boundary_points, torch.ones_like(im_e), create_graph=True)[0]
    dre_dn = torch.sum(grad_re * normals, dim=1, keepdim=True)
    dim_dn = torch.sum(grad_im * normals, dim=1, keepdim=True)

    bc_re = dre_dn - k0 * im_e
    bc_im = dim_dn + k0 * re_e
    return torch.mean(bc_re**2) + torch.mean(bc_im**2)
