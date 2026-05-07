import numpy as np
import torch

from .models import GPT
from .optimizer import GradDescent
from .precomp import pt_nu_p_xx


def offline_generation(nu_train, xt_size, ic_size, bc_size, ic_u, bc_u, out, out_x, out_t, out_xx, out_ic, out_bc, f_hat, epochs_gpt, lr_gpt, neurons):
    neuron_params = neurons
    neuron_cnt = out.shape[1]
    largest_case = 0
    largest_loss = 0

    if neuron_cnt == 1:
        c_initial = torch.ones(1, device=out.device)
    else:
        dist = np.zeros(neuron_cnt)
        c_initial = torch.zeros(neuron_cnt, device=out.device)

    for nu in nu_train:
        if neuron_cnt != 1:
            if nu in neuron_params:
                index = np.where(nu == neuron_params)
                c_initial[:] = 0
                c_initial[index] = 1
            else:
                for k, nu_neuron in enumerate(neuron_params):
                    dist[k] = np.abs(nu_neuron - nu)
                d = np.argsort(dist)
                first, second = d[0], d[1]
                a, b = dist[first], dist[second]
                bottom = a + b
                c_initial[:] = 0
                c_initial[first] = b / bottom
                c_initial[second] = a / bottom

        pt_term = pt_nu_p_xx(nu, out_t, out_xx)
        gd = GradDescent(nu, xt_size, ic_size, bc_size, ic_u, out, out_ic, out_bc, out_x, pt_term, lr_gpt)
        gpt_nn = GPT(nu, out, out_ic, out_bc, out_x, pt_term, ic_u, bc_u, f_hat).to(out.device)

        c = c_initial
        c_reshaped = c.unsqueeze(1)
        loss = gpt_nn.loss(c_reshaped)
        for _ in range(1, epochs_gpt + 1):
            if loss < largest_loss:
                break
            c = gd.update(c, c_reshaped)
            c_reshaped = c.unsqueeze(1)
            loss = gpt_nn.loss(c_reshaped)

        if loss > largest_loss:
            largest_case = nu
            largest_loss = loss

    return largest_loss, largest_case


def pinn_train(pinn, xt_resid, ic_xt, ic_u, bc_xt, bc_u, f_hat, epochs_pinn, lr_pinn, tol):
    optimizer = torch.optim.Adam(pinn.parameters(), lr=lr_pinn)
    for _ in range(1, epochs_pinn + 1):
        loss = pinn.loss(xt_resid, ic_xt, ic_u, bc_xt, bc_u, f_hat)
        if loss < tol:
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return float(loss.item())
