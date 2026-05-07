import numpy as np
import torch
import time

from .models import GPT, NN
from .optimizer import GradDescent
from .precomp import pt_nu_p_xx


def gpt_test(nu_train, xt_size, ic_size, bc_size, ic_u, bc_u, out, out_x, out_t, out_xx, out_ic, out_bc, f_hat, epochs_gpt, lr_gpt, neurons, out_test):
    times = np.zeros(len(nu_train))
    gpt_soln = np.zeros((out_test.shape[0], len(nu_train)))

    neuron_cnt = out.shape[1]
    neuron_params = neurons

    dist = np.zeros(neuron_cnt)
    c_initial = torch.zeros(neuron_cnt, device=out.device)

    for i, nu in enumerate(nu_train):
        t_start = time.time()

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

        c = c_initial
        c_reshaped = c.unsqueeze(1)
        for _ in range(1, epochs_gpt + 1):
            c = gd.update(c, c_reshaped)
            c_reshaped = c.unsqueeze(1)

        soln = torch.matmul(out_test, c_reshaped)
        t_end = time.time()

        times[i] = (t_end - t_start) / 3600 if i == 0 else (t_end - t_start) / 3600 + times[i - 1]
        gpt_soln[:, i][:, None] = soln.detach().cpu().numpy()

    return times, gpt_soln


def gpt_test_loss(nu_train, xt_size, ic_size, bc_size, ic_u, bc_u, out, out_x, out_t, out_xx, out_ic, out_bc, f_hat, epochs_gpt, lr_gpt, neurons):
    losses = np.zeros((11, len(nu_train)))

    neuron_cnt = out.shape[1]
    neuron_params = neurons

    dist = np.zeros(neuron_cnt)
    c_initial = torch.zeros(neuron_cnt, device=out.device)

    for i, nu in enumerate(nu_train):
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
        losses[0, i] = gpt_nn.loss(c_reshaped)
        for j in range(1, epochs_gpt + 1):
            c = gd.update(c, c_reshaped)
            c_reshaped = c.unsqueeze(1)
            if j % 500 == 0:
                losses[int(j / 500), i] = gpt_nn.loss(c_reshaped)

    return losses


def pinn_test(b_test, layers_pinn, xt_resid, ic_xt, ic_u, bc_xt, bc_u, f_hat, epochs_pinn, lr_pinn, xt_test, tol):
    times = np.zeros(len(b_test))
    pinn_soln = np.zeros((xt_test.shape[0], len(b_test)))

    for i, b_param in enumerate(b_test):
        t_start = time.time()
        pinn = NN(layers_pinn, b_param).to(xt_resid.device)
        optimizer = torch.optim.Adam(pinn.parameters(), lr=lr_pinn)

        for _ in range(1, epochs_pinn + 1):
            loss = pinn.loss(xt_resid, ic_xt, ic_u, bc_xt, bc_u, f_hat)
            if loss < tol:
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        soln = pinn(xt_test)

        t_end = time.time()
        times[i] = (t_end - t_start) / 3600 if i == 0 else (t_end - t_start) / 3600 + times[i - 1]
        pinn_soln[:, i][:, None] = soln.detach().cpu().numpy()

    return times, pinn_soln


def pinn_test_loss(b_test, layers_pinn, xt_resid, ic_xt, ic_u, bc_xt, bc_u, f_hat, epochs_pinn, lr_pinn, tol):
    losses = np.zeros((61, len(b_test)))

    for i, b_param in enumerate(b_test):
        pinn = NN(layers_pinn, b_param).to(xt_resid.device)
        optimizer = torch.optim.Adam(pinn.parameters(), lr=lr_pinn)

        losses[0, i] = pinn.loss(xt_resid, ic_xt, ic_u, bc_xt, bc_u, f_hat).item()

        for j in range(1, epochs_pinn + 1):
            loss = pinn.loss(xt_resid, ic_xt, ic_u, bc_xt, bc_u, f_hat)

            if (j % 1000 == 0) or (j == epochs_pinn):
                losses[int(j / 1000), i] = loss.item()

            if loss < tol:
                losses[np.where(losses[:, i] == 0)[0][0], i] = loss.item()
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return losses
