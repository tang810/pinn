import torch
import torch.autograd as autograd


def derivatives(xt, out):
    device = xt.device
    out_xt = autograd.grad(out, xt, torch.ones_like(out).to(device), create_graph=True)[0]
    out_xx_tt = autograd.grad(out_xt, xt, torch.ones_like(out_xt).to(device))[0]

    out_x = out_xt[:, 0].detach().unsqueeze(1)
    out_t = out_xt[:, 1].detach().unsqueeze(1)
    out_xx = out_xx_tt[:, 0].unsqueeze(1)
    return out_t, out_x, out_xx


def pt_nu_p_xx(nu, out_t, out_xx):
    return torch.add(torch.mul(-nu, out_xx), out_t)


def inputs(pinn, xt, out, out_t, out_x, out_xx, out_ic, out_bc, ic_xt, bc_xt, i, out_test, xt_test, xt_size, num_mag, idx_list):
    device = xt.device
    end = i + 1

    p = pinn(xt)
    out[:, i][:, None] = p.detach()
    p_t, p_x, p_xx = derivatives(xt, p)
    out_x[:, i][:, None] = p_x
    out_xx[:, i][:, None] = p_xx
    out_t[:, i][:, None] = p_t

    _, index = torch.sort(torch.abs(p_xx.view(-1)))
    largest_indices = torch.LongTensor(index[xt_size - num_mag : xt_size].cpu())
    idx_list[i] = largest_indices

    out_t[:, i][:, None].put_(idx_list[i].to(device), torch.zeros(num_mag).to(device))
    out_x[:, i][:, None].put_(idx_list[i].to(device), torch.zeros(num_mag).to(device))
    out_xx[:, i][:, None].put_(idx_list[i].to(device), torch.zeros(num_mag).to(device))
    out[:, i][:, None].put_(idx_list[i].to(device), torch.zeros(num_mag).to(device))

    out_ic[:, i][:, None] = pinn(ic_xt).detach()
    out_bc[:, i][:, None] = pinn(bc_xt).detach()
    out_test[:, i][:, None] = pinn(xt_test).detach()

    return out[:, 0:end], out_x[:, 0:end], out_t[:, 0:end], out_xx[:, 0:end], out_ic[:, 0:end], out_bc[:, 0:end]
