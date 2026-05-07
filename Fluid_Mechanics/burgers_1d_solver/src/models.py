import torch
import torch.autograd as autograd
import torch.nn as nn


class GPT(nn.Module):
    def __init__(self, nu, out, out_ic, out_bc, out_x, pt_nu_p_xx_term, ic_u, bc_u, fhat):
        super().__init__()
        self.nu = nu
        self.loss_function = nn.MSELoss(reduction="mean")

        self.ic_u = ic_u
        self.bc_u = bc_u
        self.f_hat = fhat
        self.pt_nu_p_xx_term = pt_nu_p_xx_term
        self.out_ic = out_ic
        self.out_bc = out_bc
        self.out = out
        self.out_x = out_x

    def lossR(self, c):
        u = torch.matmul(self.out, c)
        ux = torch.matmul(self.out_x, c)
        u_ux = torch.mul(u, ux)
        ut_vuxx = torch.matmul(self.pt_nu_p_xx_term, c)
        f = torch.add(ut_vuxx, u_ux)
        return self.loss_function(f, self.f_hat)

    def lossBC(self, c):
        return self.loss_function(torch.matmul(self.out_bc, c), self.bc_u)

    def lossIC(self, c):
        return self.loss_function(torch.matmul(self.out_ic, c), self.ic_u)

    def loss(self, c):
        return self.lossR(c) + self.lossIC(c) + self.lossBC(c)


class NN(nn.Module):
    def __init__(self, layers, nu, seed=1234):
        super().__init__()
        torch.manual_seed(seed)

        self.layers = layers
        self.nu = nu
        self.loss_function = nn.MSELoss(reduction="mean")

        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linears[i].weight.data)
            nn.init.zeros_(self.linears[i].bias.data)

        self.activation = nn.Tanh()

    def forward(self, x):
        a = x
        for i in range(0, len(self.layers) - 2):
            a = self.activation(self.linears[i](a))
        return self.linears[-1](a)

    def lossR(self, xt, f_hat):
        device = xt.device
        u = self.forward(xt)

        u_xt = autograd.grad(u, xt, torch.ones_like(u).to(device), create_graph=True)[0]
        u_xx_tt = autograd.grad(u_xt, xt, torch.ones_like(u_xt).to(device), create_graph=True)[0]

        u_xx = u_xx_tt[:, 0].unsqueeze(1)
        u_x = u_xt[:, 0].unsqueeze(1)
        u_t = u_xt[:, 1].unsqueeze(1)

        f = torch.add(u_t, torch.add(torch.mul(u, u_x), torch.mul(-self.nu, u_xx)))
        return self.loss_function(f, f_hat)

    def lossICBC(self, icbc_xt, icbc_u):
        return self.loss_function(self.forward(icbc_xt), icbc_u)

    def loss(self, xt_resid, ic_xt, ic_u, bc_xt, bc_u, f_hat):
        return self.lossR(xt_resid, f_hat) + self.lossICBC(ic_xt, ic_u) + self.lossICBC(bc_xt, bc_u)
