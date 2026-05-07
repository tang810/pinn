import torch


class GradDescent:
    def __init__(self, nu, xt_size, ic_size, bc_size, ic_u, out, out_ic, out_bc, out_x, pt_nu_p_xx_term, lr_gpt):
        self.nu = nu
        self.fac1 = 2 / xt_size
        self.fac2 = 2 / bc_size
        self.fac3 = 2 / ic_size

        self.out = out
        self.out_bc = out_bc
        self.out_ic = out_ic
        self.out_x = out_x
        self.pt_nu_p_xx_term = pt_nu_p_xx_term
        self.ic_u = ic_u
        self.lr_gpt = lr_gpt

    def update(self, c, c_reshaped):
        u = torch.matmul(self.out, c_reshaped)
        ux = torch.matmul(self.out_x, c_reshaped)
        u_ux = torch.mul(u, ux)

        ut_vuxx = torch.matmul(self.pt_nu_p_xx_term, c_reshaped)
        first_product = torch.add(ut_vuxx, u_ux)

        u_px = torch.mul(u, self.out_x)
        p_ux = torch.mul(self.out, ux)
        second_product = torch.add(torch.add(u_px, p_ux), self.pt_nu_p_xx_term)

        grad_list = torch.mul(self.fac1, torch.sum(torch.mul(first_product, second_product), axis=0))

        bc_term = torch.matmul(self.out_bc, c_reshaped)
        ic_term = torch.sub(torch.matmul(self.out_ic, c_reshaped), self.ic_u)

        grad_list += torch.mul(self.fac2, torch.sum(torch.mul(bc_term, self.out_bc), axis=0))
        grad_list += torch.mul(self.fac3, torch.sum(torch.mul(ic_term, self.out_ic), axis=0))

        return torch.sub(c, torch.mul(self.lr_gpt, grad_list))
