import torch

class HeatEquationPINNLoss:
    def __init__(self, args):
        self.k = args.k
        self.rho = args.rho
        self.C_p = args.c_p
        self.eps = args.eps
        self.sigma = args.sigma
        self.T_amb = args.t_amb
        self.T_0 = args.t_0
        self.q_sum = args.q_sum

        self.lambda_pde = args.lambda_pde
        self.lambda_bc = args.lambda_bc
        self.lambda_ic = args.lambda_ic
        self.lambda_reg = args.lambda_reg

    def _compute_derivatives(self, model, x, y, z, t):
        T_physical = model(x, y, z, t)

        grad_params = {'retain_graph': True, 'create_graph': True}
        ones = torch.ones_like(T_physical)

        T_t = torch.autograd.grad(T_physical, t, grad_outputs=ones, **grad_params)[0]
        T_x = torch.autograd.grad(T_physical, x, grad_outputs=ones, **grad_params)[0]
        T_y = torch.autograd.grad(T_physical, y, grad_outputs=ones, **grad_params)[0]
        T_z = torch.autograd.grad(T_physical, z, grad_outputs=ones, **grad_params)[0]

        T_xx = torch.autograd.grad(T_x, x, grad_outputs=ones, **grad_params)[0]
        T_yy = torch.autograd.grad(T_y, y, grad_outputs=ones, **grad_params)[0]
        T_zz = torch.autograd.grad(T_z, z, grad_outputs=ones, **grad_params)[0]

        return T_physical, T_t, T_x, T_y, T_z, T_xx, T_yy, T_zz

    def compute_loss(self, model, res_tensor, ic_tensor, bc_tensors):
        x_res, y_res, z_res, t_res = (res_tensor[..., i:i+1] for i in range(4))
        _, T_t_res, T_x_res, T_y_res, T_z_res, T_xx_res, T_yy_res, T_zz_res = self._compute_derivatives(
            model, x_res, y_res, z_res, t_res
        )
        pde_residual = T_t_res - (self.k / (self.rho * self.C_p)) * (T_xx_res + T_yy_res + T_zz_res)
        loss_pde = torch.mean(pde_residual ** 2)

        x_ic, y_ic, z_ic, t_ic = (ic_tensor[..., i:i+1] for i in range(4))
        pred_ic, *_ = self._compute_derivatives(model, x_ic, y_ic, z_ic, t_ic)
        loss_ic = torch.mean((pred_ic.squeeze() - self.T_0) ** 2)

        def get_bc_loss(boundary_key, axis_derivative_index):
            x_bc, y_bc, z_bc, t_bc = (bc_tensors[boundary_key][..., i:i+1] for i in range(4))
            outputs = self._compute_derivatives(model, x_bc, y_bc, z_bc, t_bc)
            pred_bc = outputs[0]
            deriv_bc = outputs[2 + axis_derivative_index]
            
            rad_flux = self.eps * self.sigma * (self.T_amb ** 4 - pred_bc ** 4)
            
            if boundary_key == 'x1':
                residual = self.k * deriv_bc - self.q_sum
            elif boundary_key in ['y1', 'z1']:
                residual = rad_flux - self.k * deriv_bc
            else:
                residual = rad_flux + self.k * deriv_bc
            return torch.mean(residual ** 2)
        
        loss_bc1 = get_bc_loss('x1', 0)
        loss_bc2 = get_bc_loss('x0', 0)
        loss_bc3 = get_bc_loss('y0', 1)
        loss_bc4 = get_bc_loss('y1', 1)
        loss_bc5 = get_bc_loss('z0', 2)
        loss_bc6 = get_bc_loss('z1', 2)

        bc_losses_list = [loss_bc1, loss_bc2, loss_bc3, loss_bc4, loss_bc5, loss_bc6]
        total_loss_bc = sum(bc_losses_list)

        # (dT/dx)^2 + (dT/dy)^2 + (dT/dz)^2
        gradient_norm_squared = T_x_res**2 + T_y_res**2 + T_z_res**2
        loss_reg = torch.mean(gradient_norm_squared)

        total_loss_physics = self.lambda_pde * loss_pde + self.lambda_bc * total_loss_bc + self.lambda_ic * loss_ic
        total_loss = total_loss_physics + self.lambda_reg * loss_reg
        
        return total_loss, loss_pde.item(), total_loss_bc.item(), loss_ic.item(), loss_reg.item()
