import torch


class ThermoElasticPINNLoss:
    def __init__(
        self,
        rho, Cp, k, E, nu, alpha, T0,
        w_heat=1.0, w_elastic=0.1, w_ic=1.0, w_reg=1e-4,
        w_bc=1.0,
        bc_outer_type: str = "dirichlet",
        bc_outer_value: float = 273.15,
        bc_hole_type: str = "neumann",
        bc_hole_value: float = 0.0,
    ):
        self.rho = float(rho)
        self.Cp = float(Cp)
        self.k = float(k)
        self.E = float(E)
        self.nu = float(nu)
        self.alpha = float(alpha)
        self.T0 = float(T0)

        self.w_heat = float(w_heat)
        self.w_elastic = float(w_elastic)
        self.w_ic = float(w_ic)
        self.w_reg = float(w_reg)

        self.w_bc = float(w_bc)
        self.bc_outer_type = bc_outer_type
        self.bc_outer_value = float(bc_outer_value)
        self.bc_hole_type = bc_hole_type
        self.bc_hole_value = float(bc_hole_value)

    def compute_derivatives(self, T, inputs):
        """
        Robust derivatives for T w.r.t. inputs in the last dim: [x, y, z, t]
        inputs: (N, time_steps, 4)
        T:      (N, time_steps, 1)
        Returns: T_t, T_xx, T_yy, T_zz, T_x, T_y, T_z  (all shapes like T)
        """
        # Ensure autograd tracks inputs
        if not inputs.requires_grad:
            inputs = inputs.clone().detach().requires_grad_(True)

        grads = torch.autograd.grad(
            outputs=T,
            inputs=inputs,
            grad_outputs=torch.ones_like(T),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]

        if grads is None:
            grads = torch.zeros_like(inputs)

        # last dim: [x,y,z,t]
        T_x = grads[..., 0:1]
        T_y = grads[..., 1:2]
        T_z = grads[..., 2:3]
        T_t = grads[..., 3:4] if inputs.shape[-1] > 3 else torch.zeros_like(T)

        def second_derivative(first_deriv, dim_index: int):
            g2 = torch.autograd.grad(
                outputs=first_deriv,
                inputs=inputs,
                grad_outputs=torch.ones_like(first_deriv),
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if g2 is None:
                return torch.zeros_like(first_deriv)
            return g2[..., dim_index:dim_index + 1]

        T_xx = second_derivative(T_x, 0)
        T_yy = second_derivative(T_y, 1)
        T_zz = second_derivative(T_z, 2)

        return T_t, T_xx, T_yy, T_zz, T_x, T_y, T_z

    def compute_loss(self, model, res_inputs, bc_inputs, bc_normals, bc_tag, ic_inputs, ic_T_true):
        """
        res_inputs, bc_inputs, ic_inputs: (N, time_steps, 4)
        ic_T_true: (N, 1)
        """
        # -------------------------
        # Residual loss (heat eq.)
        # -------------------------
        T_res = model(res_inputs)  # (N, time_steps, 1)

        T_t, T_xx, T_yy, T_zz, T_x, T_y, T_z = self.compute_derivatives(T_res, res_inputs)

        heat_residual = self.rho * self.Cp * T_t - self.k * (T_xx + T_yy + T_zz)
        loss_heat = torch.mean(heat_residual ** 2)

        # -----------------------------------------
        # Elastic equilibrium constraint (simplified)
        # -----------------------------------------
        sigma_th = self.E * self.alpha * (T_res - self.T0) / (1 - 2 * self.nu)

        # robust grads w.r.t. inputs (avoid "not used in graph")
        if not res_inputs.requires_grad:
            res_inputs_el = res_inputs.clone().detach().requires_grad_(True)
            sigma_th_el = model(res_inputs_el)
            sigma_th_el = self.E * self.alpha * (sigma_th_el - self.T0) / (1 - 2 * self.nu)
        else:
            res_inputs_el = res_inputs
            sigma_th_el = sigma_th

        g_sigma = torch.autograd.grad(
            outputs=sigma_th_el,
            inputs=res_inputs_el,
            grad_outputs=torch.ones_like(sigma_th_el),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]

        if g_sigma is None:
            g_sigma = torch.zeros_like(res_inputs_el)

        sigma_x = g_sigma[..., 0:1]
        sigma_y = g_sigma[..., 1:2]
        sigma_z = g_sigma[..., 2:3]

        elastic_residual = sigma_x + sigma_y + sigma_z
        loss_elastic = torch.mean(elastic_residual ** 2)

        # -------------------------
        # Initial condition loss
        # -------------------------
        T_ic = model(ic_inputs)[:, 0, :]  # first time step
        loss_ic = torch.mean((T_ic - ic_T_true) ** 2)

        # -------------------------
        # Regularization (optional)
        # -------------------------
        loss_reg = torch.mean(T_x ** 2 + T_y ** 2 + T_z ** 2)

        # -------------------------
        # Boundary condition loss
        # -------------------------
        loss_bc = torch.tensor(0.0, device=res_inputs.device)

        if bc_normals is not None:
            # Ensure bc_inputs requires grad for Neumann
            if not bc_inputs.requires_grad:
                bc_inputs_g = bc_inputs.clone().detach().requires_grad_(True)
                T_bc = model(bc_inputs_g)  # (Nbc, time_steps, 1)
            else:
                bc_inputs_g = bc_inputs
                T_bc = model(bc_inputs)

            # grads of T_bc w.r.t bc_inputs_g
            gT = torch.autograd.grad(
                outputs=T_bc,
                inputs=bc_inputs_g,
                grad_outputs=torch.ones_like(T_bc),
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if gT is None:
                gT = torch.zeros_like(bc_inputs_g)

            T_xb = gT[..., 0:1]
            T_yb = gT[..., 1:2]
            T_zb = gT[..., 2:3]

            n = bc_normals.to(bc_inputs.device).unsqueeze(1)  # (Nbc,1,3)
            dTdn = T_xb * n[..., 0:1] + T_yb * n[..., 1:2] + T_zb * n[..., 2:3]

            if bc_tag is None:
                bc_tag = torch.zeros((bc_inputs.shape[0],), device=bc_inputs.device, dtype=torch.long)
            bc_tag = bc_tag.to(bc_inputs.device).view(-1, 1, 1)

            w_outer = (bc_tag == 0).float().expand_as(T_bc)
            w_hole = (bc_tag == 1).float().expand_as(T_bc)

            def masked_mse(err, w):
                return (err * w).sum() / (w.sum() + 1e-12)

            # outer BC
            if self.bc_outer_type.lower() == "dirichlet":
                err_outer = (T_bc - self.bc_outer_value) ** 2
            elif self.bc_outer_type.lower() == "neumann":
                err_outer = (dTdn - self.bc_outer_value) ** 2
            else:
                raise ValueError(f"Unknown bc_outer_type: {self.bc_outer_type}")
            loss_outer = masked_mse(err_outer, w_outer)

            # hole BC
            if self.bc_hole_type.lower() == "dirichlet":
                err_hole = (T_bc - self.bc_hole_value) ** 2
            elif self.bc_hole_type.lower() == "neumann":
                err_hole = (dTdn - self.bc_hole_value) ** 2
            else:
                raise ValueError(f"Unknown bc_hole_type: {self.bc_hole_type}")
            loss_hole = masked_mse(err_hole, w_hole)

            loss_bc = loss_outer + loss_hole

        total_loss = (self.w_heat * loss_heat +
                      self.w_elastic * loss_elastic +
                      self.w_ic * loss_ic +
                      self.w_reg * loss_reg +
                      self.w_bc * loss_bc)

        return (total_loss,
                loss_heat.detach().item(),
                loss_elastic.detach().item(),
                loss_ic.detach().item(),
                loss_reg.detach().item(),
                loss_bc.detach().item())