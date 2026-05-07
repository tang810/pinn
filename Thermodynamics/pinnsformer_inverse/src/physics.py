import torch
import torch.nn as nn

class ThermoElasticPINNLoss(nn.Module):
    def __init__(self, args, trainable_params=None, T_mean=None, T_std=None):
        """
        热弹性耦合 PINN 损失函数（支持反演）
        :param args: 配置对象，包含所有物理常数和损失权重
        :param trainable_params: 需要反演的参数名列表，如 ['k', 'alpha_t', 'q_bc_x1']
        :param T_mean, T_std: 温度归一化参数（可选）
        """
        super().__init__()
        if trainable_params is None:
            trainable_params = []

        # ---------- 注册可训练参数（Parameter）----------
        for name in trainable_params:
            if hasattr(args, name):
                init_val = getattr(args, name)
                self.register_parameter(name, nn.Parameter(torch.tensor(init_val, dtype=torch.float32)))
            else:
                raise ValueError(f"Unknown parameter name: {name}")

        # ---------- 注册所有物理常数（不可训练 buffer）----------
        # 定义所有可能用到的常数（包括可能被反演的，但已注册过则跳过）
        all_constants = {
            'k': args.k,
            'rho': args.rho,
            'Cp': args.c_p,          # 注意命名与 forward 中一致
            'T0': args.t_0,
            'Q_source': args.q_sum,
            'E': args.e,
            'nu': args.nu,
            'alpha_t': args.alpha_t,
            'T_ref': args.t_ref,
            'eps': args.eps,
            'sigma': args.sigma,
            'T_amb': args.t_amb,
            'q_bc_x1': args.q_bc_x1,
        }
        for name, val in all_constants.items():
            if not hasattr(self, name):   # 避免覆盖已注册的 Parameter
                self.register_buffer(name, torch.tensor(val, dtype=torch.float32))

        # ---------- 损失权重（普通属性，不可训练）----------
        self.lambda_pde = args.lambda_pde
        self.lambda_elastic = args.lambda_elastic
        self.lambda_ic = args.lambda_ic
        self.lambda_ic_disp = args.lambda_ic_disp
        self.lambda_bc = args.lambda_bc
        self.lambda_bc_disp = args.lambda_bc_disp
        self.lambda_data = args.lambda_data
        self.lambda_heatflux = args.lambda_heatflux
        self.lambda_rad = args.lambda_rad

        # ---------- 温度归一化参数 ----------
        self.T_mean = T_mean
        self.T_std = T_std

    # =========================================================
    # 安全梯度（防 None / NaN / Inf）
    # =========================================================
    def _grad(self, y, x):
        g = torch.autograd.grad(
            y,
            x,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]

        if g is None:
            return torch.zeros_like(x)

        g = torch.nan_to_num(g, nan=0.0, posinf=1e6, neginf=-1e6)
        return g

    # =========================================================
    # 前向计算损失
    # =========================================================
    def forward(
        self,
        model,
        res_tensor=None,
        ic_tensor=None,
        bc_tensors=None,
        data_tensor=None,
        T_data=None,
    ):
        device = next(model.parameters()).device

        # 初始化各项损失
        loss_heat = torch.zeros(1, device=device)
        loss_elastic = torch.zeros(1, device=device)
        loss_ic = torch.zeros(1, device=device)
        loss_ic_disp = torch.zeros(1, device=device)
        loss_bc_temp = torch.zeros(1, device=device)
        loss_bc_disp = torch.zeros(1, device=device)
        loss_data = torch.zeros(1, device=device)
        loss_heatflux = torch.zeros(1, device=device)
        loss_rad = torch.zeros(1, device=device)

        # ---------- 计算依赖于可训练参数的派生量（每次 forward 重新计算）----------
        mu = self.E / (2.0 * (1.0 + self.nu))
        lmbda = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        beta = (3.0 * lmbda + 2.0 * mu) * self.alpha_t

        # =====================================================
        # PDE 残差
        # =====================================================
        if res_tensor is not None and res_tensor.numel() > 0:
            res_tensor = res_tensor.clone().detach().requires_grad_(True)

            x = res_tensor[:, 0:1]
            y = res_tensor[:, 1:2]
            z = res_tensor[:, 2:3]
            t = res_tensor[:, 3:4]

            out = model(x, y, z, t)
            T = out[:, 0:1]
            u = out[:, 1:2]
            v = out[:, 2:3]
            w = out[:, 3:4]

            # 热方程
            T_t = self._grad(T, t)
            T_x = self._grad(T, x)
            T_y = self._grad(T, y)
            T_z = self._grad(T, z)

            T_xx = self._grad(T_x, x)
            T_yy = self._grad(T_y, y)
            T_zz = self._grad(T_z, z)

            heat_res = (
                self.rho * self.Cp * T_t
                - self.k * (T_xx + T_yy + T_zz)
                - self.Q_source
            )

            heat_res = torch.nan_to_num(heat_res, nan=0.0, posinf=1e6, neginf=-1e6)
            loss_heat = torch.mean(heat_res ** 2)

            # 弹性方程
            if self.lambda_elastic > 0:
                u_x = self._grad(u, x)
                u_y = self._grad(u, y)
                u_z = self._grad(u, z)

                v_x = self._grad(v, x)
                v_y = self._grad(v, y)
                v_z = self._grad(v, z)

                w_x = self._grad(w, x)
                w_y = self._grad(w, y)
                w_z = self._grad(w, z)

                div_u = u_x + v_y + w_z

                div_u_x = self._grad(div_u, x)
                div_u_y = self._grad(div_u, y)
                div_u_z = self._grad(div_u, z)

                lap_u = self._grad(u_x, x) + self._grad(u_y, y) + self._grad(u_z, z)
                lap_v = self._grad(v_x, x) + self._grad(v_y, y) + self._grad(v_z, z)
                lap_w = self._grad(w_x, x) + self._grad(w_y, y) + self._grad(w_z, z)

                res_x = (lmbda + mu) * div_u_x + mu * lap_u - beta * T_x
                res_y = (lmbda + mu) * div_u_y + mu * lap_v - beta * T_y
                res_z = (lmbda + mu) * div_u_z + mu * lap_w - beta * T_z

                elastic_res = res_x ** 2 + res_y ** 2 + res_z ** 2
                elastic_res = torch.nan_to_num(elastic_res, nan=0.0, posinf=1e6, neginf=-1e6)

                loss_elastic = torch.mean(elastic_res)

        # =====================================================
        # 初始条件
        # =====================================================
        if ic_tensor is not None and ic_tensor.numel() > 0:
            x_ic = ic_tensor[:, 0:1]
            y_ic = ic_tensor[:, 1:2]
            z_ic = ic_tensor[:, 2:3]
            t_ic = ic_tensor[:, 3:4]

            out_ic = model(x_ic, y_ic, z_ic, t_ic)

            T_ic_pred = out_ic[:, 0:1]
            u_ic_pred = out_ic[:, 1:2]
            v_ic_pred = out_ic[:, 2:3]
            w_ic_pred = out_ic[:, 3:4]

            loss_ic = torch.mean((T_ic_pred - self.T0) ** 2)
            loss_ic_disp = torch.mean(
                u_ic_pred ** 2 + v_ic_pred ** 2 + w_ic_pred ** 2
            )

        # =====================================================
        # 边界条件
        # =====================================================
        if bc_tensors is not None and isinstance(bc_tensors, dict):
            for face_name, bc_pts in bc_tensors.items():
                if bc_pts.numel() == 0:
                    continue
                bc_pts = bc_pts.clone().detach().requires_grad_(True)
                x_bc = bc_pts[:, 0:1]
                y_bc = bc_pts[:, 1:2]
                z_bc = bc_pts[:, 2:3]
                t_bc = bc_pts[:, 3:4]

                out_bc = model(x_bc, y_bc, z_bc, t_bc)

                T_bc_pred = out_bc[:, 0:1]
                u_bc_pred = out_bc[:, 1:2]
                v_bc_pred = out_bc[:, 2:3]
                w_bc_pred = out_bc[:, 3:4]

                if face_name == 'x1':   # x=1 面施加热流边界
                    T_x = self._grad(T_bc_pred, x_bc)
                    # 直接使用 self.q_bc_x1（已注册为 buffer 或 Parameter）
                    heat_flux_res = self.k * T_x - self.q_bc_x1
                    loss_heatflux = loss_heatflux + torch.mean(heat_flux_res ** 2)

                else:
                    # 其他面：辐射边界（根据面确定法向导数的符号）
                    if face_name == 'x0':
                        T_n = self._grad(T_bc_pred, x_bc)   # 法向为 -x
                        sign = 1.0
                    elif face_name == 'y0':
                        T_n = self._grad(T_bc_pred, y_bc)   # 法向为 -y
                        sign = 1.0
                    elif face_name == 'y1':
                        T_n = self._grad(T_bc_pred, y_bc)   # 法向为 +y
                        sign = -1.0
                    elif face_name == 'z0':
                        T_n = self._grad(T_bc_pred, z_bc)   # 法向为 -z
                        sign = 1.0
                    elif face_name == 'z1':
                        T_n = self._grad(T_bc_pred, z_bc)   # 法向为 +z
                        sign = -1.0
                    else:
                        continue

                    # 辐射残差：sign * k * T_n + εσ (T_amb⁴ - T⁴) = 0
                    rad_res = sign * self.k * T_n + self.eps * self.sigma * (self.T_amb**4 - T_bc_pred**4)
                    loss_rad = loss_rad + torch.mean(rad_res ** 2)

                    if self.lambda_bc_disp > 0:
                        loss_bc_disp = loss_bc_disp + torch.mean(
                            u_bc_pred ** 2 + v_bc_pred ** 2 + w_bc_pred ** 2
                        )

        # =====================================================
        # 数据损失
        # =====================================================
        if data_tensor is not None and T_data is not None and data_tensor.numel() > 0:
            x_d = data_tensor[:, 0:1]
            y_d = data_tensor[:, 1:2]
            z_d = data_tensor[:, 2:3]
            t_d = data_tensor[:, 3:4]

            T_pred = model(x_d, y_d, z_d, t_d)[:, 0:1]
            if self.T_mean is not None and self.T_std is not None:
                T_pred_norm = (T_pred - self.T_mean) / self.T_std
                loss_data = torch.mean((T_pred_norm - T_data) ** 2)
            else:
                loss_data = torch.mean((T_pred - T_data) ** 2)

        # =====================================================
        # 加权总损失
        # =====================================================
        total_loss = (
            self.lambda_pde * loss_heat
            + self.lambda_elastic * loss_elastic
            + self.lambda_ic * loss_ic
            + self.lambda_ic_disp * loss_ic_disp
            + self.lambda_bc * loss_bc_temp
            + self.lambda_bc_disp * loss_bc_disp
            + self.lambda_data * loss_data
            + self.lambda_heatflux * loss_heatflux
            + self.lambda_rad * loss_rad
        )

        total_loss = torch.nan_to_num(total_loss, nan=1e6, posinf=1e6, neginf=-1e6)

        return {
            "total": total_loss,
            "heat": loss_heat.detach(),
            "elastic": loss_elastic.detach(),
            "ic": loss_ic.detach(),
            "ic_disp": loss_ic_disp.detach(),
            "bc_temp": loss_bc_temp.detach(),
            "bc_disp": loss_bc_disp.detach(),
            "data": loss_data.detach(),
            "heatflux": loss_heatflux.detach(),
            "rad": loss_rad.detach(),
        }