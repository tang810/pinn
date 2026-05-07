# src/loss.py
import torch
from .physics import Curl


# =========================================================
# 材料参数
# =========================================================
class MaterialProperties:
    @staticmethod
    def epsilon_r(x, interface_position: float = 0.0):
        transition_width = 0.05
        sigma = 1.0 / (1.0 + torch.exp(
            -(x[:, 0:1] - interface_position) / transition_width
        ))
        epsilon_air = 1.0
        epsilon_dielectric = 4.0
        return epsilon_air + (epsilon_dielectric - epsilon_air) * (1.0 - sigma)

    @staticmethod
    def mu_r(x):
        return torch.ones_like(x[:, 0:1])

    @staticmethod
    def sigma(x):
        return torch.zeros_like(x[:, 0:1])


# =========================================================
# 源项
# =========================================================
class SourceTerms:
    @staticmethod
    def plane_wave_source(x, omega, device):
        direction = torch.tensor([1.0, 0.0, 0.0], device=device)
        k = omega * torch.sqrt(
            MaterialProperties.epsilon_r(x) *
            MaterialProperties.mu_r(x)
        )
        phase = torch.sum(
            k * x[:, :3] * direction.unsqueeze(0),
            dim=1,
            keepdim=True
        )

        J = torch.zeros(
            x.shape[0], 3,
            device=device,
            dtype=torch.complex64
        )
        J[:, 0] = torch.exp(-1j * phase).squeeze()
        return J


# =========================================================
# PDE 残差
# =========================================================
class PDEResidual:
    @staticmethod
    def compute_full_wave_residual(net, x, device):
        """
        ∇×(μ⁻¹ ∇×E) − ε ω² E = i ω μ J
        """

        omega = x[:, 3:4]

        out = net(x)
        E_real = out[:, 0:3]
        E_imag = out[:, 3:6]

        epsilon = MaterialProperties.epsilon_r(x)
        mu = MaterialProperties.mu_r(x)

        # curl–curl（只对 xyz 求导，physics 里已处理）
        helmholtz_real = Curl.helmholtz_operator(
            E_real, x, epsilon, mu, omega
        )
        helmholtz_imag = Curl.helmholtz_operator(
            E_imag, x, epsilon, mu, omega
        )

        J = SourceTerms.plane_wave_source(x, omega, device)
        J_real = J.real
        J_imag = J.imag

        residual_real = helmholtz_real + omega * mu * J_imag
        residual_imag = helmholtz_imag - omega * mu * J_real

        return residual_real, residual_imag


# =========================================================
# Loss
# =========================================================
class Loss:

    # -----------------------------
    # PDE Loss
    # -----------------------------
    @staticmethod
    def pde(net, x, device):
        res_r, res_i = PDEResidual.compute_full_wave_residual(
            net, x, device
        )
        return res_r.square().mean() + res_i.square().mean()

    # -----------------------------
    # Boundary Loss
    # -----------------------------
    @staticmethod
    def bc(net, x):
        out = net(x)
        E_real = out[:, 0:3]
        E_imag = out[:, 3:6]
        E = torch.sqrt(E_real**2 + E_imag**2)

        loss = torch.tensor(0.0, device=x.device)

        # =====================
        # PEC: x = ±1
        # =====================
        mask_pec = (x[:, 0].abs() > 0.99)
        if mask_pec.any():
            E_tan = torch.cat(
                [E[mask_pec, 1:2], E[mask_pec, 2:3]],
                dim=1
            )
            loss = loss + E_tan.square().mean()

        # =====================
        # Sommerfeld 辐射边界: z = 1
        # =====================
        mask_rad = (x[:, 2] > 0.99)
        if mask_rad.any():
            x_bc = x[mask_rad].clone().detach().requires_grad_(True)
            omega = x_bc[:, 3:4]

            Ez = net(x_bc)[:, 2:3]

            # ⚠️ 必须 allow_unused
            grad = torch.autograd.grad(
                Ez.sum(),
                x_bc,
                create_graph=True,
                allow_unused=True
            )[0]

            if grad is None:
                grad_z = torch.zeros_like(Ez)
            else:
                grad_z = grad[:, 2:3]

            epsilon = MaterialProperties.epsilon_r(x_bc)
            mu = MaterialProperties.mu_r(x_bc)
            beta = omega * torch.sqrt(epsilon * mu)

            radiation_residual = grad_z + beta * Ez
            loss = loss + radiation_residual.square().mean()

        return loss

    # -----------------------------
    # Total Loss
    # -----------------------------
    @staticmethod
    def total(net, x, device, lambda_pde=1.0, lambda_bc=10.0):
        lpde = Loss.pde(net, x, device)
        lbc = Loss.bc(net, x)
        return (
            lambda_pde * lpde + lambda_bc * lbc,
            lpde.item(),
            lbc.item()
        )
