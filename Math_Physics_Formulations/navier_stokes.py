import torch

class NavierStokes:
    def __init__(self, dimension=2, nu=0.01, rho=1.0):
        assert dimension in [1, 2, 3], "Only support 1D, 2D or 3D"
        self.dimension = dimension
        self.nu = nu
        self.rho = rho

    def compute_grad(self, y, x):
        return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True)[0]

    def pde(self, coords, velocity, pressure):
        """
        Args:
            coords: (x, y, z, t) -> tensor with shape [N, D] (D = dimension + 1)
            velocity: list of velocity components -> [u], [u,v], [u,v,w]
            pressure: tensor -> [N, 1]
        Returns:
            residuals: list -> [momentum_eqs..., continuity_eq]
        """
        grads = {}
        for i, name in enumerate(['x', 'y', 'z'][:self.dimension] + ['t']):
            grads[name] = coords[:, i:i+1]

        # Compute derivatives for each velocity component
        velocity_grads = {}
        velocity_laplacian = {}

        for idx, vel in enumerate(velocity):
            name = ['u', 'v', 'w'][idx]
            vel_t = self.compute_grad(vel, grads['t'])

            spatial_derivs = []
            spatial_derivs2 = []
            for d in ['x', 'y', 'z'][:self.dimension]:
                vel_d = self.compute_grad(vel, grads[d])
                vel_dd = self.compute_grad(vel_d, grads[d])
                spatial_derivs.append(vel_d)
                spatial_derivs2.append(vel_dd)

            velocity_grads[name] = (vel_t, spatial_derivs)
            velocity_laplacian[name] = sum(spatial_derivs2)

        # Compute pressure gradients
        pressure_grads = []
        for d in ['x', 'y', 'z'][:self.dimension]:
            p_d = self.compute_grad(pressure, grads[d])
            pressure_grads.append(p_d)

        # Build Momentum residuals
        momentum_residuals = []
        for idx, vel_name in enumerate(['u', 'v', 'w'][:self.dimension]):
            vel_t, vel_space_grads = velocity_grads[vel_name]
            convection = sum([velocity[i] * vel_space_grads[i] for i in range(self.dimension)])
            pressure_term = pressure_grads[idx] / self.rho
            diffusion = self.nu * velocity_laplacian[vel_name]

            f_momentum = vel_t + convection + pressure_term - diffusion
            momentum_residuals.append(f_momentum)

        # Continuity residual
        div = 0
        for idx, vel in enumerate(velocity):
            vel_d = self.compute_grad(vel, grads[['x', 'y', 'z'][idx]])
            div += vel_d

        return momentum_residuals + [div]
