import torch

class NavierCauchy:
    def __init__(self, dimension=2, lam=1.0, mu=0.5):
        assert dimension in [2, 3], "Navier-Cauchy equation only supports 2D or 3D"
        self.dimension = dimension
        self.lam = lam
        self.mu = mu

    def grad(self, y, x):
        return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True)[0]

    def pde(self, coords, displacement, force=None):
        """
        Args:
            coords: [N, D] tensor, D = 2 or 3
            displacement: list of tensors, [u, v] or [u, v, w]
            force: tensor [N, D], default zero
        Returns:
            residuals: list of tensors, one per dimension
        """
        grads = {}
        for i, d in enumerate(['x', 'y', 'z'][:self.dimension]):
            grads[d] = coords[:, i:i+1]

        # Compute divergence: ∇·u
        div_u = 0.0
        for i, u_i in enumerate(displacement):
            du_dx = self.grad(u_i, grads[['x', 'y', 'z'][i]])
            div_u += du_dx

        residuals = []

        for i, u_i in enumerate(displacement):
            laplace = 0.0
            for d in ['x', 'y', 'z'][:self.dimension]:
                du_d = self.grad(u_i, grads[d])
                ddu_dd = self.grad(du_d, grads[d])
                laplace += ddu_dd

            ddiv_dx = self.grad(div_u, grads[['x', 'y', 'z'][i]])
            rhs = torch.zeros_like(u_i) if force is None else force[:, i:i+1]

            # Residual of Navier-Cauchy equation
            res_i = self.mu * laplace + (self.lam + self.mu) * ddiv_dx - rhs
            residuals.append(res_i)

        return residuals
