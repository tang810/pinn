import torch

class WaveEquation:
    def __init__(self, dimension=1, c=1.0):
        assert dimension in [1, 2, 3]
        self.dimension = dimension
        self.c = c

    def grad(self, y, x):
        return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                                   create_graph=True, retain_graph=True)[0]

    def pde(self, coords, u):
        grads = {}
        for i, d in enumerate(['x', 'y', 'z'][:self.dimension]):
            grads[d] = coords[:, i:i+1]
        t = coords[:, self.dimension:self.dimension+1]

        # 时间二阶导
        u_t = self.grad(u, t)
        u_tt = self.grad(u_t, t)

        # Laplace项
        laplace = 0.0
        for d in ['x', 'y', 'z'][:self.dimension]:
            u_d = self.grad(u, grads[d])
            u_dd = self.grad(u_d, grads[d])
            laplace += u_dd

        residual = u_tt - self.c**2 * laplace
        return residual
