import torch

class KdV:
    def compute_grad(self, y, x):
        return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                                    create_graph=True, retain_graph=True)[0]

    def pde(self, coords, u):
        x = coords[:, 0:1]
        t = coords[:, 1:2]

        u_t = self.compute_grad(u, t)
        u_x = self.compute_grad(u, x)
        u_xx = self.compute_grad(u_x, x)
        u_xxx = self.compute_grad(u_xx, x)

        residual = u_t + 6 * u * u_x + u_xxx
        return residual
