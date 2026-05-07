import torch

class Poisson:
    def __init__(self, dimension=2):
        assert dimension in [1, 2, 3], "Only support 1D, 2D or 3D"
        self.dimension = dimension

    def compute_grad(self, y, x):
        return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True)[0]

    def pde(self, coords, u, f):
        """
        泊松方程残差计算

        Args:
            coords: 坐标 (x, y, z), tensor [N, D]
            u: 预测值 (标量), tensor [N, 1]
            f: 右侧项 (已知函数), tensor [N, 1]

        Returns:
            残差: tensor [N, 1]
        """
        grads = {}
        for i, name in enumerate(['x', 'y', 'z'][:self.dimension]):
            grads[name] = coords[:, i:i+1]

        # 计算拉普拉斯项
        laplacian = 0.0
        for d in ['x', 'y', 'z'][:self.dimension]:
            u_d = self.compute_grad(u, grads[d])
            u_dd = self.compute_grad(u_d, grads[d])
            laplacian += u_dd

        # 残差
        residual = laplacian - f
        return residual
