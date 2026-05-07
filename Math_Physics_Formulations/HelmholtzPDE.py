import torch

class HelmholtzPDE:
    """
    Helmholtz 方程类
    - 支持基于场变量的PDE残差计算
    - 直接可与PINN框架集成
    """

    def __init__(self, k):
        """
        参数:
            k: 波数 (float or torch.tensor)
        """
        self.k = k

    def compute_pde_residuals(self, u, coords):
        """
        计算Helmholtz方程残差
        方程形式:
            ∇²u + k²u = 0
        参数:
            u: 场变量 (torch.tensor)，shape: [N, 1]
            coords: 坐标 (torch.tensor)，shape: [N, D]，D=1,2,3
        返回:
            torch.tensor: 残差，shape: [N, 1]
        """
        laplacian = 0.0
        for d in range(coords.shape[1]):
            x_d = coords[:, d:d+1]
            u_x = torch.autograd.grad(u, x_d, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x, x_d, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
            laplacian += u_xx

        residual = laplacian + (self.k ** 2) * u
        return residual

    def summary(self):
        """
        输出波数信息
        """
        print(f"Wave Number: k = {self.k}")
