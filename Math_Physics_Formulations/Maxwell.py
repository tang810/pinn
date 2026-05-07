import torch

class MaxwellPDE:
    """
    频域Maxwell方程组类（适用于电磁波传播模拟）
    - 支持基于电场的PDE残差计算
    - 直接可与PINN框架集成
    """

    def __init__(self, mu, epsilon, omega):
        """
        参数:
            mu: 磁导率 (float or torch.tensor)
            epsilon: 介电常数 (float or torch.tensor)
            omega: 角频率 (float or torch.tensor)
        """
        self.mu = mu
        self.epsilon = epsilon
        self.omega = omega

    def compute_curl(self, E, coords):
        """
        计算电场 E 的旋度
        参数:
            E: 电场向量 (torch.tensor)，shape: [N, 3]
            coords: 坐标 (torch.tensor)，shape: [N, 3]
        返回:
            torch.tensor: shape [N, 3]，分别对应 curl_x, curl_y, curl_z
        """
        x = coords[:, 0:1]
        y = coords[:, 1:2]
        z = coords[:, 2:3]

        E_x = E[:, 0:1]
        E_y = E[:, 1:2]
        E_z = E[:, 2:3]

        E_z_y = torch.autograd.grad(E_z, y, grad_outputs=torch.ones_like(E_z), create_graph=True)[0]
        E_y_z = torch.autograd.grad(E_y, z, grad_outputs=torch.ones_like(E_y), create_graph=True)[0]
        E_x_z = torch.autograd.grad(E_x, z, grad_outputs=torch.ones_like(E_x), create_graph=True)[0]
        E_z_x = torch.autograd.grad(E_z, x, grad_outputs=torch.ones_like(E_z), create_graph=True)[0]
        E_y_x = torch.autograd.grad(E_y, x, grad_outputs=torch.ones_like(E_y), create_graph=True)[0]
        E_x_y = torch.autograd.grad(E_x, y, grad_outputs=torch.ones_like(E_x), create_graph=True)[0]

        curl_x = E_z_y - E_y_z
        curl_y = E_x_z - E_z_x
        curl_z = E_y_x - E_x_y

        return torch.cat([curl_x, curl_y, curl_z], dim=1)

    def compute_pde_residuals(self, E, coords):
        """
        计算频域Maxwell方程残差
        参数:
            E: 电场向量 (torch.tensor)，shape: [N, 3]
            coords: 坐标 (torch.tensor)，shape: [N, 3]
        返回:
            torch.tensor: shape [N, 3]，对应 x,y,z 方向的残差
        """
        curl_E = self.compute_curl(E, coords)
        curl_curl_E = self.compute_curl(curl_E, coords)

        residual = curl_curl_E - (self.mu * self.epsilon * self.omega ** 2) * E
        return residual

    def summary(self):
        """
        输出介质参数
        """
        print(f"Material Properties: μ = {self.mu}, ε = {self.epsilon}, ω = {self.omega}")
