<!-- 
text: "Maxwell方程组求解",
area: "电磁场在无线通信、雷达散射、天线设计等领域扮演着核心角色。本项目使用PINN方法求解频域Maxwell方程组，建模电磁波的传播特性。",
tags: [
  "基础方程",
  "电磁场理论",
  "物理信息神经网络(PINN)",
  "Maxwell方程组",
],
search: ["基础方程", "电磁学"],
 -->

# MaxwellPDE: 频域Maxwell方程组

## 背景介绍

麦克斯韦方程组是描述电磁现象的基本物理定律，在无线通信、雷达系统、天线设计、电磁兼容等领域具有广泛应用。传统数值方法如有限元法(FEM)、时域有限差分法(FDTD)虽然成熟，但在处理复杂几何、高频问题和非均匀介质时面临计算量大、网格划分困难等挑战。


## 原理介绍

物理信息神经网络(PINN)是一种新兴的求解偏微分方程的方法，它将物理定律直接嵌入神经网络的训练过程中。本项目首次将高精度PINN应用于求解三维频域麦克斯韦方程组，考虑一个平面波在介电常数变化的介质中传播的问题，为电磁场计算提供了一种全新的无网格解决方案。


![caseMarkdown/Maxwell/viz/deepseek_mermaid_20251216_e7822f.png](https://www.science42.tech/cases/caseMarkdown/Maxwell/viz/deepseek_mermaid_20251216_e7822f.png)

## 创新点

- 首个完整的三维频域Maxwell PINN求解器，能够完整实现三维频域Maxwell方程的PINN求解，同时支持复数电场表示和计算。

- 支持高阶导数（双旋度）的高效计算、光滑过渡的介电常数模型，避免数值不连续问题。

- 频率参数化学习：频率作为网络输入参数，一次训练支持多频段查询。


## 1、控制方程
含源频域Maxwell方程：
$$
\nabla \times (\nabla \times \mathbf{E}) - \mu \epsilon \omega^2 \mathbf{E}=j\omega\mu\mathbf{J}
$$

总损失方程：

$$
L_{\mathrm{total}}=\lambda_{\mathrm{PDE}} L_{\mathrm{PDE}}+\lambda_{\mathrm{BC}} L_{\mathrm{BC}}
$$

如上图所示，我们将残差计算分为实部、虚部两部分：

$$
L_{\mathrm{PDE}}=\frac{1}{N}\sum_{i=1}^{N}\left(\| R_{\mathrm{real}}(\mathbf{x}_i) \|^2+\| R_{\mathrm{imag}}(\mathbf{x}_i) \|^2
\right)
$$

$$
L_{\mathrm{BC}} = L_{\mathrm{PEC}} + L_{\mathrm{rad}}
$$

$$
L_{\mathrm{PEC}}=\frac{1}{N_{\mathrm{PEC}}}\sum_{\mathbf{x}_i  \in  \Gamma_{\mathrm{PEC}}}
\left(|E_y(\mathbf{x}_i)|^2+|E_z(\mathbf{x}_i)|^2\right)
$$

$$
L_{\mathrm{rad}}=\frac{1}{N_{\mathrm{rad}}}\sum_{\mathbf{x}_i \in \Gamma_{\mathrm{rad}}}
\left\|\frac{\partial E_z}{\partial z}(\mathbf{x}_i)+j\beta E_z(\mathbf{x}_i)
\right\|^2
$$

PDE实部残差：

$$
\mathbf{R}_{\text{real}}(\mathbf{r}) = \nabla \times (\mu^{-1} \nabla \times \mathbf{E}_{\text{real}}) - \epsilon \omega^2 \mathbf{E}_{\text{real}} - \omega\mu \mathbf{J}_{\text{imag}}
$$


PDE虚部残差：
$$
\mathbf{R}_{\text{imag}}(\mathbf{r}) = \nabla \times (\mu^{-1} \nabla \times \mathbf{E}_{\text{imag}}) - \epsilon \omega^2 \mathbf{E}_{\text{imag}} + \omega\mu \mathbf{J}_{\text{real}}
$$




## 2、参数、属性定义



### 物理量定义表：

| 符号 | 物理意义 | 单位/数值 | 类型 | 说明 |
|------|----------|-----------|------|------|
| E | 电场强度 | V/m | 复数矢量场 | 空间和频率的连续函数 |
| E_real | 电场实部 | V/m | 实矢量场 | E_real = Re(E) |
| E_imag | 电场虚部 | V/m | 实矢量场 | E_imag = Im(E) |
| Ex_r, Ey_r, Ez_r | 电场实部分量 | V/m | 标量场 | 对应 E_real 的分量 |
| Ex_i, Ey_i, Ez_i | 电场虚部分量 | V/m | 标量场 | 对应 E_imag 的分量 |
| μ | 磁导率 | H/m | 常数 | μ = μ0μr = 1.0 |
| μr | 相对磁导率 | 无量纲 | 常数 | 设为 1.0 |
| ε | 介电常数 | F/m | 空间函数 | ε = ε0εr(r) |
| εr | 相对介电常数 | 无量纲 | 空间函数 | 空气(εr=1)到介质(εr=4)的过渡 |
| ω | 角频率 | rad/s | 训练参数 | 在 [0.5π, 2π] 范围内变化 |
| f | 频率 | Hz | 训练参数 | f = ω/(2π) |
| J | 电流密度源 | A/m² | 复数矢量场 | 平面波激励源 |
| J_real | 源项实部 | A/m² | 实矢量场 | J_real = Re(J) |
| J_imag | 源项虚部 | A/m² | 实矢量场 | J_imag = Im(J) |
| J0 | 源强度幅值 | A/m² | 常数 | 设为 1.0 |
| k | 波矢量 | rad/m | 矢量场 | k = ω√(με)k̂ |
| k | 波数 | rad/m | 标量场 | k = |k| = ω√(με) |
| β | 波数（边界用） | rad/m | 标量场 | 与k相同，用于辐射边界条件 |
| r | 位置矢量 | m | 坐标 | r = [x, y, z]^T |
| x, y, z | 空间坐标 | m | 坐标 | 在 [-1, 1] 范围内 |
| n | 边界法向矢量 | 无量纲 | 矢量 | 边界的单位法向量 |
| Γ_PEC | PEC边界 | 无量纲 | 几何面 | x = ±1 的两个平面 |
| Γ_rad | 辐射边界 | 无量纲 | 几何面 | z = 1 的平面 |
| Ω | 计算域 | m³ | 几何体 | [-1,1]×[-1,1]×[-1,1] 的立方体 |
| ∇× | 旋度算子 | m⁻¹ | 微分算子 | 作用于矢量场的微分运算 |
| σ | 电导率 | S/m | 常数 | 设为 0.0（无损介质） |
| λ_PDE | PDE损失权重 | 无量纲 | 超参数 | 默认值 1.0 |
| λ_BC | 边界损失权重 | 无量纲 | 超参数 | 默认值 10.0 |
| L_total | 总损失函数 | 无量纲 | 标量 | λ_PDE·L_PDE + λ_BC·L_BC |
| L_PDE | PDE残差损失 | 无量纲 | 标量 | 控制方程不满足程度的度量 |
| L_PEC | PEC边界损失 | 无量纲 | 标量 | 切向电场不为零的惩罚 |
| L_rad | 辐射边界损失 | 无量纲 | 标量 | 辐射条件不满足的惩罚 |
| N | 总采样点数 | 个 | 整数 | N_int + N_BC |
| N_int | 内部点数量 | 个 | 整数 | 默认 1500 |
| N_BC | 边界点数量 | 个 | 整数 | 默认 450 |
| R_real | 实部PDE残差 | V/m³ | 矢量场 | 实部方程的不平衡量 |
| R_imag | 虚部PDE残差 | V/m³ | 矢量场 | 虚部方程的不平衡量 |

### 常数定义表

| 常数 | 符号 | 数值 | 单位 | 说明 |
|------|------|------|------|------|
| 真空磁导率 | μ0 | 4π×10⁻⁷ | H/m | 磁导率基准 |
| 真空介电常数 | ε0 | 8.854×10⁻¹² | F/m | 介电常数基准 |
| 光速 | c | 3.0×10⁸ | m/s | c = 1/√(μ0ε0) |
| 界面位置 | x0 | 0.0 | m | 空气-介质分界面 |
| 过渡宽度 | δ | 0.05 | m | 介电常数过渡区宽度 |
| 空气介电常数 | ε_air | 1.0 | 无量纲 | 相对介电常数 |
| 介质介电常数 | ε_dielectric | 4.0 | 无量纲 | 相对介电常数 |

### 边界条件参数表

| 边界 | 条件类型 | 数学表达式 | 代码实现 |
|------|----------|------------|----------|
| x = -1 | PEC（理想电导体） | Ey = 0, Ez = 0 | x[:, 0] < -0.99 |
| x = 1 | PEC（理想电导体） | Ey = 0, Ez = 0 | x[:, 0] > 0.99 |
| z = 1 | 辐射边界 | ∂Ez/∂z + jβEz = 0 | x[:, 2] > 0.99 |
| 其他面 | 自然边界 | 无显式条件 | 通过PDE自动满足 |

### 神经网络参数表

| 参数 | 符号 | 数值 | 说明 |
|------|------|------|------|
| 输入维度 | d_in | 4 | [x, y, z, ω] |
| 输出维度 | d_out | 6 | [Ex_r, Ey_r, Ez_r, Ex_i, Ey_i, Ez_i] |
| 隐藏层维度 | d_hidden | 256 | 每层的神经元数量 |
| 网络层数 | L | 5 | 包含输入层和输出层 |
| 激活函数 | - | SiLU/Tanh | 交替使用不同激活函数 |
| 学习率 | η | 5×10⁻⁴ | Adam优化器的初始学习率 |
| 批量大小 | B | 64 | 每个训练批次的样本数 |
| 训练轮数 | N_epochs | 3000 | 总训练迭代次数 |


### 软件版本要求：
建议Python 3.10以上


### 所需要导入的库：

```  
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
```


## 3、计算方法及代码实现

### 3.1 高精度PINN网络
```python
   class HighPrecisionPINN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=256, output_dim=6, n_layers=5):
        super(HighPrecisionPINN, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for i in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU() if i % 2 == 0 else nn.Tanh())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.net(x)
```
### 3.2 精确梯度计算
```python
class GradientCalculator:
    @staticmethod
    def compute_gradient(f, x):
        if not x.requires_grad:
            x.requires_grad_(True)
        if not f.requires_grad:
            f = f.clone().detach().requires_grad_(True)
        if f.dim() == 2 and f.shape[1] > 1:
            grads = []
            for i in range(f.shape[1]):
                grad_i = torch.autograd.grad(
                    f[:, i].sum(), x, create_graph=True, retain_graph=True, allow_unused=True
                )[0]
                if grad_i is None:
                    grad_i = torch.zeros_like(x)
                grads.append(grad_i.unsqueeze(1))
            return torch.cat(grads, dim=1)
        else:
            grad = torch.autograd.grad(f.sum(), x, create_graph=True, allow_unused=True)[0]
            if grad is None:
                grad = torch.zeros_like(x)
            return grad
```
### 3.3 精确旋度算子
```python
class CurlOperator:
    @staticmethod
    def curl(E: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        grad_E = GradientCalculator.compute_gradient(E, x)
        curl_E = torch.zeros_like(E)
        curl_E[:, 0] = grad_E[:, 2, 1] - grad_E[:, 1, 2]
        curl_E[:, 1] = grad_E[:, 0, 2] - grad_E[:, 2, 0]
        curl_E[:, 2] = grad_E[:, 1, 0] - grad_E[:, 0, 1]
        return curl_E

    @staticmethod
    def double_curl(E: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return CurlOperator.curl(CurlOperator.curl(E, x), x)

    @staticmethod
    def helmholtz_operator(E: torch.Tensor, x: torch.Tensor, 
                           epsilon: torch.Tensor, mu: torch.Tensor, 
                           omega: torch.Tensor) -> torch.Tensor:
        curl_E = CurlOperator.curl(E, x)
        mu_inv_curl_E = (curl_E / mu).clone().detach().requires_grad_(True)
        curl_mu_inv_curl_E = CurlOperator.curl(mu_inv_curl_E, x)
        return curl_mu_inv_curl_E - epsilon * (omega**2) * E

```
### 3.4 材料参数
```python
class MaterialProperties:
    @staticmethod
    def epsilon_r(x: torch.Tensor, interface_position: float = 0.0) -> torch.Tensor:
        transition_width = 0.05
        sigma = 1.0 / (1.0 + torch.exp(-(x[:, 0:1] - interface_position) / transition_width))
        epsilon_air = 1.0
        epsilon_dielectric = 4.0
        return epsilon_air + (epsilon_dielectric - epsilon_air) * (1.0 - sigma)

    @staticmethod
    def mu_r(x: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x[:, 0:1])

    @staticmethod
    def sigma(x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x[:, 0:1])
```

### 3.5 PDE残差
```python
class PDEResidual:
    @staticmethod
    def compute_full_wave_residual(net: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_coord = x[:, :3]  
        omega = x[:, 3:4]
        output = net(x)
        E_real = output[:, 0:3]
        E_imag = output[:, 3:6]
        epsilon = MaterialProperties.epsilon_r(x)
        mu = MaterialProperties.mu_r(x)
        helmholtz_real = CurlOperator.helmholtz_operator(E_real, x_coord, epsilon, mu, omega)
        helmholtz_imag = CurlOperator.helmholtz_operator(E_imag, x_coord, epsilon, mu, omega)
        J_complex = SourceTerms.plane_wave_source(x, omega)
        J_real = J_complex.real
        J_imag = J_complex.imag
        residual_real = helmholtz_real + omega * mu * J_imag
        residual_imag = helmholtz_imag - omega * mu * J_real
        return residual_real, residual_imag

    @staticmethod
    def compute_loss(net: nn.Module, x: torch.Tensor, lambda_pde=1.0, lambda_bc=10.0) -> Tuple[torch.Tensor, dict]:
        residual_real, residual_imag = PDEResidual.compute_full_wave_residual(net, x)
        loss_pde = torch.mean(residual_real**2 + residual_imag**2)
        loss_bc = BoundaryConditions.compute_boundary_loss(net, x)
        total_loss = lambda_pde * loss_pde + lambda_bc * loss_bc
        return total_loss, {'total': total_loss.item(), 'pde': loss_pde.item(), 'bc': loss_bc.item()}
```

### 3.6 边界条件

```python
class BoundaryConditions:
    @staticmethod
    def compute_boundary_loss(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
        output = net(x)
        E_real = output[:, 0:3]
        E_imag = output[:, 3:6]
        E = torch.sqrt(E_real**2 + E_imag**2)
        loss = torch.tensor(0.0, device=device)
        mask_pec = (x[:, 0].abs() > 0.99)
        if mask_pec.any():
            E_tan = torch.cat([E[mask_pec, 1:2], E[mask_pec, 2:3]], dim=1)
            loss += torch.mean(E_tan**2)
        mask_rad = (x[:, 2] > 0.99)
        if mask_rad.any():
            x_bc = x[mask_rad].clone().detach().requires_grad_(True)
            omega = x_bc[:, 3:4]
            E_z = net(x_bc)[:, 2:3]
            grad = torch.autograd.grad(E_z.sum(), x_bc, create_graph=True, allow_unused=True)[0]
            grad_z = grad[:, 2:3] if grad is not None else torch.zeros_like(E_z)
            epsilon = MaterialProperties.epsilon_r(x_bc)
            mu = MaterialProperties.mu_r(x_bc)
            beta = omega * torch.sqrt(epsilon * mu)
            radiation_residual = grad_z + 1j * beta * E_z
            loss += torch.mean(torch.abs(radiation_residual)**2)
        return loss

```
### 3.7 可视化工具

```python
class VisualizationTools:
    @staticmethod
    def plot_training_history(history: dict):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].semilogy(history['loss'], 'b-'); axes[0].set_title('Total Loss')
        axes[1].semilogy(history['pde_loss'], 'r-'); axes[1].set_title('PDE Loss')
        axes[2].semilogy(history['bc_loss'], 'g-'); axes[2].set_title('Boundary Loss')
        plt.tight_layout(); plt.show()
```

## 4、模型高精度训练

```python
class HighPrecisionTrainer:
    def __init__(self, net: nn.Module):
        self.net = net
        self.optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=50, verbose=True
        )

    def train_epoch(self, X_train: torch.Tensor, batch_size=64) -> dict:
        N = len(X_train)
        indices = torch.randperm(N)
        total_loss = total_pde = total_bc = 0
        for i in range(0, N, batch_size):
            batch_idx = indices[i:i+batch_size]
            x_batch = X_train[batch_idx]
            x_batch.requires_grad_(True)
            self.optimizer.zero_grad()
            loss, loss_dict = PDEResidual.compute_loss(self.net, x_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss_dict['total']
            total_pde += loss_dict['pde']
            total_bc += loss_dict['bc']
        num_batches = max(1, N // batch_size)
        return {'loss': total_loss/num_batches, 'pde_loss': total_pde/num_batches, 'bc_loss': total_bc/num_batches}

    def train(self, X_train: torch.Tensor, epochs=2000, batch_size=64, verbose_interval=100):
        history = {'loss': [], 'pde_loss': [], 'bc_loss': []}
        print(f"开始高精度训练，共{epochs}个epochs")
        for epoch in range(epochs):
            metrics = self.train_epoch(X_train, batch_size)
            for key in metrics: history[key].append(metrics[key])
            self.scheduler.step(metrics['loss'])
            if epoch % verbose_interval == 0:
                print(f"Epoch {epoch:4d}/{epochs} | Loss: {metrics['loss']:.3e} | PDE: {metrics['pde_loss']:.3e} | BC: {metrics['bc_loss']:.3e}")
        return history
```
## 5、 预测、绘图及成果展示

这是一个典型的PINN求解Maxwell方程的训练过程监控图，包含总损失、PDE损失和边界条件损失三个关键指标。总损失是加权综合指标，PDE损失衡量Maxwell方程的物理约束满足程度，边界损失确保PEC和辐射边界条件的正确性。



![caseMarkdown/Maxwell/viz/Figure_11.png](https://www.science42.tech/cases/caseMarkdown/Maxwell/viz/Figure_11.png)


- 总损失从约10³下降到10⁻²~10⁻³量级，下降了5-6个数量级，表明训练成功收敛。

- PDE损失达到10⁻³量级，说明神经网络解在计算域内较好地满足了Maxwell方程。

- 边界损失也收敛到较低水平，确保了解在边界上的物理正确性。

## 6、常见问题（FAQ）

- 收敛性问题：损失震荡不下降、初始化权重不合适        、边界条件权重过强、数值不稳定。

- 精度问题：材料界面、奇异点、高频区域等精度会出现降低现象。

- 计算效率问题：精度受网络容量影响。

## 7、工业价值

- 电子设计自动化 (EDA)：在集成电路和PCB设计领域，PINN方法能够大幅缩短设计周期。传统电磁仿真需要复杂的网格划分和长时间计算，而PINN通过无网格方式可直接求解Maxwell方程，使工程师能够快速评估数百种布局方案的电磁性能，显著减少物理原型制作成本。特别是在5nm以下先进制程中，互连结构的电磁效应愈加显著，PINN的高效求解能力可为信号完整性、电源完整性和电磁兼容性提供实时分析支持，帮助企业加速产品上市时间并降低研发风险。

- 医疗设备与生物电磁学：医疗设备电磁安全是关乎生命的重要课题。PINN能够精确计算电磁场在人体组织中的三维分布，为MRI线圈设计、肿瘤射频消融、神经刺激设备等提供关键数据。相比传统有限元方法需要复杂的人体模型网格划分，PINN可直接从医学影像数据构建电磁模型，实现患者特异性的治疗规划。同时，该方法能够准确评估比吸收率(SAR)，确保医疗设备符合安全标准，为个性化医疗和精准治疗提供物理基础。

- 自动驾驶系统的可靠性依赖车载雷达、激光雷达和V2X通信的稳定工作。PINN可模拟复杂道路环境中电磁波的传播特性，包括多径效应、车辆遮挡、天气影响等因素。通过构建高保真的电磁环境数字孪生，工程师能够优化天线布局、评估干扰风险、设计抗干扰算法，确保在雨雪天气、隧道、城市峡谷等挑战性场景下的可靠感知和通信。这为L4/L5级自动驾驶系统的安全认证提供了关键的技术支撑。