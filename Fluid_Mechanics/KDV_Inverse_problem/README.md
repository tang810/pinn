<!-- 
README.md文件名统一大写
 -->


<!-- 
text: "内孤立波KdV参数反演与分层识别",

area: "本项目面向内孤立波观测反演场景，利用物理信息神经网络（PINNs）将KdV方程残差、初始边界条件与稀疏波高观测共同嵌入优化过程。案例重点展示在观测数据有限或带噪时，如何从波高曲线反推出非线性系数、色散系数以及密度、层深等分层参数，适用于物理参数识别、传感器数据解释和反问题建模。",

tags: [ "内孤立波", "稀疏观测", "参数反演", "反问题", "PINN", "分层流体" ]
-->

# 内孤立波KdV参数反演与分层识别

## 快速使用

您可跳转[使用示例](#使用示例)查看如何使用秋月白平台快速执行该类问题，以下是实现逻辑详解：


准确模拟与识别内孤立波（ISWs），对工程结构安全、水下声学和物质输运评估至关重要。**科特韦赫–德弗里斯（KdV）方程**是双层流体系统中描述弱非线性、弱色散内孤立波的重要模型。实际观测中，波高数据往往稀疏、带噪，底层流体参数也不易直接测量，因此需要能够同时利用数据和方程约束的反问题方法。

本项目采用**物理信息神经网络（PINNs）**，将 KdV 方程作为物理约束直接嵌入损失函数，同时利用稀疏/带噪观测进行监督或正则化，聚焦一类核心任务：

- **反问题求解（Inverse Problem）**：在仅有稀疏或带噪的波高观测下，**反演 KdV 关键参数**（如非线性系数 $c_1$、色散系数 $c_2$），并进一步推断密度 $\rho$ 与层深 $h$ 等流体特性。

本文档介绍该参数反演效果的实现逻辑以及使用秋月白平台辅助解决该类问题的操作指南。若需要“给定参数后快速生成波形”的代理预测，请参考 `kdv_operator_learning` 案例。

---
## 数学背景
本项目主要基于流体力学中的非线性波动理论，特别是针对内孤立波建模的科特韦赫-德弗里斯（KdV）方程。同时，应用了深度学习中的物理信息神经网络（PINNs）方法。

### 物理约束

1.  **科特韦赫–德弗里斯 (Korteweg-De Vries, KdV) 方程**  
    描述双层流体系统中波高 $\eta(x,t)$ 演化的KdV方程为：

    $$
    \frac{\partial\eta}{\partial t} + c_0 \frac{\partial\eta}{\partial x} + c_1\eta \frac{\partial\eta}{\partial x} + c_2 \frac{\partial^3\eta}{\partial x^3} = 0
    $$

    其中 $c_0$ 是线性相速度， $c_1$ 是非线性系数， $c_2$ 是色散系数。这些系数依赖于流体特性（上层密度 $\rho_1$、下层密度 $\rho_2$、上层未扰动厚度 $h_1$、下层未扰动厚度 $h_2$）以及重力加速度 $g$。

    $$
    c_0=\sqrt{\frac{g'h_1h_2}{h_1+h_2}}
    $$

    $$
    c_1=\frac{3c_0}{2}\frac{h_1-h_2}{h_1h_2}
    $$

    $$
    c_2=\frac{c_0h_1h_2}{6}
    $$

    $$
    g'=g\frac{\rho_2-\rho_1}{\rho_2}
    $$

2.  **孤立波解 (Soliton Solution)**  
    对于KdV方程，一个著名的解析解是孤立波解，在随波坐标系 $X = x - c t$ （其中 $c$ 是波速）下，其形式为：

    $$
    \eta(X) = a \text{sech}^2\left(\frac{X}{\lambda}\right)
    $$

    其中 $a$ 是孤立波的振幅， $\lambda$ 是特征波长。 对于KdV方程的孤立波，波速 $c$ 和波长 $\lambda$ 与振幅 $a$ 及系数 $c_1, c_2$ 相关： $c = c_0 + \frac{1}{3} a c_1$ 和 $\lambda = \sqrt{\frac{12 c_2}{a c_1}}$。

3.  **边界条件 (Boundary Conditions)**  
    对于孤立波问题，通常假设在远离波中心的区域，波高及其导数趋于零：
    $$
    \eta(x,t) \to 0, \quad \frac{\partial\eta}{\partial x} \to 0, \quad \text{as} \quad |x| \to \infty
    $$
    在有限计算域中，这通常在计算域的边界上施加。

#### 物理方程参数说明：

| 参数             | 含义                                                             | 默认值（示例性，具体问题中可能为变量或待求） |
|:----------------:|:-----------------------------------------------------------------|:-------------------------------------------:|
| $\eta(x,t)$      | 时间 $t$ 和空间 $x$ 处的波高                                      | —                                           |
| $x$              | 空间坐标                                                         | —                                           |
| $t$              | 时间坐标                                                         | —                                           |
| $c_0$            | KdV方程的线性相速度系数                                             | 依赖于 $\rho_1, \rho_2, h_1, h_2, g$             |
| $c_1$            | KdV方程的非线性系数                                               | 依赖于 $\rho_1, \rho_2, h_1, h_2, g$             |
| $c_2$            | KdV方程的色散系数                                                 | 依赖于 $\rho_1, \rho_2, h_1, h_2, g$             |
| $\rho_1, \rho_2$ | 上、下层流体密度                                                   | 例如 $\rho_1=998$, $\rho_2=1025$ kg/m³       |
| $h_1, h_2$       | 上、下层流体未扰动厚度                                             | 例如 $h_1=50$, $h_2=200$ m                   |
| $g$              | 重力加速度                                                       | `9.81` m/s²                                 |
| $a$              | 孤立波振幅                                                       | — (算子学习输入或反问题输出)                |
| $\lambda$        | 孤立波特征波长                                                   | 依赖于 $a, c_1, c_2$                        |
| $p$              | 密度比 $\rho_2/\rho_1$                                          | — (算子学习输入)                            |

---

### PINN 损失函数
为了将KdV方程的物理约束和已知的观测数据/边界条件嵌入到神经网络的训练中，我们定义一个总损失函数 $L_{total}$，它由以下几部分组成：

1.  **PDE 残差损失 ($L_{PDE}$)**:  
    该损失项惩罚神经网络的输出 $\tilde{\eta}(x,t;\theta)$ (其中 $\theta$ 是网络参数) 在计算域内部选取的配置点 $(x_r, t_r)$ 上对KdV方程的偏离程度。KdV方程的残差 $R(x,t)$ 定义为：
    $$
    R(x,t) = \frac{\partial\tilde{\eta}}{\partial t} + c_0 \frac{\partial\tilde{\eta}}{\partial x} + c_1\tilde{\eta} \frac{\partial\tilde{\eta}}{\partial x} + c_2 \frac{\partial^3\tilde{\eta}}{\partial x^3}
    $$
    PDE损失计算为这些残差的均方误差：
    $$
    L_{PDE} = \frac{1}{N_r} \sum_{i=1}^{N_r} |R(x_r^{(i)}, t_r^{(i)})|^2
    $$
    其中 $N_r$ 是内部配置点的数量。导数通过自动微分（Automatic Differentiation, AD）精确计算。

2.  **边界条件损失 ($L_{BC}$)**:  
    此损失项确保网络预测满足指定的边界条件。例如，对于周期性边界条件或Dirichlet/Neumann边界条件。对于孤立波，常在计算域边界 $(x_b, t_b)$ 处施加 $\tilde{\eta} \to 0$：
    $$
    L_{BC} = \frac{1}{N_b} \sum_{i=1}^{N_b} |\tilde{\eta}(x_b^{(i)}, t_b^{(i)}) - \eta_{boundary}(x_b^{(i)}, t_b^{(i)})|^2
    $$
    其中 $N_b$ 是边界点的数量， $\eta_{boundary}$ 是边界上的目标值 (通常为0)。

3.  **数据损失 ($L_{data}$)**:  
    如果存在已知的观测数据点 $(x_d, t_d, \eta_{obs})$, 该损失项衡量网络预测与这些观测数据之间的差异：
    $$
    L_{data} = \frac{1}{N_d} \sum_{i=1}^{N_d} |\tilde{\eta}(x_d^{(i)}, t_d^{(i)}) - \eta_{obs}(x_d^{(i)}, t_d^{(i)})|^2
    $$
    其中 $N_d$ 是观测数据点的数量。在算子学习任务中，这可能是从精确解中采样得到的点；在反问题中，这可能是稀疏的实验测量数据。

4.  **初始条件损失 ($L_{IC}$)** (如果适用):  
    如果给定初始波形 $\eta(x, t=0) = \eta_0(x)$，则可以添加初始条件损失：
    $$
    L_{IC} = \frac{1}{N_{ic}} \sum_{i=1}^{N_{ic}} |\tilde{\eta}(x_{ic}^{(i)}, 0) - \eta_0(x_{ic}^{(i)})|^2
    $$
    其中 $N_{ic}$ 是初始时刻采样点的数量。


总损失函数是这些分量的加权和：
$$
L_{total} = w_{PDE} L_{PDE} + w_{BC} L_{BC} + w_{data} L_{data} (+ w_{IC} L_{IC})
$$
其中 $w_{PDE}, w_{BC}, w_{data}, w_{IC}$ 是相应的权重因子，用于平衡不同损失项的贡献。

#### 符号说明：

| 符号                   | 含义                                                                 |
|:----------------------:|:--------------------------------------------------------------------:|
| $\tilde{\eta}(x,t;\theta)$ | 神经网络对波高 $\eta$ 在 $(x,t)$ 处的预测，参数为 $\theta$               |
| $\theta$               | 神经网络的权重和偏置                                                   |
| $R(x,t)$               | KdV方程在 $(x,t)$ 处的残差                                               |
| $N_r, N_b, N_d, N_{ic}$  | 内部配置点、边界点、数据点、初始条件点的数量                             |
| $(x_r, t_r)$           | 内部配置点坐标                                                           |
| $(x_b, t_b)$           | 边界点坐标                                                               |
| $(x_d, t_d)$           | 观测数据点坐标                                                           |
| $\eta_{obs}$           | 在 $(x_d, t_d)$ 处的观测波高值                                           |
| $\eta_{boundary}$      | 在边界 $(x_b, t_b)$ 处的目标波高值                                        |
| $\eta_0(x)$            | 初始时刻 $t=0$ 的波形                                                    |
| $w_{PDE}, w_{BC}, \dots$ | 对应损失项的权重                                                         |

> 💡 **说明：**  
> - 在算子学习任务中， $L_{data}$ 通常使用从不同参数下的KdV精确解采样的数据。
> - 在反问题任务中， $L_{data}$ 使用稀疏的观测数据，KdV方程的系数 ($c_0, c_1, c_2$) 或其底层的物理参数 ($\rho_1, h_1$, 等) 与网络参数 $\theta$ 一同作为优化变量。
> - 权重 $w$ 的选择对训练效果有较大影响，可能需要根据具体问题进行调整。

---
## 快速开始
本单元中介绍如何以最快速度配置环境并运行一个完整的实例，预期输出结果包括：
- 对于算子学习：给定参数集 $(p, h_1, a)$，预测孤立波解 $\eta(X)$。
- 对于反问题：根据观测数据 $\eta_{obs}(x_d, t_d)$，估计KdV方程的系数或相关的物理参数。

您也可直接跳转[秋月白平台使用实例](#平台使用示例)查看如何直接调用该功能。

通过将物理约束与数据驱动的优化过程相结合，PINNs能够有效应对KdV方程求解中的挑战，例如参数不可观测性、数据稀疏性等，为内波研究提供一种高效、准确的数值工具。


<!-- 添加图片方式：纯Markdown 格式 -->
| ![Fig.9](https://www.science42.tech/cases/caseMarkdown/KdV_Korteweg-de-Vries-km/viz/图三.png) | ![](https://www.science42.tech/cases/caseMarkdown/KdV_Korteweg-de-Vries-km/viz/图四.png) |
|:--:|:--:|
| **Fig.9** 算子学习：不同参数下PINN预测的孤立波（实线）与精确解（虚线）对比 | **Fig.10** 参数反演：不同已知参数条件下 $h_1/h_2$ 和 $\rho_1$ 的相对误差 |
---
### 环境与依赖: 代码需要配置的环境

- Python ≥3.8
- PyTorch ≥1.12 (项目在NVIDIA RTX 6000 Ada Generation GPU上测试)
- NumPy
- Matplotlib (用于绘图)

以上依赖可通过 `requirements.txt` 一键安装：

```bash
pip install -r requirements.txt
```

### 快速安装

```bash
# 建议使用虚拟环境
conda create -n PINNs-KdV python=3.9
conda activate PINNs-KdV
pip install -r requirements.txt
```

在本教程的 Notebook 中，我们将逐步定义所需函数并构造基于纯物理约束的 PINNs 网络模型。

以下是导入库部分的代码块，请在 **Jupyter Notebook** 中依次运行以下代码块：
#### 代码单元 1：导入必要库&设置设备

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time # For timing epochs

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
```
以下是数学背景章节涉及到的代码块，包括计算KdV系数的函数、定义PINN损失的函数等，请在 **Jupyter Notebook** 中依次运行以下代码块：
#### 代码单元 2：定义KdV系数计算函数 (示例)
```python
# 示例：KdV系数的简化计算 (实际计算可能更复杂，依赖于具体物理模型)
def calculate_kdv_coefficients(rho1, rho2, h1, h2, g=9.81):
    # Placeholder for actual coefficient calculation based on two-layer fluid dynamics
    # These are illustrative and may not be physically accurate without full derivation
    c0 = np.sqrt(g * h1 * h2 * (rho2 - rho1) / (rho1 * h2 + rho2 * h1)) # Simplified example
    c1 = 1.5 * c0 * (h1 - h2) / (h1 * h2) # Simplified example for nonlinearity
    c2 = c0 * h1 * h2 / 6.0 # Simplified example for dispersion
    return c0, c1, c2
```
---


## 核心实现详解：用于逼近KdV解的PINN网络模型
本项目采用全连接神经网络（FCNN），也称为多层感知机（MLP），作为物理信息神经网络（PINN）的基础架构，用于逼近KdV方程的解 $\eta(x,t)$。

`PINN_KdV` 模型的核心功能包括：
1.  **逼近解函数**：将时空坐标 $(x,t)$ (以及在算子学习任务中的物理参数 $s$)作为输入，输出预测的波高 $\tilde{\eta}$。
2.  **自动微分**：利用PyTorch的自动微分功能计算 $\tilde{\eta}$ 相对于 $x$ 和 $t$ 的各阶导数，用于构造PDE残差。
3.  **参数化**：网络的权重和偏置 $\theta$ 是通过最小化总损失函数来优化的。在反问题中，KdV方程的部分系数或其对应的物理参数也可以作为可训练参数。

### 构造基本的 MLP 网络
在这一代码单元中，我们定义了一个标准的多层感知机（MLP）网络，用作 `PINN_KdV` 模型的基础。
- **可配置层次**：通过传入 `layers` 列表控制输入层、隐藏层和输出层的神经元数量。
- **激活函数**：通常使用 `torch.tanh` 或 `torch.sin` 作为隐藏层的激活函数。`tanh` 是本研究中常用的选择。

#### 代码单元 3：定义 MLP 网络
```python
class MLP(nn.Module):
    def __init__(self, layers):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2: # No activation after the last linear layer
                self.layers.append(nn.Tanh()) # Or other activation like nn.SiLU()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Example: layers_kdv = [2, 20, 20, 20, 20, 1] for inputs (x,t) and output eta
# model_mlp = MLP(layers_kdv).to(device)
```
---
### 构造 `PINN_KdV` 模型
`PINN_KdV` 类封装了MLP网络，并实现了计算KdV方程残差和各项损失的逻辑。

#### 1. `__init__`: 构造函数
初始化网络、KdV方程的系数 (可以是固定的，也可以是可训练的)。

请在 **Jupyter Notebook** 中依次运行以下代码块：
#### 代码单元 4： `PINN_KdV` 类构造函数和网络前向传播

```python
class PINN_KdV(nn.Module):
    def __init__(self, layers_nn, c0_val, c1_val, c2_val, for_inverse_problem=False):
        super(PINN_KdV, self).__init__()
        self.net_eta = MLP(layers_nn).to(device) # Neural network for eta

        # KdV coefficients
        # For forward problem (operator learning), these might be fixed or part of input
        # For inverse problem, they (or underlying physical params) can be torch.nn.Parameter
        if for_inverse_problem:
            # Example: making c1 and c2 learnable
            self.c0 = torch.tensor([c0_val], dtype=torch.float32, device=device, requires_grad=False)
            self.c1 = nn.Parameter(torch.tensor([c1_val], dtype=torch.float32, device=device))
            self.c2 = nn.Parameter(torch.tensor([c2_val], dtype=torch.float32, device=device))
            # Or, if learning p, h1, etc. define them as nn.Parameter
            # and compute c0, c1, c2 from them in the loss_pde function
        else:
            self.c0 = torch.tensor([c0_val], dtype=torch.float32, device=device, requires_grad=False)
            self.c1 = torch.tensor([c1_val], dtype=torch.float32, device=device, requires_grad=False)
            self.c2 = torch.tensor([c2_val], dtype=torch.float32, device=device, requires_grad=False)

    def forward(self, x_t_input):
        # x_t_input is a tensor of shape [N, 2] where columns are x and t
        # For operator learning, it might be [N, 2+num_params_s]
        eta = self.net_eta(x_t_input)
        return eta

    def pde_residual(self, x_t_input):
        # x_t_input requires_grad=True for AD
        x_coords = x_t_input[:, 0:1]
        t_coords = x_t_input[:, 1:2]

        x_coords.requires_grad_(True)
        t_coords.requires_grad_(True)
        
        # Create a combined input for the network if it expects (x,t) separately
        # or if parameters are concatenated for operator learning.
        # Assuming self.net_eta takes concatenated (x,t) or (x,t,params)
        
        # If network takes (x,t) as a single [N,2] tensor:
        combined_input = torch.cat((x_coords, t_coords), dim=1)
        if x_t_input.shape[1] > 2: # Operator learning with extra params
            extra_params = x_t_input[:, 2:]
            combined_input = torch.cat((combined_input, extra_params), dim=1)

        eta = self.net_eta(combined_input)

        eta_t = torch.autograd.grad(eta, t_coords, grad_outputs=torch.ones_like(eta), create_graph=True, retain_graph=True)[0]
        eta_x = torch.autograd.grad(eta, x_coords, grad_outputs=torch.ones_like(eta), create_graph=True, retain_graph=True)[0]
        eta_xx = torch.autograd.grad(eta_x, x_coords, grad_outputs=torch.ones_like(eta_x), create_graph=True, retain_graph=True)[0]
        eta_xxx = torch.autograd.grad(eta_xx, x_coords, grad_outputs=torch.ones_like(eta_xx), create_graph=True, retain_graph=True)[0]

        # Use self.c0, self.c1, self.c2 which might be tensors or nn.Parameters
        f_pde = eta_t + self.c0 * eta_x + self.c1 * eta * eta_x + self.c2 * eta_xxx
        return f_pde
```

#### 2. 计算损失函数
定义方法来计算 $L_{PDE}$, $L_{BC}$, $L_{data}$, 和 $L_{IC}$。

---
请在 `PINN_KdV` 类中继续输入以下代码:
#### 代码单元 5： `PINN_KdV` 类损失计算方法
```python
# Continuing the PINN_KdV class
# ... (previous __init__, forward, pde_residual methods) ...

    def loss(self, x_t_pde, x_t_ic, eta_ic, x_t_bc, eta_bc_target, x_t_data, eta_data_observed, weights):
        # PDE loss
        f_pred = self.pde_residual(x_t_pde)
        loss_pde = torch.mean(f_pred**2)

        # Initial condition loss
        eta_pred_ic = self.net_eta(x_t_ic) # Assuming x_t_ic has t=0
        loss_ic = torch.mean((eta_pred_ic - eta_ic)**2)
        
        # Boundary condition loss
        eta_pred_bc = self.net_eta(x_t_bc)
        loss_bc = torch.mean((eta_pred_bc - eta_bc_target)**2)
        
        # Data loss (for observations or operator learning targets)
        loss_data = torch.tensor(0.0, device=device) # Default to 0 if no data points
        if x_t_data is not None and eta_data_observed is not None:
            eta_pred_data = self.net_eta(x_t_data)
            loss_data = torch.mean((eta_pred_data - eta_data_observed)**2)

        total_loss = weights['pde'] * loss_pde + \
                     weights['ic'] * loss_ic + \
                     weights['bc'] * loss_bc + \
                     weights['data'] * loss_data
        
        return total_loss, loss_pde, loss_ic, loss_bc, loss_data

# Example instantiation for a forward problem
# layers = [2, 20, 20, 20, 20, 1] # For (x,t) input
# c0, c1, c2 = 1.0, 6.0, 1.0 # Example coefficients
# model = PINN_KdV(layers, c0, c1, c2).to(device)

# Example instantiation for an inverse problem (learning c1, c2)
# initial_c1_guess, initial_c2_guess = 5.0, 0.5
# model_inverse = PINN_KdV(layers, c0, initial_c1_guess, initial_c2_guess, for_inverse_problem=True).to(device)

```

> 💡 **解读：**
> - **`forward` 方法**：接收时空坐标（和可能的物理参数 $s$），通过MLP网络输出波高预测 $\tilde{\eta}$。
> - **`pde_residual` 方法**：核心的物理约束实现。它对输入坐标 `x_t_input` (其中包含 $x$ 和 $t$) 设置 `requires_grad=True`，以便PyTorch能够追踪这些张量的操作并计算梯度。然后，它调用 `torch.autograd.grad` 多次来获得 $\tilde{\eta}$ 对 $t$ 的一阶导数，以及对 $x$ 的一至三阶导数。最后，将这些导数代入KdV方程的表达式得到PDE残差 $R(x,t)$。
> - **`loss` 方法**：计算各项损失（PDE、初始条件、边界条件、数据匹配）的均方误差，并根据给定的权重将它们组合成总损失。

> 🔍 **提示：**
> - 对于算子学习，输入 `x_t_input` 给 `net_eta` 可能需要包含物理参数 $s=(p, h_1, a)$，例如，网络输入层大小为 `2 + num_params_s`。`pde_residual` 和 `loss` 方法中的 `x_t_pde`, `x_t_ic`, `x_t_bc`, `x_t_data` 张量也应包含这些参数列。
> - KdV系数 $c_0, c_1, c_2$ 在算子学习中可能根据输入的 $s$ 动态计算，或者如果网络学习从 $s$ 到解的直接映射，这些系数可以被隐式学习。在反问题中，待求的系数或物理参数会被定义为 `nn.Parameter`，这样优化器才能更新它们。

---

## 模型训练与评估
模型训练的目标是找到一组神经网络参数 $\theta$ (以及在反问题中的物理参数) 使得总损失函数 $L_{total}$ 最小化。评估则关注模型预测的准确性以及物理约束的满足程度。

### 数据准备与加载
训练PINN需要几类数据点：
1.  **PDE配置点 ($X_{PDE}$)**: 在计算域 $(x,t) \in \Omega$ 内部随机或规则采样得到的点，用于计算 $L_{PDE}$。
2.  **初始条件点 ($X_{IC}$)**: 在初始时刻 $t=0$ (或指定的初始时间) 沿空间轴 $x$ 采样得到的点，以及对应的初始波形 $\eta_0(x)$，用于计算 $L_{IC}$。
3.  **边界条件点 ($X_{BC}$)**: 在计算域的空间边界（例如 $x_{min}, x_{max}$）和时间跨度内采样得到的点，以及对应的边界目标值，用于计算 $L_{BC}$。
4.  **观测数据点 ($X_{data}$)** (可选):
    *   **对于算子学习**: 从不同物理参数 $s$ 下的KdV精确解中采样得到的 $(x,t,\eta)$ 点，用于计算 $L_{data}$。
    *   **对于反问题**: 稀疏的实验测量数据 $(x_d, t_d, \eta_{obs})$，用于计算 $L_{data}$。

#### 代码单元 6: 数据生成/加载函数 (示例)
```python
def get_exact_soliton_solution(x, t, c0, c1, c2, amplitude, x0=0):
    """Generates exact KdV soliton solution."""
    # speed c = c0 + 1/3 * amplitude * c1
    # lambda_width_sq = (12 * c2) / (amplitude * c1)
    # This simplified version assumes c0 is the full phase velocity for the moving frame part
    # A more precise form for X = x - (c0 + 1/3 * a * c1)t
    # For this example, let's assume a simple form for X where velocity is given
    # Effective velocity for the soliton peak
    wave_speed = c0 + (amplitude * c1) / 3.0
    lambda_val = np.sqrt((12.0 * c2) / (amplitude * c1 + 1e-9)) # add small epsilon for stability
    
    X = x - wave_speed * t - x0 # x0 is initial position of peak
    eta_exact = amplitude * (1.0 / np.cosh(X / lambda_val))**2
    return eta_exact

def sample_training_data(N_ic, N_bc, N_pde, N_data, t_domain, x_domain, kdv_params, op_learn_params=None):
    # Domain boundaries
    t_min, t_max = t_domain
    x_min, x_max = x_domain
    c0, c1, c2, amplitude, initial_peak_x0 = kdv_params # For exact solution if needed

    # Initial condition points (t=t_min)
    x_ic = torch.linspace(x_min, x_max, N_ic, device=device).unsqueeze(-1)
    t_ic = torch.full_like(x_ic, t_min)
    xt_ic = torch.cat([x_ic, t_ic], dim=1)
    # For IC, use exact solution at t=t_min
    eta_ic = get_exact_soliton_solution(x_ic.cpu().numpy(), t_ic.cpu().numpy(), c0, c1, c2, amplitude, initial_peak_x0)
    eta_ic = torch.from_numpy(eta_ic).float().to(device)
    xt_ic.requires_grad_(True) # For pde_residual if some IC points are also PDE points

    # Boundary condition points (x=x_min and x=x_max for all t)
    t_bc = torch.linspace(t_min, t_max, N_bc // 2, device=device).unsqueeze(-1)
    x_bc_left = torch.full_like(t_bc, x_min)
    x_bc_right = torch.full_like(t_bc, x_max)
    xt_bc_left = torch.cat([x_bc_left, t_bc], dim=1)
    xt_bc_right = torch.cat([x_bc_right, t_bc], dim=1)
    xt_bc = torch.cat([xt_bc_left, xt_bc_right], dim=0)
    eta_bc_target = torch.zeros(xt_bc.shape[0], 1, device=device) # Soliton decays at boundaries
    xt_bc.requires_grad_(True)

    # PDE collocation points (randomly sampled in the domain)
    x_pde = torch.rand(N_pde, 1, device=device) * (x_max - x_min) + x_min
    t_pde = torch.rand(N_pde, 1, device=device) * (t_max - t_min) + t_min
    xt_pde = torch.cat([x_pde, t_pde], dim=1)
    xt_pde.requires_grad_(True)

    # Data points (e.g., from exact solution or observations)
    xt_data, eta_data = None, None
    if N_data > 0:
        # Example: sample from exact solution for forward problem / operator learning
        x_data_samples = torch.rand(N_data, 1, device=device) * (x_max - x_min) + x_min
        t_data_samples = torch.rand(N_data, 1, device=device) * (t_max - t_min) + t_min
        xt_data = torch.cat([x_data_samples, t_data_samples], dim=1)
        eta_data_np = get_exact_soliton_solution(x_data_samples.cpu().numpy(), t_data_samples.cpu().numpy(),
                                                c0, c1, c2, amplitude, initial_peak_x0)
        eta_data = torch.from_numpy(eta_data_np).float().to(device)
        xt_data.requires_grad_(True) # Usually not needed for data points, but harmless

    # For operator learning, concatenate physical parameters s to all xt tensors
    if op_learn_params is not None: # op_learn_params = (p_val, h1_val, a_val) tensor
        s_tensor = torch.tensor(op_learn_params, device=device).float().unsqueeze(0)
        
        def concat_s(xt_tensor):
            if xt_tensor is None: return None
            s_repeated = s_tensor.repeat(xt_tensor.shape[0], 1)
            return torch.cat([xt_tensor, s_repeated], dim=1)

        xt_ic = concat_s(xt_ic)
        xt_bc = concat_s(xt_bc)
        xt_pde = concat_s(xt_pde)
        xt_data = concat_s(xt_data)

    return xt_pde, xt_ic, eta_ic, xt_bc, eta_bc_target, xt_data, eta_data

# Example usage:
# t_domain = (0.0, 1.0)
# x_domain = (-10.0, 10.0)
# kdv_c = [1.0, 6.0, 1.0, 2.0, 0.0] # c0, c1, c2, amplitude, x0 for exact solution
# N_ic, N_bc, N_pde, N_data = 100, 100, 10000, 500
# xt_pde, xt_ic, eta_ic, xt_bc, eta_bc_target, xt_data, eta_data = \
#     sample_training_data(N_ic, N_bc, N_pde, N_data, t_domain, x_domain, kdv_c)
```

---
### 训练函数
训练过程通常使用Adam优化器。在每个epoch中，模型基于采样点计算总损失，然后通过反向传播更新网络参数（和可学习的物理参数）。

#### 代码单元 7: 定义训练循环
```python
def train_pinn_kdv(model, optimizer, epochs, training_data, loss_weights):
    xt_pde, xt_ic, eta_ic, xt_bc, eta_bc_target, xt_data, eta_data_observed = training_data
    
    loss_history = {'total': [], 'pde': [], 'ic': [], 'bc': [], 'data': []}
    
    # For inverse problem, if c1, c2 (or other phys params) are learnable, track them
    # param_history = {'c1': [], 'c2': []} # Example

    for epoch in range(epochs):
        model.train() # Set model to training mode
        optimizer.zero_grad()
        
        total_loss, loss_pde, loss_ic, loss_bc, loss_data = model.loss(
            xt_pde, xt_ic, eta_ic, xt_bc, eta_bc_target, 
            xt_data, eta_data_observed, loss_weights
        )
        
        total_loss.backward()
        optimizer.step()
        
        loss_history['total'].append(total_loss.item())
        loss_history['pde'].append(loss_pde.item())
        loss_history['ic'].append(loss_ic.item())
        loss_history['bc'].append(loss_bc.item())
        loss_history['data'].append(loss_data.item() if isinstance(loss_data, torch.Tensor) else loss_data) # data_loss could be float 0.0

        # if hasattr(model, 'c1') and isinstance(model.c1, nn.Parameter): # Track learnable params
        #    param_history['c1'].append(model.c1.item())
        # if hasattr(model, 'c2') and isinstance(model.c2, nn.Parameter):
        #    param_history['c2'].append(model.c2.item())

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}/{epochs} - Total Loss: {total_loss.item():.4e}, "
                  f"PDE: {loss_pde.item():.4e}, IC: {loss_ic.item():.4e}, "
                  f"BC: {loss_bc.item():.4e}, Data: {loss_data.item() if isinstance(loss_data, torch.Tensor) else loss_data:.4e}")
            # if hasattr(model, 'c1') and isinstance(model.c1, nn.Parameter):
            #    print(f"  c1_est: {model.c1.item():.4f}, c2_est: {model.c2.item():.4f}")
                
    return loss_history #, param_history

```

---
### 模型评估
评估模型通常包括：
- **损失曲线**: 观察总损失及各分量损失随训练迭代的变化。
- **预测精度 (前向问题/算子学习)**: 将PINN的预测结果与精确解或高质量数值解进行比较，计算相对 $L_2$ 误差等指标。如kdv-README中提到的算子学习预测误差低至 $10^{-4}$。
- **参数估计精度 (反问题)**: 将PINN反演得到的物理参数与真实值进行比较，计算相对误差。如kdv-README中图4、图6-9所示的参数估计结果。
- **物理一致性**: 检查解是否满足物理规律，例如孤立波的形状、传播速度是否合理，PDE残差是否足够小。

#### 损失曲线
绘制训练过程中各项损失的变化有助于诊断训练过程。

#### 代码单元 8： 绘制损失曲线
```python
def plot_loss_history(loss_history, title="Loss History"):
    plt.figure(figsize=(10, 6))
    for key, value in loss_history.items():
        if key != 'total' and len(value) > 0 and value[0] is not None: # Skip if not tracked or empty
             # Plot only if there are significant values to show, avoid plotting flat zero lines for unused losses
            if np.any(np.array(value) > 1e-9) or key == 'pde': # Always plot PDE for reference
                plt.plot(value, label=key.capitalize() + ' Loss')
    plt.plot(loss_history['total'], label='Total Loss', linewidth=2, color='black')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.yscale('log')
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()

# Example usage:
# plot_loss_history(loss_hist_fwd, title="Loss History (Forward Problem)")
# if 'loss_hist_inv' in locals():
#    plot_loss_history(loss_hist_inv, title="Loss History (Inverse Problem)")
```
---
#### 物理评估指标
对于算子学习，可以评估预测波形与真实波形的均方误差或相对误差。
对于反问题，可以评估估计参数的相对误差： $E_{rel} = |\frac{\theta_{estimated} - \theta_{true}}{\theta_{true}}|$。

| ![](https://www.science42.tech/cases/caseMarkdown/KdV_Korteweg-de-Vries-km/viz/表1.png) |
|:--:|
| **Fig.11** 算子学习：PINN预测的孤立波与真实解的相对 $L_2$ 误差 (改编自kdv-README 表I) |


| ![](https://www.science42.tech/cases/caseMarkdown/KdV_Korteweg-de-Vries-km/viz/图四.png) |
|:--:|
| **Fig.12** 参数反演性能：不同已知信息下 `h₁/h₂` 与 `ρ₁` 的相对误差 (kdv-README 图4) |


---
## 结果可视化
可视化是理解PINN模型性能和KdV方程解的关键。

### 波形预测
比较PINN预测的波形 $\tilde{\eta}(x,t)$ 与精确解或观测数据。

#### 代码单元 9： 绘制波形预测与比较
```python
def plot_wave_comparison(model, t_slice, x_domain, kdv_params_true, op_learn_params_slice=None, title_suffix=""):
    model.eval() # Set model to evaluation mode
    x_min, x_max = x_domain
    C0_TRUE, C1_TRUE, C2_TRUE, AMP_TRUE, X0_TRUE = kdv_params_true

    x_dense = torch.linspace(x_min, x_max, 500, device=device).unsqueeze(-1)
    t_dense = torch.full_like(x_dense, t_slice)
    xt_dense_input = torch.cat([x_dense, t_dense], dim=1)

    # If operator learning model, append the parameters s
    if op_learn_params_slice is not None:
        s_tensor = torch.tensor(op_learn_params_slice, device=device).float().unsqueeze(0).repeat(xt_dense_input.shape[0], 1)
        xt_dense_input_model = torch.cat([xt_dense_input, s_tensor], dim=1)
    else:
        xt_dense_input_model = xt_dense_input
        
    with torch.no_grad():
        eta_pred = model(xt_dense_input_model).cpu().numpy()

    eta_exact = get_exact_soliton_solution(x_dense.cpu().numpy(), t_dense.cpu().numpy(),
                                           C0_TRUE, C1_TRUE, C2_TRUE, AMP_TRUE, X0_TRUE)

    plt.figure(figsize=(10, 6))
    plt.plot(x_dense.cpu().numpy(), eta_exact, 'k-', label='Exact Solution', linewidth=2)
    plt.plot(x_dense.cpu().numpy(), eta_pred, 'r--', label='PINN Prediction', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('$\eta(x,t)$')
    plt.title(f'Wave Profile Comparison at t={t_slice:.2f} {title_suffix}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage:
# t_vis = 0.5 # Time slice for visualization
# C0_TRUE, C1_TRUE, C2_TRUE, AMP_TRUE, X0_TRUE = kdv_params_true # Defined earlier
# plot_wave_comparison(model_fwd, t_vis, x_domain, kdv_params_true, title_suffix="(Forward Problem)")
# if 'model_inv' in locals():
#    plot_wave_comparison(model_inv, t_vis, x_domain, kdv_params_true, title_suffix="(Inverse Problem)")
```
运行上述代码将生成波形对比图。

| ![](https://www.science42.tech/cases/caseMarkdown/KdV_Korteweg-de-Vries-km/viz/图三.png) |
|:--:|
| **Fig.13** 算子学习示例：PINN预测的波形（实线）与精确解（虚线）对比 (kdv-README 图3) |

### 参数反演结果可视化
对于逆问题，可以可视化：
- **参数可辨识性**: 如[Fig.2](#fig2) / [Fig.5](#fig5)所示，展示不同条件下参数估计的误差。
- **数据效率与空间稀疏性**: 如kdv-README中图6、图7所示，展示在稀疏数据下参数估计的准确性。
- **最优传感器布放**: 如kdv-README中图8所示，展示不同传感器位置对参数估计的影响。
- **噪声鲁棒性**: 如kdv-README中图9所示，展示在含噪声数据下参数估计的性能。

| ![](https://www.science42.tech/cases/caseMarkdown/KdV_Korteweg-de-Vries-km/viz/图六.png) | ![](https://www.science42.tech/cases/caseMarkdown/KdV_Korteweg-de-Vries-km/viz/图7.png) |
|:--:|:--:|
| **Fig.14** 数据稀疏性：参数估计相对误差随观测点数量 $N_x$ 的变化 (kdv-README 图6) | **Fig.15** 单点监测：$x=0$ 处单点监测的参数估计误差 (kdv-README 图7) |

| ![](https://www.science42.tech/cases/caseMarkdown/KdV_Korteweg-de-Vries-km/viz/图八.png) | ![](https://www.science42.tech/cases/caseMarkdown/KdV_Korteweg-de-Vries-km/viz/图九.png) |
|:--:|:--:|
| **Fig.16** 传感器布放：测量窗口中心对参数估计误差的影响 (kdv-README 图8) | **Fig.17** 噪声鲁棒性：参数估计误差随观测数据噪声水平的变化 (kdv-README 图9) |


## 主函数与数据加载
主程序将整合数据准备、模型初始化、训练和评估/可视化等步骤。

#### 代码单元 10： 构造主函数 (示例结构)
下面是在 Notebook 中运行一个完整实验的示例结构：
```python
def run_kdv_pinn_experiment(config):
    # 1. Configuration
    problem_type = config.get('problem_type', 'forward') # 'forward', 'inverse', 'operator_learning'
    layers = config['layers']
    kdv_coeffs_true = config['kdv_coeffs_true'] # [c0, c1, c2, amplitude, x0_initial]
    
    t_domain = config['t_domain']
    x_domain = config['x_domain']
    
    N_ic = config['N_ic']
    N_bc = config['N_bc']
    N_pde = config['N_pde']
    N_data = config.get('N_data', 0) # Number of data points for L_data

    epochs = config['epochs']
    lr = config['lr']
    loss_weights = config['loss_weights']

    print(f"--- Running KdV PINN Experiment: {problem_type} ---")
    print(f"Device: {device}")

    # 2. Model Initialization
    if problem_type == 'inverse':
        c0_known = kdv_coeffs_true[0]
        c1_guess = config.get('c1_guess', 5.0)
        c2_guess = config.get('c2_guess', 0.5)
        model = PINN_KdV(layers, c0_known, c1_guess, c2_guess, for_inverse_problem=True).to(device)
        print(f"Inverse problem: Learning c1 (guess: {c1_guess}), c2 (guess: {c2_guess})")
    elif problem_type == 'operator_learning':
        # For operator learning, network input layers should be layers[0] = 2 + num_op_params
        # KdV coefficients c0,c1,c2 might be fixed or dynamically computed from op_params in loss
        # This example assumes fixed c0,c1,c2 for simplicity in PINN_KdV,
        # and op_params are concatenated to x,t inputs.
        num_op_params = config.get('num_op_params', 0) # e.g., density_ratio, h1, amplitude
        model_input_dim = 2 + num_op_params
        op_layers = [model_input_dim] + layers[1:]
        # Assuming c0,c1,c2 are fixed for this basic operator learning PINN example.
        # A more advanced operator learner might take c0,c1,c2 as inputs or learn them implicitly.
        model = PINN_KdV(op_layers, kdv_coeffs_true[0], kdv_coeffs_true[1], kdv_coeffs_true[2], for_inverse_problem=False).to(device)
        print(f"Operator learning: Input dim {model_input_dim}")
    else: # Forward problem
        model = PINN_KdV(layers, kdv_coeffs_true[0], kdv_coeffs_true[1], kdv_coeffs_true[2], for_inverse_problem=False).to(device)
        print("Forward problem: Solving KdV with known coefficients.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 3. Data Preparation
    # For operator learning, this sampling would iterate over a dataset of op_learn_params
    # Here, simplified for a single set of true params.
    # op_learn_params_instance = config.get('op_learn_params_instance', None) # e.g. [rho_ratio_val, h1_val, amp_val]
    # training_data = sample_training_data(N_ic, N_bc, N_pde, N_data,
    #                                     t_domain, x_domain, kdv_coeffs_true, op_learn_params_instance)
    
    # Simplified data sampling for one run (forward/inverse)
    # In a real operator learning setup, you'd loop through different (p,h1,a) and their corresponding c0,c1,c2,amp
    current_kdv_params_for_data = kdv_coeffs_true # Use true params to generate IC/BC/Data for this run
    
    # For inverse problems, N_data points are crucial and come from "observations"
    # These observations would ideally be generated using kdv_coeffs_true
    training_data = sample_training_data(N_ic, N_bc, N_pde, N_data,
                                         t_domain, x_domain, current_kdv_params_for_data)


    # 4. Training
    start_time = time.time()
    loss_history = train_pinn_kdv(model, optimizer, epochs, training_data, loss_weights)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds.")

    # 5. Evaluation & Visualization
    plot_loss_history(loss_history, title=f"Loss History ({problem_type})")
    
    t_vis = (t_domain[0] + t_domain[1]) / 2.0 # Visualize at midpoint of time domain
    plot_wave_comparison(model, t_vis, x_domain, kdv_coeffs_true, # op_learn_params_instance for op_learn
                         title_suffix=f"({problem_type})")

    if problem_type == 'inverse':
        print(f"True c1: {kdv_coeffs_true[1]:.4f}, Estimated c1: {model.c1.item():.4f}, Relative Error: {abs(model.c1.item() - kdv_coeffs_true[1])/abs(kdv_coeffs_true[1]):.2%}")
        print(f"True c2: {kdv_coeffs_true[2]:.4f}, Estimated c2: {model.c2.item():.4f}, Relative Error: {abs(model.c2.item() - kdv_coeffs_true[2])/abs(kdv_coeffs_true[2]):.2%}")

    print(f"--- Experiment Ended ---")
    return model, loss_histor

```

## 使用示例

1. 进入秋月白页面，点击聊天框上方的 **方程求解** 图标
<!-- 添加图片方式：纯Markdown 格式 -->
| ![](https://www.science42.tech/cases/caseMarkdown/KdV_Korteweg-de-Vries-km/viz/图十六.png) | 
|:--:|
| **Fig.1 使用示例** |

2. 点击 **上传文件** 图标，上传数据文件（可选）（本项目不需要上传数据文件）

   支持格式：**CNPZ**。  
   如果没有数据文件，模型会**自动生成仿真数据**用于训练和推理。

| ![](https://www.science42.tech/cases/caseMarkdown/KdV_Korteweg-de-Vries-km/viz/图十七.png) |
|:--:|
| **Fig.2 上传文件示例** |


3. 如果没有数据文件，模型会自动生成仿真数据用于训练和推理。

---

## 交互式示例流程（图示）

### A. **算子学习（Operator Learning）**

1. 在对话框输入：**“求解kdv算子学习方法”**

| ![](https://www.science42.tech/cases/caseMarkdown/KdV_Korteweg-de-Vries-km/viz/图十.png) |
|:--:|
| **Fig.3 在秋月白上的提问（算子学习）** |  

2. 模型将先进行**分析与规划**，随后自动给出**完整的 PINN 建模实现示例（Python 代码）**  

| ![](https://www.science42.tech/cases/caseMarkdown/KdV_Korteweg-de-Vries-km/viz/图十一.png) |
|:--:|
| **Fig.4 模型分析规划与完整代码（算子学习）** |
3. 运行代码并训练，输出 **训练损失曲线**  

| ![](https://www.science42.tech/cases/caseMarkdown/KdV_Korteweg-de-Vries-km/viz/图十二.png) |
|:--:|
| **Fig.5 训练过程的 Loss 曲线（算子学习）** |

---

### B. **反问题求解（Inverse Problem）**

1. 在对话框输入：**“求解kdv逆向问题方法”** 

| ![](https://www.science42.tech/cases/caseMarkdown/KdV_Korteweg-de-Vries-km/viz/图十三.png) |
|:--:|
| **Fig.6 在秋月白上的提问（逆向问题）** |

2. 模型将先进行**分析与规划**，随后自动给出**完整的 PINN 逆问题建模示例（Python 代码）**  

| ![](https://www.science42.tech/cases/caseMarkdown/KdV_Korteweg-de-Vries-km/viz/图十四.png) |
|:--:|
| **Fig.7 模型分析规划与完整代码（逆向问题）** |

3. 运行代码并训练，输出 **预测值与真实值的对比图**  

| ![](https://www.science42.tech/cases/caseMarkdown/KdV_Korteweg-de-Vries-km/viz/图十五.png) |
|:--:|
| **Fig.8 预测值与真实值对比（逆向问题）** |

---
## 贡献与许可证

恭喜您完成本项目的基本 Tutorial 结构学习！您可以使用本文提供的模块化代码在您自己的 Notebook 中运行各个代码单元，并根据实际需求调整参数（如KdV系数、网络结构、采样点数量、损失权重）或代码结构以解决更复杂的KdV相关问题，包括更精细的算子学习或多参数反演。

**贡献者：**
- Ming Kang
- Yuhao Ma

**许可证：**  
本项目代码遵循 [GNU 通用公共许可证 第三版（GPLv3）](https://www.gnu.org/licenses/gpl-3.0.html)。您可以自由使用、修改和分发该代码，但需保留原始版权声明，并在分发时提供相同的许可证。

版权 © 硒钼科技（北京）。更多信息请访问 [Science42](https://www.science42.tech/#/index)。


## 文件结构（示例）


```text
KDV_INVERSE_PROBLEM/
│
├── KDV_Inverse_problem_main.py  # 🔹 主入口脚本，可选择 train / quick 模式
├── Readme.md                    # 📖 项目说明文档
│
├── .dist/                        # 🗂️ 临时或打包相关目录（可忽略）
│
├── data/                         # 🗃️ 数据目录
│   └── heat_eq_data.npz          # 🔹 保存初始数据/训练数据的 NumPy 压缩文件
│
├── model/                        # 🧠 模型权重目录
│   └── trained_model.pth         # ✅ 已训练完成的 PyTorch 模型权重
│
├── results/                      # 📊 保存训练和推理结果
│   ├── all_results.txt            # 🔹 汇总训练/推理结果的文本文件
│   ├── quick_inference_curve.png # 🔹 Quick 模式预测曲线 + 训练点 + 参考曲线
│   ├── train_fit_curve.png        # 🔹 训练拟合对比曲线 (t=1.0)
│   └── training_result.txt        # 🔹 训练过程中记录的日志或变量值
│
├── src/                           # 🗂️ 核心源码模块
│   ├── __pycache__/               # 🔹 Python 缓存目录，自动生成
│   ├── data_utils.py              # 🔹 数据处理工具函数
│   ├── data.py                    # 🔹 数据生成模块，提供 gen_testdata() 等函数
│   ├── net.py                     # 🔹 神经网络构建模块，create_network()
│   ├── pinn_solver.py             # 🔹 PINN 训练/推理相关函数，PDE/BC 定义
│   └── runner.py                  # 🔹 封装训练和快速推理的调用函数
