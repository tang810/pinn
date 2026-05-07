
<!-- 
text: "亥姆霍兹方程求解",
area: "亥姆霍兹方程在声学、电磁学和光学等波动问题中广泛应用。本项目使用PINN方法求解任意维度的亥姆霍兹方程，建模定常波场。",
tags: [
  "基础方程",
  "波动问题",
  "物理信息神经网络(PINN)",
  "亥姆霍兹方程",
],
search: ["基础方程", "声学工程","电磁学"] 
-->


# HelmholtzPDE: 亥姆霍兹方程

## 项目概述

亥姆霍兹方程是描述定常波场的重要控制方程，广泛应用于声学、光学、电磁场等波动问题的建模。传统数值方法在复杂几何和高维场景中计算代价高、难以适配多物理场耦合。

本项目实现了一个支持 **任意维度（1D / 2D / 3D）** 的物理信息神经网络（Physics-Informed Neural Networks, PINNs）框架，用于求解亥姆霍兹方程。框架通过自动构造对应维度的控制方程残差，结合波数参数，实现对谐波波动问题的数值建模。

模型以空间坐标 $(x, y, z)$ 为输入，输出对应点的场变量 $u$，并同时约束其满足：

- 亥姆霍兹控制方程

> 🧠 **说明：** 本项目可应用于声波、光波和电磁波的频域建模场景，适用于复杂几何与任意边界条件的高维波动问题。

## 环境与依赖

- Python ≥3.8
- PyTorch ≥1.12

以上依赖可通过 `requirements.txt` 一键安装：

```bash
pip install -r requirements.txt
```

## 快速安装

```bash
# 建议使用虚拟环境
conda create -n PINN-Helmholtz python=3.9
conda activate PINN-Helmholtz
pip install -r requirements.txt
```

## 数学背景

亥姆霍兹方程描述了稳态波动问题，可统一表示为：

$$
\nabla^2 u + k^2 u = 0
$$

其中：

* $u$ 是场变量（声压、电场等）；
* $k$ 是波数。

本项目支持 $d=1,2,3$ 三种情形，自动构造对应维度下的物理残差，并通过 PINN 框架实现端到端建模与求解。

> 🔍 **提示：**
>
> * 该方程的二阶导数可通过自动微分统一计算，无需手动推导各维形式。
> * 使用 PINN 框架求解时，可将该残差作为物理损失项。

## 控制方程构造

为了自动适配 1D、2D 和 3D 不同空间维度的波动问题，本项目定义了一个通用的 `HelmholtzPDE` 类，用于构造物理残差项。

### 代码单元 1：控制方程残差构造类

```python
class HelmholtzPDE:
    def __init__(self, k):
        self.k = k

    def compute_pde_residuals(self, u, coords):
        ...
        return residual
```

该类支持自动计算以下项：

* 所有坐标分量的二阶导数（Laplacian）
* 波数项 $k^2 u$

最终 `.compute_pde_residuals()` 方法返回：

$$
f_{\text{Helmholtz}} = \nabla^2 u + k^2 u
$$

> 💡 **解读：**
>
> * **功能**：该类构造器自动适配任意维度的亥姆霍兹方程残差。
> * **用法**：结合 PINN 输出 $u$ 可直接传入 `.compute_pde_residuals()` 接口，获得损失函数中物理残差部分。

## PDE残差构造

在 `HelmholtzPDE` 类中，`.compute_pde_residuals()` 方法实现了控制方程的残差构造，接受神经网络输出的场变量 $u$，并计算每个物理控制方程的残差，作为 PINN 的物理损失项。


### 残差输出形式

最终 `.compute_pde_residuals()` 返回 shape: `[N, 1]` 的残差张量。

> 🔍 **提示：**
>
> * 所有导数项均通过自动微分 `torch.autograd.grad` 实现，确保残差计算在计算图中连续可导。
> * 该实现框架可无缝适配不同空间维度，并适用于复杂边界或任意初始条件场景。

## 调用示例

以下为 `HelmholtzPDE` 控制方程模块在 PINN 框架中的典型调用示例，展示如何构造残差并用于损失函数计算：

### 代码单元 2：调用控制方程模块计算残差与损失

```python
from Math_Physics_Formulations.helmholtz_pde import HelmholtzPDE

# 初始化Helmholtz方程
helmholtz = HelmholtzPDE(k=3.14)

# 神经网络预测输出
coords = ...    # shape: [N, D]
u = ...         # shape: [N, 1]

# 计算残差
residuals = helmholtz.compute_pde_residuals(u, coords)

# 计算总物理损失
loss = (residuals**2).mean()
```

## 贡献与许可证

恭喜您完成本项目的全部 Tutorial 学习！您可以使用本文提供的模块化代码在您自己的 Notebook 中运行各个代码单元，并根据实际需求调整参数或代码结构。

**贡献者：**

* Yuhao Ma

**许可证：**
本项目代码遵循 [GNU 通用公共许可证 第三版（GPLv3）](https://www.gnu.org/licenses/gpl-3.0.html)。您可以自由使用、修改和分发该代码，但需保留原始版权声明，并在分发时提供相同的许可证。

版权 © 硒钼科技（北京）。更多信息请访问 [Science42](https://www.science42.tech/#/index)。

