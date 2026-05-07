<!--
text: "孤立波形高速代理模型：KdV参数扫描神经算子"
area: "本项目面向KdV孤立波参数空间的批量预测任务，利用物理信息神经网络学习密度比、层深、波幅等输入参数到孤立波形的映射关系。案例重点展示如何构建可复用的神经算子代理模型，用一次训练支撑多组参数下的波形快速推理，适用于参数扫描、灵敏度分析和非线性波动代理建模场景。"
tags: ["流体力学", "KdV方程", "神经算子", "参数扫描", "PINN", "孤立波"]
search: ["孤立波代理模型", "KdV参数扫描", "神经算子", "波形快速预测", "流体力学"],
-->

# 孤立波形高速代理模型：KdV参数扫描神经算子

## 项目概述

Korteweg-de Vries (KdV) 方程是描述浅水波、等离子体孤子等非线性波动行为的基本偏微分方程（PDE），广泛应用于流体力学、等离子体物理和非线性科学等领域。传统数值方法在每组物理参数下通常都需要重新求解，面对密度比、层深、初始波幅等多参数组合时，参数扫描成本较高。

本项目聚焦 **孤立波形高速代理建模**：基于PINN框架学习从物理参数到孤立波解的映射关系，并支持仿真样本生成、模型训练、快速推理和结果可视化。与单个方程实例的正问题求解不同，本案例强调在一组参数空间中复用训练后的神经网络，实现对不同孤立波形的快速预测。

模型围绕空间坐标、密度比 $\rho$、层深 $h_1$、初始波幅 $a$ 等输入信息，学习对应的参考孤立波形 $u$（如水面高度或界面位移）。在 `quick` 模式下，可直接生成示例参数族并输出预测曲线。

> 🧠 **说明：**  
> 本项目为 **KdV参数化算子学习案例**，重点不是反演未知物理系数，而是学习“参数 $\rightarrow$ 波形”的快速代理模型，适用于孤立波参数扫描、批量预测和非线性波动代理建模。

---

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
conda create -n PINN-KdV python=3.9
conda activate PINN-KdV
pip install -r requirements.txt
```

---

在本教程的 Notebook 或脚本中，我们将逐步定义所需函数并构造基于物理约束的 PINN 网络模型。

#### 代码单元 1：导入必要库 & 设置设备

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
```

---

## 数学背景

KdV方程的标准形式为：

$$
\frac{\partial u}{\partial t} + 6u \frac{\partial u}{\partial x} + \frac{\partial^3 u}{\partial x^3} = 0
$$

其中：
- $u(x, t)$ 为扰动量（如水面高度、等离子体密度等）；
- $\frac{\partial u}{\partial t}$ 为时间一阶导数；
- $\frac{\partial u}{\partial x}$ 为空间一阶导数，$\frac{\partial^3 u}{\partial x^3}$ 为空间三阶导数；
- $6u \frac{\partial u}{\partial t}$ 为非线性项。

---

## 控制方程构造

为自动适配KdV方程的物理残差项，本项目定义了 `KdV` 类，用于构造PDE残差：

- 时间一阶导数 $\frac{\partial u}{\partial t}$
- 空间一阶导数 $\frac{\partial u}{\partial x}$
- 空间三阶导数 $\frac{\partial^3 u}{\partial x^3}$
- 非线性项 $6u \frac{\partial u}{\partial t}$

该类通过自动微分动态计算所需导数，并返回KdV方程的残差。

#### 代码单元 2：KdV方程残差构造类

```python
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
```

> 💡 **解读：**
> * **功能**：该类自动计算KdV方程的所有导数项和非线性项，输出残差。
> * **用法**：结合PINN输出 $u$ 可直接传入 `.pde()` 接口，获得损失函数中物理残差部分。

---

## PDE残差构造

在 `KdV` 类中，`.pde()` 方法实现了控制方程的残差构造。它接受神经网络输出的扰动量 $u$，并计算每个物理控制方程的残差，作为 PINN 的物理损失部分。

### 残差表达式

根据扰动 $u$ 构造对应的KdV残差：

$$
f_{\text{KdV}} = \frac{\partial u}{\partial t} + 6u \frac{\partial u}{\partial x} + \frac{\partial^3 u}{\partial x^3}
$$

- **时间一阶导数**：通过对 $t$ 求导获得。
- **空间三阶导数**：通过三次对 $x$ 求导获得。
- **非线性项**：$6u \frac{\partial u}{\partial x}$。

最终 `.pde()` 返回 $f_{\text{KdV}}$，可直接用于 PINN 损失。

---

> 🔍 **提示：**
>
> * 所有导数项均通过自动微分 `torch.autograd.grad` 实现，确保残差计算在计算图中连续可导。
> * 该实现适用于任意初值和边界条件的KdV问题。

## 调用示例

以下为 `KdV` 控制方程模块在 PINN 框架中的典型调用示例，展示如何构造残差并用于损失函数计算：

#### 代码单元 3：调用控制方程模块计算残差与损失

```python
from Math_Physics_Formulations.kdv import KdV

# 初始化KdV方程
solver = KdV()

# 输入的坐标点
coords = ...  # shape: [N, 2], 列为 [x, t]

# 获取网络预测结果  
u = neural_net(coords)  # shape: [N, 1]

# 计算残差  
residual = solver.pde(coords, u)

# 计算损失
loss = (residual**2).mean()
```

其中：

* `coords` 是网络输入坐标张量，维度为 [N, 2]（x, t）；
* `u` 为神经网络预测的扰动量；
* `residual` 为KdV方程残差，可直接用于损失函数；
* 损失函数采用残差的均方值。

> 🔍 **提示：**
>
> * 可将 `.pde()` 输出与数据误差、边界误差组合形成总损失；
> * 适用于孤立子传播、非线性波等多种物理场景。

---

## 贡献与许可证

**贡献者：**  
- Yooyu

**许可证：**  
本项目代码遵循 [GNU 通用公共许可证 第三版（GPLv3）](https://www.gnu.org/licenses/gpl-3.0.html)。您可以自由使用、修改和分发该代码，但需保留原始版权声明，并在分发时提供相同的许可证。

版权 © 硒钼科技（北京）。更多信息请访问 [Science42](https://www.science42.tech/#/index)。
