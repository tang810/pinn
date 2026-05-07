# 📘 Math_Physics_Formulations

本模块为 AI 科学计算任务提供统一且可扩展的数学与物理方程标准化实现。  
它是 PINNs4Science 框架中的核心物理方程库，旨在帮助研究者和工程师快速将常见的**偏微分方程（PDEs）、常微分方程（ODEs）**及其他公式集成到 AI 模型中（如 PINNs、FNO、图网络求解器等）。

---

## 📌 功能特点

- **统一接口**：所有方程均提供一致的 API，方便与各类 AI 模型集成。
- **维度灵活**：如 Navier-Stokes 方程支持 1D、2D、3D 等不同维度的版本，可自动选择。
- **自动微分支持**：完全兼容 PyTorch 的自动求导功能（autograd）。
- **可扩展设计**：方便用户扩展新方程，加入更多物理建模问题。

---

## 📦 已支持的方程

- [Navier-Stokes 方程](#-navier-stokes-方程navier_stokespy)
- [泊松方程](#-泊松方程poissonpy)
- [热传导方程](#-热传导方程heat_equationpy)
- [Navier-Cauchy 方程](#-naviercauchy-方程navier_cauchypy)
- [Korteweg–de Vries 方程](#-kortewegde-vries-kdv-方程kdvpy)
- [波动方程](#-波动方程wave_equationpy)
- [Vlasov 方程](#-vlasov-方程vlasov_equationpy)
- [Maxwell 方程](#-maxwell-方程maxwell_pdepy)
- [亥姆霍兹方程](#-亥姆霍兹方程helmholtz_pdepy)

### ✅ Navier-Stokes 方程（`navier_stokes.py`）

**方程形式**：

$$
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} = -\nabla p + \nu \nabla^2 \mathbf{u},
\quad \nabla \cdot \mathbf{u} = 0
$$

其中：
- $\mathbf{u}$ 为速度矢量场  
- $p$ 为流体压力  
- $\nu$ 为运动粘度系数  
- $\nabla \cdot \mathbf{u} = 0$ 表示不可压缩条件

**描述**：

不可压缩Navier-Stokes方程是描述流体运动中速度与压力关系的基本方程，  
广泛应用于流体力学、湍流模拟、空气动力学和海洋动力学等领域。  
方程包含动量守恒和连续性（不可压缩）两部分。

**功能模块**：

- 支持 1D / 2D / 3D 版本自动切换
- 计算动量方程残差
- 计算连续性方程残差（不可压缩条件）
- 基于PyTorch自动微分计算残差

**示例**

```python
from Math_Physics_Formulations.navier_stokes import NavierStokes

# 初始化2D Navier-Stokes方程
ns = NavierStokes(dimension=2, nu=0.01)

# 神经网络预测输出
coords = ...
u, v, p = ...

# 计算残差
residuals = ns.pde(coords, [u, v], p)

# 计算损失
loss = sum((r**2).mean() for r in residuals)
```
---
### ✅ 泊松方程（`poisson.py`）

**方程形式**：

$$
\nabla^2 u = f(\mathbf{x})
$$

其中：
- $u$ 为标量场（如电势、温度等）  
- $f$ 为右侧源项函数（如电荷密度、热源项等）

**描述**：

泊松方程是描述标量场分布的基本方程，广泛用于静电势、热传导、重力场等问题的模拟。  
可根据空间维度扩展为 1D、2D 或 3D 版本。

**功能模块**：

- 支持 1D / 2D  版本自动切换
- 计算泊松方程残差
- 支持右侧函数 \( f(x) \) 作为外部输入
- 基于PyTorch自动微分计算残差

**示例**

```python
from Math_Physics_Formulations.poisson import Poisson

# 初始化2D泊松方程
poisson = Poisson(dimension=2)

# 网络预测输出
coords = ...
u = ...
f = ...

# 计算残差
residual = poisson.pde(coords, u, f)

# 计算损失
loss = (residual**2).mean()
```

---



### ✅ 热传导方程（`heat_equation.py`）

**方程形式**：

$$
\frac{\partial u}{\partial t} = \alpha \nabla^2 u
$$

其中：
- $u$ 为温度（或浓度）标量场  
- $\alpha$ 为热扩散系数  
- $\nabla^2 u$ 表示空间中的扩散项（Laplace项）

**描述**：

热传导方程是描述热量在空间中传递过程的基本偏微分方程，  
可用于热传导、扩散问题等科学与工程模拟，支持稳态与非稳态建模。

**功能模块**：

- 支持 1D / 2D / 3D 版本自动切换
- 计算热传导方程残差（时间导数与拉普拉斯项）
- 支持热扩散率 $\alpha$ 自定义
- 基于PyTorch自动微分计算残差

**示例**

```python
from Math_Physics_Formulations.heat_equation import HeatEquation

# 初始化2D热传导方程
heat = HeatEquation(dimension=2, alpha=0.01)

# 网络预测输出
coords = ...
u = ...

# 计算残差
residual = heat.pde(coords, u)

# 计算损失
loss = (residual**2).mean()
```

### ✅ Navier–Cauchy 方程（`navier_cauchy.py`）

**方程形式**：


$$
\mu \nabla^2 \mathbf{u} + (\lambda + \mu) \nabla (\nabla \cdot \mathbf{u}) = \mathbf{f}
$$

其中：
- $\mathbf{u}$ 为位移向量
- $\mu$ 为剪切模量（Shear modulus）
- $\lambda$ 为第一拉梅常数
- $\mathbf{f}$ 为体积力（如重力）

**描述**：

Navier–Cauchy 方程描述线性弹性体中的应力平衡条件，是连续介质力学和弹性力学的基本方程。  
广泛应用于材料模拟、应力分析、结构设计等场景。

**功能模块**：

- 支持 2D / 3D 自动切换
- 支持拉梅常数参数化输入$(λ, μ)$
- 支持体积力项 $\mathbf{f}$ 输入
- 输出每个分量的残差，可用于PINNs loss

**示例**

```python
from Math_Physics_Formulations.navier_cauchy import NavierCauchy

# 初始化2D Navier–Cauchy 方程
nc = NavierCauchy(dimension=2, lam=1.0, mu=0.5)

# coords: [x, y]
# u, v: 神经网络输出的位移
# f: 外力（如重力），shape = [N, 2]
residuals = nc.pde(coords, [u, v], f)

loss = sum((r**2).mean() for r in residuals)
```

### ✅ Korteweg–de Vries (KdV) 方程（`kdv.py`）

**方程形式**：

$$
\frac{\partial u}{\partial t} + 6u \frac{\partial u}{\partial x} + \frac{\partial^3 u}{\partial x^3} = 0
$$

其中：
- $u$ 为波的振幅（标量）  
- $x, t$ 为空间与时间变量

**描述**：

KdV方程是一类非线性、三阶的抛物型方程，用于描述浅水表面孤立波（soliton）或非线性色散波的传播过程。  
该方程具有精确解（孤立子）且适合用PINNs进行非线性动力学模拟。

**功能模块**：

- 支持1D空间形式
- 支持非线性项和三阶导项计算
- 可用于PINNs、孤立波建模、非线性PDE求解

**示例**

```python
from Math_Physics_Formulations.kdv import KdV

kdv = KdV()

# coords: [x, t]；u: 网络输出
residual = kdv.pde(coords, u)

loss = (residual**2).mean()
```

### ✅ 波动方程（`wave_equation.py`）

**方程形式**：

$$
\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u
$$

其中：
- $u$ 为波的扰动量（如位移、电场强度等）  
- $c$ 为波速  
- $\nabla^2 u$ 为Laplace项（空间传播）

**描述**：

波动方程是描述机械波、电磁波、声波等传播行为的基本PDE，  
可建模弦振动、电磁波传播、膜面震动等现象。  
常用于信号模拟、结构响应、地震波建模等场景。

**功能模块**：

- 支持1D / 2D / 3D自动切换
- 支持自定义波速 $$c$$
- 输出二阶时间残差和Laplace项差值残差

**示例**

```python
from Math_Physics_Formulations.wave_equation import WaveEquation

wave = WaveEquation(dimension=1, c=1.0)

# coords: [x, t]；u: 网络输出
residual = wave.pde(coords, u)

loss = (residual**2).mean()
```

### ✅ Vlasov 方程（`vlasov_equation.py`）

**方程形式**：

$$
\frac{\partial f}{\partial t}+\lambda_1 \cdot v \cdot \frac{\partial f}{\partial x}+\lambda_2 \cdot E \cdot \frac{\partial f}{\partial v}=0
$$

其中：
- $f$ 为分布函数（描述粒子在位置-速度相空间中的分布）
- $v$ 为速度
- $E$ 为电场强度
- $\lambda_1, \lambda_2$ 为参数（理论值分别为1.0和-1.0）

**描述**：

Vlasov方程是描述无碰撞等离子体中带电粒子分布演化的基本方程，由Boltzmann方程在忽略碰撞项后简化得到。
该方程广泛应用于等离子体物理、加速器物理和天体物理学，可用于模拟双流不稳定性、朗缪尔波等现象。

**功能模块**：

- 支持1D空间+1D速度形式
- 支持参数自动学习（如$\lambda_1$和$\lambda_2$）
- 通过PINN方法求解分布函数$f$和电场$E$的演化
- 基于PyTorch自动微分计算残差

**示例**

```python
from Math_Physics_Formulations.vlasov_equation import PhysicsInformedNN

# 初始化PINN模型
layers = [3, 50, 50, 50, 50, 50, 50, 50, 50, 2]
model = PhysicsInformedNN(X_data, E_data, f_data, layers)

# 训练模型
model.train(20000)  # 训练20000步

# 预测分布函数和电场
f_pred, E_pred = model.predict(T_star, X_star, V_star)

# 获取学习到的参数
lambda_1 = model.lambda_1.item()
lambda_2 = model.lambda_2.item()
```

### ✅ Maxwell 方程（`maxwell_pde.py`）

**方程形式**（频域）：

$$
\nabla \times (\nabla \times \mathbf{E}) - \mu \epsilon \omega^2 \mathbf{E} = 0
$$

其中：
- $\mathbf{E}$ 为电场矢量
- $\mu$ 为磁导率
- $\epsilon$ 为介电常数
- $\omega$ 为角频率

**描述**：

Maxwell方程组是描述电磁波传播的基本方程，频域形式在电磁散射、光波导和微波工程等领域广泛应用。  
该方程通过双旋度（curl of curl）表达了电场的传播特性，可用于模拟电磁波在均匀或非均匀介质中的传播。

**功能模块**：

- 计算电场的旋度和双旋度
- 计算频域Maxwell方程残差（支持PINN）
- 基于PyTorch自动微分计算残差

**示例**

```python
from Math_Physics_Formulations.maxwell_pde import MaxwellPDE

# 初始化Maxwell方程
maxwell = MaxwellPDE(mu=1.0, epsilon=1.0, omega=2 * 3.1416)

# 神经网络预测输出
coords = ...
E = ...  # shape: [N, 3]

# 计算残差
residuals = maxwell.compute_pde_residuals(E, coords)

# 计算损失
loss = (residuals**2).mean()
```


### ✅ 亥姆霍兹方程（`helmholtz_pde.py`）

**方程形式**：

$$
\nabla^2 u + k^2 u = 0
$$

其中：
- $u$ 为场变量
- $k$ 为波数

**描述**：

亥姆霍兹方程用于描述定常波场问题，如声波、光波、电磁波的传播特性，  
广泛应用于声学、电磁学和振动问题等工程领域。

**功能模块**：

- 支持 1D / 2D / 3D 版本自动切换
- 计算Laplacian项和波数项
- 基于PyTorch自动微分计算残差

**示例**

```python
from Math_Physics_Formulations.helmholtz_pde import HelmholtzPDE

# 初始化Helmholtz方程
helmholtz = HelmholtzPDE(k=3.14)

# 神经网络预测输出
coords = ...
u = ...

# 计算残差
residuals = helmholtz.compute_pde_residuals(u, coords)

# 计算损失
loss = (residuals**2).mean()
```
