<!-- text: "参数化 PINN：二维弹性力学基函数预计算与在线求解", area: "本项目旨在解决二维弹性力学问题中的参数化挑战，通过创新的**参数化物理信息神经网络（GPT-PINN）**方法实现高效求解。针对传统PINN每次参数变化需重新训练的巨大计算开销，GPT-PINN将求解分解为离线预计算基函数和在线优化元网络两个阶段，以快速逼近新参数下的解。项目解决具有解析解的二维线性弹性力学问题，预测域内任意点的位移场 $(u_x, u_y)$ 和应力张量分量 $(\sigma_{xx}, \sigma_{yy}, \sigma_{xy})$。", tags: [ "Parametric PINN", "二维弹性力学", "GPT-PINN", "基函数方法", "应力位移预测" ] -->

# 一维 Burgers 方程基函数预计算与在线求解

## 项目概述

本项目旨在解决经典**一维非线性 Burgers 方程**的参数化挑战。Burgers 方程是一个在流体力学、非线性声学等领域具有广泛应用的偏微分方程，其特点是包含对流项和扩散项，并且其解行为受**粘性系数 $\nu$** 的影响而呈现出从光滑到具有激波（冲击波）的复杂变化。

传统的物理信息神经网络 (PINN) 在处理此类问题时，通常针对单个特定的粘性系数 $\nu$ 进行训练。然而，当我们需要在不同的 $\nu$ 值下求解方程时，每次参数变化都需要重新训练一个完整的 PINN 模型，这会带来巨大的计算负担。

为了高效地处理这一参数化挑战，本项目采用并实现了**参数化物理信息神经网络 (Parametric Physics-Informed Neural Network, GPT-PINN)** 方法。GPT-PINN 将复杂的求解过程解耦为两个主要阶段：

1.  **离线预计算阶段 (Offline Precomputation):** 在此阶段，我们选择参数空间中离散的 $\nu$ 值，并为每个值独立训练一个标准的 PINN 模型。这些训练好的 PINN 的输出（包括解本身及其导数）在整个计算域内的特定采样点上被提取，形成一套**基函数 (Basis Functions)**。
2.  **在线优化阶段 (Online Optimization):** 对于任何新的、未在离线阶段训练过的粘性系数 $\nu_{\text{new}}$，GPT-PINN 无需从头训练一个大型神经网络。相反，它通过一个轻量级的**元网络 (Meta-network)**（本项目中是一个简单的线性组合），学习如何高效地组合（加权求和）预先计算的基函数。这个在线优化过程（仅需学习基函数的系数）通常比从零开始训练一个完整 PINN 快几个数量级。

通过这种两阶段方法，GPT-PINN 结合了 PINN 强大的物理建模能力与参数化建模的效率，为快速求解涉及可变参数的物理系统提供了有潜力的新途径。本项目模型的目标是预测域内任意点的一维解 $u(x,t)$。


| ![img](https://www.science42.tech/cases/caseMarkdown/一维Burgers方程基函数预计算与在线求解/viz/time_comparison.png)                |
| :---: |
| **Fig. 1** GPT-PINN 与标准 PINN 的总计算时间对比 |

该图对比了 GPT-PINN (在线阶段) 和标准 PINN (从头训练) 在不同测试案例下的累计计算时间，突出 GPT-PINN 在多参数求解中的效率优势。



## 核心功能与方法

*   **参数化 PINN (GPT-PINN):** 有效处理 Burgers 方程中随粘性系数 $\nu$ 变化的连续参数。
*   **两阶段训练框架:**
    *   **离线阶段:** 训练一组独立的标准 PINN，它们的输出作为基函数。
    *   **在线阶段:** 优化基函数的线性组合系数，以快速适应新的参数值。
*   **自适应基函数生成:** 离线阶段采用启发式方法选择下一个 $\nu$ 值来训练新的基函数，通常选择在当前已生成基函数集合下近似误差最大的 $\nu$ 值。
*   **物理信息损失函数:** 将 Burgers 方程的控制定律和初始/边界条件直接嵌入到神经网络的损失函数中：
    *   **PDE 残差 (PDE Residual):** 确保预测解满足 Burgers 方程。
    *   **初始条件 (Initial Condition, IC):** 强制解在初始时刻满足给定分布。
    *   **边界条件 (Boundary Conditions, BC):** 强制解在空间边界满足给定条件。
*   **自动微分:** 利用 PyTorch 的自动微分功能精确计算解对空间和时间的各阶导数，从而计算物理方程的残差。
*   **可视化与评估:** 提供详细的解决方案场可视化（$u(x,t)$ 场图），并计算相对 $L^2$ 误差以评估模型性能。同时，展示元网络学习到的基函数组合系数随参数 $\nu$ 的变化，以及不同方法的计算时间对比。

## 数学背景

本项目解决的是在域 $x \in [-1, 1]$ 和时间 $t \in [0, 1]$ 内定义的**一维非线性 Burgers 方程**。

### 1. 物理常数

本模型中主要参数是**粘性系数 $\nu$**，它决定了方程中扩散项的强度。

### 2. 控制方程

Burgers 方程的数学形式为：
$$
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} - \nu \frac{\partial^2 u}{\partial x^2} = 0
$$
其中：

*   $u(x, t)$ 是在位置 $x$ 和时间 $t$ 处的物理量（例如流体速度）。
*   $\frac{\partial u}{\partial t}$ 是时间导数。
*   $u \frac{\partial u}{\partial x}$ 是非线性对流项。
*   $\nu \frac{\partial^2 u}{\partial x^2}$ 是扩散项。

### 3. 初始条件 (IC)

在初始时刻 $t=0$：
$$
u(x, 0) = -\sin(\pi x)
$$

### 4. 边界条件 (BC)

在空间域的两个边界 $x=-1$ 和 $x=1$ 上，施加 Dirichlet 边界条件：
$$
u(-1, t) = 0
$$

$$
u(1, t) = 0
$$

### 5. 损失函数

PINN 的训练通过最小化一个综合损失函数来实现，该损失函数包含以下几部分：

*   **PDE 残差损失 ($L_R$):** 衡量预测解在内部域搭配点上对 Burgers 方程的满足程度。
    $$
    L_R = \frac{1}{N_{resid}} \sum_{(x_c, t_c) \in \Omega_{resid}} \left( \left(\frac{\partial u}{\partial t}\right)_{NN} + \left(u \frac{\partial u}{\partial x}\right)_{NN} - \nu \left(\frac{\partial^2 u}{\partial x^2}\right)_{NN} \right)^2
    $$
    其中，$(\cdot)_{NN}$ 表示由神经网络预测的量，$N_{resid}$ 是内部采样点数量。由于方程右侧为零，目标值为 0。

*   **初始条件损失 ($L_{IC}$):** 衡量预测解在初始时刻 $t=0$ 上与给定初始分布的偏差。
    $$
    L_{IC} = \frac{1}{N_{IC}} \sum_{(x_0, 0) \in \partial\Omega_{IC}} \left( u_{NN}(x_0, 0) - u_{\text{target}}(x_0, 0) \right)^2
    $$
    其中 $u_{\text{target}}(x_0, 0) = -\sin(\pi x_0)$。

*   **边界条件损失 ($L_{BC}$):** 衡量预测解在空间边界 $x=-1$ 和 $x=1$ 上与给定值的偏差。
    $$
    L_{BC} = \frac{1}{N_{BC}} \sum_{(x_b, t_b) \in \partial\Omega_{BC}} \left( u_{NN}(x_b, t_b) - u_{\text{target}}(x_b, t_b) \right)^2
    $$
    其中 $u_{\text{target}}(x_b, t_b) = 0$。

总损失函数为这些损失项的简单相加：
$$
\mathcal{L} = L_R + L_{IC} + L_{BC}
$$
通过最小化此总损失，神经网络被驱动学习满足所有物理约束的解。

## 环境与依赖

- Python ≥ 3.8
- PyTorch ≥ 1.10 (代码中默认使用 `torch.float64`)
- NumPy
- Matplotlib
- pyDOE2 (用于拉丁超立方抽样)
- SciencePlots (可选，用于美化图表)

## 快速开始

### 安装依赖

建议使用虚拟环境进行安装：

```bash
conda create -n gpt_pinn_burgers python=3.9
conda activate gpt_pinn_burgers
pip install torch torchvision torchaudio
pip install numpy matplotlib pyDOE2 scienceplots
```

### 运行脚本

本项目主要通过运行 `B_main.py` 来执行完整的训练、测试和数据保存流程。`B_plotting.py` 用于后续的数据可视化。

**训练与测试 (一体化运行)**

直接运行 `B_main.py` 将执行整个训练管道：

- **离线预计算:** 自适应地训练 `number_of_neurons` (默认为 9) 个标准 PINN 作为基函数。
- **在线优化:** 对 `b_train` (默认为 129 个参数点) 中的每个 $\nu$ 值，以及 `test_cases` (默认为 25) 个随机测试 $\nu$ 值，优化 GPT-PINN 的系数。
- **性能对比:** 比较 GPT-PINN 和标准 PINN 在测试集上的时间及损失表现。
- **数据保存:** 所有结果数据将保存到 `./b_data/` 目录。

```bash
python B_main.py
```


> 🔍 **提示：**
>
> * 默认生成基函数数量为 `number_of_neurons = 9`，可根据问题复杂度调整。
> * 离线 PINN 模型训练轮数默认为 `epochs_pinn = 60000`，学习率为 `lr_pinn = 0.005`，可根据收敛情况适当修改。
> * 在线 GPT-PINN 模型训练轮数为 `epochs_gpt_train = 2000`，学习率为 `lr_gpt = 0.02`，可按需调整。
> * 测试案例数量默认设为 `test_cases = 25`，用于评估泛化能力，数量可灵活增减。


**绘制结果**

在 `B_main.py` 运行完成后，可以通过 `B_plotting.py` 脚本加载 `b_data/` 目录中的结果，并生成一系列图表。

```bash
python B_plotting.py
```

这将生成以下分析图表：

- **Neurons:** 基函数在 $\nu$ 参数空间中的分布。
- **Largest loss plot:** 离线阶段自适应选择策略的"最大损失"演变。
- **Generation times:** 每个基函数生成所需的训练时间。
- **Total time:** GPT-PINN (在线阶段) 和标准 PINN (从头训练) 在测试集上的累计计算时间对比。
- **Absolute Errors:** 针对选定测试案例，$u(x,t)$ 的预测结果（GPT-PINN vs. PINN）及其绝对误差的等高线图。



## 模型构建

#### 代码单元 1：标准 PINN 模型定义

这个代码块定义了 `NN` 类，它是一个用于离线阶段训练的**标准物理信息神经网络（PINN）模型**。它是一个多层感知机（MLP），用于预测 Burgers 方程的解 $u(x,t)$。

```python
class NN(nn.Module):
    def __init__(self, layers):
        super(NN, self).__init__()
        self.layers = layers # 定义网络各层的神经元数量
        self.activation = nn.Tanh() # 使用Tanh激活函数
        self.linear_layers = nn.ModuleList() # 存储线性层
        # 逐层构建神经网络
        for i in range(len(self.layers) - 1):
            self.linear_layers.append(nn.Linear(self.layers[i], self.layers[i+1]))
            # 使用Xavier正态分布初始化权重
            nn.init.xavier_normal_(self.linear_layers[-1].weight)
            # 偏置初始化为零
            nn.init.zeros_(self.linear_layers[-1].bias)

    def forward(self, x, t):
        # 将空间(x)和时间(t)输入拼接起来
        inputs = torch.cat((x, t), dim=1) 
        # 逐层进行前向传播，应用线性变换和激活函数
        for i in range(len(self.linear_layers) - 1):
            inputs = self.linear_layers[i](inputs)
            inputs = self.activation(inputs)
        # 最后一层输出解 u
        u = self.linear_layers[-1](inputs)
        return u
```

> 💡 **解读：**
>
> - **功能**：构建一个多层感知机（MLP）作为神经网络模型，用于从输入 $(x, t)$ 预测 Burgers 方程的解 $u(x,t)$。
> - **核心**：通过 `nn.Linear` 和 `nn.Tanh` 层实现非线性映射，权重和偏置进行初始化。
> - **用途**：主要用于离线阶段训练每个基函数对应的标准PINN。

> 🔍 **提示：**
>
> - `layers` 列表定义了网络的宽度和深度，可根据问题复杂度和数据量进行调整。
> - `nn.Tanh` 是一种常见的激活函数，也可以尝试 `nn.SiLU` (Swish) 或 `nn.GELU` 等以优化性能。
> - 权重初始化（如 `xavier_normal_`）对于网络训练的稳定性和收敛速度至关重要。

#### 代码单元 2：GPT 元网络模型

这个代码块定义了 `GPT` 类，它是**参数化 PINN (GPT-PINN)** 的核心——**元网络**。在在线优化阶段，它通过线性组合预计算好的基函数来快速逼近新参数 $\nu$ 值下的解。

```python
class GPT(nn.Module):
    def __init__(self, basis_functions_tensor):
        super(GPT, self).__init__()
        # basis_functions_tensor 形状为 (N_data_points, N_basis_functions)
        # 包含所有预计算的基函数在特定数据点上的值
        self.basis_functions = basis_functions_tensor 
        # 定义可学习的系数 c，形状为 (N_basis_functions, 1)
        # 这些系数通过在线优化阶段的学习来组合基函数
        self.c = nn.Parameter(torch.ones(self.basis_functions.shape[1], 1, dtype=torch.float64))

    def forward(self):
        # 通过矩阵乘法实现基函数的线性组合：u = BasisFunctions @ c
        u = self.basis_functions @ self.c
        return u
```
> 💡 **解读：**
>
> - **功能**：实现一个轻量级的元网络，用于通过线性组合（加权求和）预计算的基函数来得到最终解。
> - **核心**：`nn.Parameter(self.c)` 是唯一的可学习参数，代表每个基函数的权重。
> - **用途**：在在线优化阶段，通过调整这些系数，快速适应新的参数值 $\nu$。

> 🔍 **提示：**
>
> - 这种设计使得在线阶段的训练速度极快，因为它只优化少量的组合系数，而不是一个完整的神经网络。
> - `basis_functions_tensor` 必须是预先计算好的，通常在离线阶段生成并在所有相关数据点上进行评估。

#### 代码单元 3：损失函数计算（PDE残差）

这个代码块展示了 `NN` 类（标准PINN模型）中用于计算**Burgers方程PDE残差损失**的方法 `lossR`。它利用自动微分计算预测解的各阶导数，并构建物理方程的残差项。

```python
def lossR(self, xt_r_tensor, nu_val):
    # 从输入的搭配点 xt_r_tensor 中分离出空间 x 和时间 t
    x_r = xt_r_tensor[:, 0:1]
    t_r = xt_r_tensor[:, 1:2]
    
    # 通过神经网络预测 u(x,t) 的值
    u_pred = self.forward(x_r, t_r)
    # 调用外部函数 derivatives (例如在 b_precomp.py 中定义)
    # 计算 u 对时间 t 的一阶导数 (u_t)
    # u 对空间 x 的一阶导数 (u_x)
    # u 对空间 x 的二阶导数 (u_xx)
    u_t, u_x, u_xx = derivatives(u_pred, t_r, x_r)
    
    # 根据 Burgers 方程 (u_t + u*u_x - nu*u_xx = 0) 计算 PDE 残差
    f_pred = u_t + u_pred * u_x - nu_val * u_xx
    # 残差的均方误差作为损失
    loss = torch.mean(f_pred**2)
    return loss
```
> 💡 **解读：**
>
> - **功能**：计算 Burgers 方程的 PDE 残差损失。
> - **核心**：利用 `torch.autograd.grad` (通过 `derivatives` 函数封装) 计算预测解的各阶导数，然后将其代入 Burgers 方程，目标是使方程左侧趋近于零。
> - **用途**：作为 PINN 训练中的物理约束项之一，驱动神经网络学习满足PDE的解。

> 🔍 **提示：**
>
> - `xt_r_tensor` 是 PDE 搭配点（collocation points），在这些点上强制满足 PDE。
> - `nu_val` 是当前的粘性系数，它直接影响 PDE 的扩散项。
> - PDE 残差通常使用均方误差（MSE）作为损失，确保预测解在搭配点上尽可能满足物理定律。

## 模型训练

#### 代码单元 4：数据生成

这个代码块展示了 `gen_data` 函数的核心逻辑，它负责为 Burgers 方程生成不同类型的训练数据点：包括用于计算PDE残差的**内部搭配点** (`xt_r`)，以及用于施加初始条件和边界条件的**边界点** (`xt_icbc` 和 `u_icbc`)。

```python
def gen_data(N_r, N_icbc):
    # 定义空间 x 和时间 t 的域范围
    lb = np.array([-1.0, 0.0]) # 下界 [x_min, t_min]
    ub = np.array([1.0, 1.0])   # 上界 [x_max, t_max]
    
    # 使用拉丁超立方抽样 (lhs) 在整个域内生成 N_r 个 PDE 残差点
    data_r = lb + (ub - lb) * lhs(2, N_r)
    xt_r = data_r

    # 生成初始条件点 (t=0 时的 x 值)
    x_ic = np.linspace(-1, 1, N_icbc).reshape(-1, 1)
    t_ic = np.zeros_like(x_ic)
    # 对应的初始条件目标值 u(x,0) = -sin(pi*x)
    u_ic = -np.sin(np.pi * x_ic)
    
    # 返回 PDE 残差点、合并后的 IC/BC 点及其目标值
    # 注意：这里提供的代码片段只展示了部分IC/BC数据的生成，完整逻辑应包括边界条件点的合并
    return xt_r.astype(np.float64), xt_icbc.astype(np.float64), u_icbc.astype(np.float64)
```
> 💡 **解读：**
>
> - **功能**：生成用于 PINN 训练所需的各类采样点。
> - **核心**：
>   - **PDE 搭配点**：使用拉丁超立方抽样 (`lhs`) 在整个时空域内均匀生成，用于计算 PDE 残差。
>   - **初始/边界条件点**：在初始时刻和空间边界上生成，并根据解析解或给定条件设置目标值。
> - **用途**：为 PINN 提供训练数据，以便模型能同时学习满足 PDE 和各项条件。

> 🔍 **提示：**
>
> - 采样点的数量 (`N_r`, `N_icbc`) 对 PINN 的训练效果和计算成本有显著影响，需要权衡。
> - 拉丁超立方抽样有助于确保采样点在整个域内分布均匀，避免采样偏差。
> - 对于复杂问题，边界条件点的密度和分布可能需要特别优化。

#### 代码单元 5：离线基函数生成

这个代码块展示了 `offline_generation` 函数的核心逻辑，它负责执行**离线预计算阶段**。它会迭代地选择不同的粘性系数 $\nu$，为每个 $\nu$ 值训练一个标准 PINN 模型，并从中提取出作为基函数的数据。

```python
def offline_generation(nu_min, nu_max, number_of_neurons, epochs_pinn, lr_pinn):
    basis_functions = []     # 存储生成的基函数
    neurons_selected = []    # 存储每个基函数对应的 nu 值
    generation_times = []    # 存储每个基函数的训练时间
    
    # 循环指定次数，生成所需数量的基函数
    for i in range(number_of_neurons):
        # 启发式选择下一个要训练的 nu 值（这里用 select_next_nu_adaptive 占位）
        current_nu = select_next_nu_adaptive(existing_basis, nu_min, nu_max)
        
        start_time = time.time()
        # 初始化一个标准 PINN 模型
        model_nn = NN(layers=[2, 20, 20, 20, 20, 1])
        # 训练当前 nu 值对应的 PINN 模型
        trained_model = pinn_train(model_nn, current_nu, epochs_pinn, lr_pinn)
        end_time = time.time()
        
        # 记录训练时间和选定的 nu 值
        generation_times.append((end_time - start_time) / 60.0) # 转换为分钟
        neurons_selected.append(current_nu)
        
        # 从训练好的 PINN 模型中提取基函数（例如，在指定数据点上的预测解 u 及其导数）
        # inputs 函数（b_precomp.py中）负责将模型输出处理成基函数格式
        current_basis = inputs(trained_model, xt_r_tensor, xt_icbc_tensor, current_nu, nu_abs_max)
        # 存储基函数，并移除梯度信息以节省内存
        basis_functions.append(current_basis.detach())
        
    # 将所有基函数堆叠成一个大的张量，供 GPT 模型使用
    return torch.cat(basis_functions, dim=1), neurons_selected, generation_times
```
> 💡 **解读：**
>
> - **功能**：管理离线预计算过程，逐步训练多个标准PINN以生成基函数。
> - **核心**：
>   - **迭代训练**：循环 `number_of_neurons` 次，每次训练一个新的 `NN` 模型。
>   - **自适应选择**：`select_next_nu_adaptive` (在此代码片段中为占位符) 是选择下一个 $\nu$ 值的关键策略，通常基于当前基函数集的近似误差。
>   - **基函数提取**：`inputs` 函数负责将训练好的PINN的输出（解和导数）转化为结构化的基函数形式。
> - **用途**：生成 `GPT` 模型在线学习所需的基函数集合。

> 🔍 **提示：**
>
> - 自适应选择策略（如选择当前误差最大的点）比均匀采样更高效，能用更少的基函数覆盖整个参数空间。
> - `detach()` 操作非常重要，它确保在GPT模型的在线训练阶段，不会反向传播梯度到离线训练好的PINN模型中。
> - `number_of_neurons` 的选择是精度和离线计算成本之间的权衡。

#### 代码单元 6：在线优化

这个代码块展示了 `gpt_test` 函数的核心逻辑，它负责执行**在线优化阶段**。对于给定的新粘性系数 $\nu$，它会初始化一个 `GPT` 元网络模型，并通过优化其系数 `c` 来快速拟合预计算的基函数，从而得到该 $\nu$ 值下的解。

```python
def gpt_test(basis_functions_tensor, nu_val, epochs_gpt_train, lr_gpt):
    start_time = time.time()
    # 初始化 GPT 模型，输入是预计算的基函数张量
    gpt_model = GPT(basis_functions_tensor)
    # 使用 Adam 优化器优化 GPT 模型的参数 c (基函数组合系数)
    optimizer = torch.optim.Adam(gpt_model.parameters(), lr=lr_gpt)

    # 在线优化循环
    for epoch in range(epochs_gpt_train):
        optimizer.zero_grad() # 清零梯度
        
        # 计算 GPT 模型的损失（包括 PDE 残差和 IC/BC 损失）
        # 注意：这里的 lossR 和 lossICBC 在 GPT 模型中可能是简化或占位符，
        # 实际实现中需要能计算基函数线性组合的导数并构建物理损失
        loss_r = gpt_model.lossR(xt_r_test_tensor, nu_val) # Placeholder
        loss_icbc = gpt_model.lossICBC(xt_icbc_test_tensor, u_icbc_test_target) # Placeholder
        total_loss = loss_r + loss_icbc
        
        total_loss.backward() # 反向传播计算梯度
        optimizer.step()      # 更新模型参数 c
        
    end_time = time.time()
    # 获取最终预测的解（通过基函数和优化后的系数 c 组合得到）
    final_u_pred_gpt = gpt_model().detach().cpu().numpy()
    
    # 返回预测解和总耗时（小时）
    return final_u_pred_gpt, (end_time - start_time) / 3600.0
```
> 💡 **解读：**
>
> - **功能**：对给定的新参数 $\nu$，快速优化 `GPT` 模型的基函数组合系数，以在线地预测解。
> - **核心**：
>   - **轻量级优化**：仅优化 `gpt_model.c`，而非整个神经网络，因此训练速度远快于从头训练PINN。
>   - **损失计算**：`lossR` 和 `lossICBC` 会在 `GPT` 模型内部根据基函数的特性和当前的 `nu_val` 计算。
> - **用途**：实现 GPT-PINN 的高效在线预测能力。

> 🔍 **提示：**
>
> - 在线优化 (`epochs_gpt_train`) 的迭代次数通常远少于离线训练 (`epochs_pinn`)。
> - GPT模型的损失函数 `lossR` 和 `lossICBC` 的具体实现至关重要，它需要能够利用预计算的基函数信息来构建物理约束。

## 可视化

#### 代码单元 7：数据加载与基础设置

这个代码块展示了 `b_plotting.py` 脚本的开头部分，它负责从预设路径加载所有用于绘图的数据文件，并进行 Matplotlib 的基本样式设置。

```python
path = "./b_data/" # 定义数据文件存放的路径

# 加载各种数据文件，这些文件在 b_main.py 运行后生成并保存
generation_time = np.loadtxt(path+"generation_time.dat") # 每个基函数生成的时间
max_losses = np.loadtxt(path+"max_losses.dat")         # 离线阶段最大损失的演变
neurons = np.loadtxt(path+"neurons.dat")               # 生成基函数对应的 nu 值
total_time = np.loadtxt(path+"total_time.dat")         # 总时间对比数据（GPT vs PINN）

# 设置 Matplotlib 的绘图样式，优先使用 'science' 或 'seaborn-v0_8-notebook' 风格
if 'science' in plt.style.available:
    plt.style.use(['science', 'notebook'])
elif 'seaborn-v0_8-notebook' in plt.style.available:
    plt.style.use(['seaborn-v0_8-notebook'])
```
> 💡 **解读：**
>
> - **功能**：加载分析所需的数值数据，并配置绘图环境。
> - **核心**：
>   - **数据加载**：使用 `np.loadtxt` 从指定路径加载`.dat`格式的预保存数据。
>   - **绘图样式**：尝试应用 'science' 或 'seaborn-v0_8-notebook' 等预定义Matplotlib样式，以获得更专业的图表外观。
> - **用途**：为后续的各种数据可视化做好准备。

> 🔍 **提示：**
>
> - 确保 `path` 变量指向正确的 `b_data/` 目录，否则会导致文件加载错误。
> - 如果没有安装 `scienceplots` 或 `seaborn`，样式设置可能会回退到默认，不影响功能。

#### 代码单元 8：神经元分布图

这个代码块用于绘制**基函数在粘性系数 $\nu$ 参数空间中的分布图**。它展示了离线阶段选择哪些 $\nu$ 值来训练基函数。

```python
fig, ax = plt.subplots(figsize=(10, 2)) # 创建一个指定大小的图形和坐标轴
# 在 x 轴上绘制 neurons（nu 值），y 轴为 0，用红色圆点表示
ax.scatter(neurons, np.zeros_like(neurons), s=100, c='red', alpha=0.7, zorder=2)
# 设置 x 轴的显示范围
ax.set_xlim(nu_min-0.05, nu_max+0.05)
# 设置 x 轴标签
ax.set_xlabel(r'Viscosity coefficient $\nu$')
# 设置图表标题
ax.set_title('Basis Function Distribution in Parameter Space')
plt.show() # 显示图表
```
> 💡 **解读：**
>
> - **功能**：可视化离线阶段选取的参数 $\nu$ 值（即基函数对应的 $\nu$ 值）在参数空间中的分布情况。
> - **核心**：使用 `ax.scatter` 绘制散点图，横坐标为 $\nu$ 值，纵坐标为0，以二维形式表示一维参数的分布。
> - **用途**：帮助理解基函数是如何在参数空间中被选取的，尤其是在采用自适应采样策略时。

> 🔍 **提示：**
>
> - 图表的标题和轴标签清晰地指明了图表的含义。
> - 如果 `neurons` 是通过自适应选择（而非均匀采样）获得的，图中的点分布可能不均匀，这正是为了优化基函数覆盖效果。

#### 代码单元 9：性能对比图

这个代码块用于绘制**GPT-PINN 与标准 PINN 在测试集上的累计计算时间对比图**。它直观地展示了 GPT-PINN 在处理多参数问题时的效率优势。

```python
fig, ax = plt.subplots() # 创建一个图形和坐标轴
# 绘制 GPT-PINN 的累计时间曲线 (蓝色实线)
ax.plot(b_test, np.cumsum(gpt_test_time), 'b-', label='GPT-PINN (Online)', linewidth=2)
# 绘制标准 PINN 的累计时间曲线 (红色虚线)
ax.plot(b_test, np.cumsum(pinn_test_time), 'r--', label='Standard PINN', linewidth=2)
ax.set_xlabel(r'Test parameter $\nu$') # 设置 x 轴标签
ax.set_ylabel('Cumulative time (hours)') # 设置 y 轴标签
ax.legend() # 显示图例
plt.show() # 显示图表
```
> 💡 **解读：**
>
> - **功能**：直观对比 GPT-PINN (在线阶段) 和标准 PINN (从头训练) 在处理多个测试参数时的累计计算时间。
> - **核心**：
>   - `np.cumsum` 计算累计时间，表示随着测试案例数量的增加，总耗时的变化趋势。
>   - `ax.plot` 绘制两条曲线，分别代表两种方法的累计时间。
> - **用途**：突出 GPT-PINN 在面对大量参数变化时，因其在线优化特性而带来的显著计算效率优势。

> 🔍 **提示：**
>
> - 通常情况下，GPT-PINN 的累计时间曲线会远低于标准 PINN 的曲线，尤其是在测试案例数量较大时。
> - x轴上的 `Test parameter nu` 应该是有序的，以便正确反映累计趋势。

#### 代码单元 10：解场等高线图

这个代码块用于循环绘制指定测试案例的**GPT-PINN 预测解 $u(x,t)$ 的等高线图**。它将一维的预测结果重塑为二维网格数据，并通过颜色等高线展示解场随空间和时间的变化。

```python
for i, param in enumerate(test_cases): # 遍历每个测试案例
    nu = param # 当前测试的粘性系数 nu
    u_gpt = gpt_test_soln[:,i][:,None] # 提取 GPT-PINN 预测的解
    u_pinn = pinn_test_soln[:,i][:,None] # 提取标准 PINN 预测的解 (此变量在此片段中未使用，但在完整脚本中可能用于对比)
    
    # 计算网格形状，将一维解数据重塑为二维 (例如 100x100)
    shape = (int(np.sqrt(u_gpt.shape[0])), int(np.sqrt(u_gpt.shape[0])))
    x_grid = xt_test[:,0].reshape(shape).transpose(1,0) # 重塑空间坐标 x 到网格
    t_grid = xt_test[:,1].reshape(shape).transpose(1,0) # 重塑时间坐标 t 到网格
    u_gpt_grid = u_gpt.reshape(shape).transpose(1,0) # 重塑 GPT 预测的解 u 到网格
    
    fig, ax = plt.subplots() # 创建一个图形和坐标轴
    # 绘制等高线图，levels 指定等高线数量，cmap 指定颜色映射
    contour = ax.contourf(x_grid, t_grid, u_gpt_grid, levels=50, cmap='viridis')
    ax.set_xlabel('Space x') # 设置 x 轴标签
    ax.set_ylabel('Time t')   # 设置 y 轴标签
    ax.set_title(f'GPT-PINN Solution for $\\nu$ = {round(nu,4)}') # 设置图表标题，显示当前 nu 值
    plt.colorbar(contour) # 添加颜色条
    plt.show() # 显示图表
```
> 💡 **解读：**
>
> - **功能**：以等高线图的形式可视化 GPT-PINN 针对特定 $\nu$ 值预测的二维解场 $u(x,t)$。
> - **核心**：
>   - **数据重塑**：将模型输出的一维解数据重塑为二维网格 (`x_grid`, `t_grid`, `u_gpt_grid`)，以适应等高线图的输入要求。
>   - `ax.contourf`：绘制填充的等高线图，通过颜色深浅表示函数值的大小。
> - **用途**：直观展示预测解 $u(x,t)$ 在不同空间和时间点上的分布，便于分析模型效果和物理现象。

> 🔍 **提示：**
>
> - `levels` 参数可以调整等高线的精细程度，`cmap` 选择合适的颜色映射以增强可视化效果。
> - 在实际分析中，通常会与解析解或高精度数值解的等高线图进行对比，以及绘制绝对误差图，以全面评估模型精度。

## 结果示例

### 最大损失随神经元数量变化图

该图展示了在离线阶段，随着生成基函数数量的增加，当前基函数集合在参数空间中最大近似误差（"最大损失"）的演变。这反映了自适应选择策略的有效性。

| ![img](https://www.science42.tech/cases/caseMarkdown/一维Burgers方程基函数预计算与在线求解/viz/max_losses.png)        |
| :---: |
| **Fig. 2** 最大损失随基函数数量变化 |

### 预测解与误差图

这些图展示了针对一个特定 $\nu$ 值，GPT-PINN 和标准 PINN 预测的 $u(x,t)$ 场，以及它们之间的绝对误差。请注意，这里的 $u(x,t)$ 场的坐标轴是时间 $t$ 和空间 $x$。
| ![img](https://www.science42.tech/cases/caseMarkdown/一维Burgers方程基函数预计算与在线求解/viz/gpt_pinn_plot_nu_0.4481.png)  |
| :---: |
|    **Fig. 3** GPT-PINN 预测的 $u(x,t)$     |

|   ![img](https://www.science42.tech/cases/caseMarkdown/一维Burgers方程基函数预计算与在线求解/viz/pinn_plot_nu_0.4481.png)    |
| :---: |
|    **Fig. 4** 标准 PINN 预测的 $u(x,t)$    |

|   ![img](https://www.science42.tech/cases/caseMarkdown/一维Burgers方程基函数预计算与在线求解/viz/error_plot_nu_0.4481.png)   |
| :---: |
| **Fig. 5** GPT-PINN 与标准 PINN 的绝对误差 |


## 贡献与许可证

感谢您对本项目 GPT-PINN 架构的关注和学习！您可以根据实际需求调整参数、修改代码结构或拓展到其他参数化 PDE 问题。

**贡献者：**

- Yang Zhang

**许可证：**  
本项目代码遵循 [GNU 通用公共许可证 第三版（GPLv3）](https://www.gnu.org/licenses/gpl-3.0.html)。您可以自由使用、修改和分发该代码，但需保留原始版权声明，并在分发时提供相同的许可证。

版权 © 硒钼科技（北京）。更多信息请访问 [Science42](https://www.science42.tech/#/index)。