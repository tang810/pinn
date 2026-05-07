<!-- 
text: "3D 线性弹性力学：翼型结构位移约束的物理信息神经网络（PINN）求解",
 area: "本项目专注于利用**物理信息神经网络（PINN）**高效解决**翼型结构**在复杂载荷下的三维线性弹性力学问题。特别是，通过施加精确的**位移边界约束**，实现对翼型变形的精确预测。传统数值方法在处理复杂几何和边界条件时面临网格划分的挑战，而PINN通过将控制方程和位移边界条件直接编码到神经网络的损失函数中，提供了一种无网格的解决方案。项目目标是精确预测三维翼型在给定载荷下的位移场 $(u_x, u_y, u_z)$。",
 tags: [ "3D 弹性力学", "PINN",   "无网格方法", "翼型结构", "位移约束" ] -->

# 3D 线性弹性力学：翼型结构位移约束的物理信息神经网络求解

## 项目概述

本项目旨在利用**物理信息神经网络 (Physics-Informed Neural Network, PINN)** 解决**三维 (3D) 线性弹性力学问题**，特别聚焦于**翼型结构**在**位移边界约束**下的变形分析。翼型结构在航空航天、风力发电等领域具有关键应用，其在复杂载荷下的位移响应对于结构完整性、气动性能和振动特性至关重要。传统的数值方法（如有限元法）在处理复杂翼型几何、多尺度问题或需要高精度梯度信息时，可能面临精细网格划分的巨大挑战和计算资源的限制。

PINN 作为一种新兴的求解偏微分方程（PDEs）的方法，通过将物理定律（控制方程）和初始/边界条件直接嵌入到神经网络的损失函数中，实现了无网格的端到端求解。它能够同时学习物理系统的状态（如位移场）和满足其所遵循的物理定律。

本项目专注于求解稳态三维弹性力学问题，通过训练一个深度神经网络，使其能够精确预测三维**翼型结构**在给定坐标 $(x, y, z)$ 处的位移场 $(u_x, u_y, u_z)$，特别是在施加了特定位移约束的工况下。本项目的核心特点包括：

*   直接在连续域上求解弹性力学方程，无需显式网格划分，尤其适用于复杂**翼型几何**。
*   利用自动微分精确计算位移场的梯度，进而得到应变和应力。
*   通过最小化结合物理方程残差和**位移边界条件**误差的复合损失函数进行训练。

|                   ![img](https://www.science42.tech/cases/caseMarkdown/翼型结构位移约束的物理信息神经网络求解/viz/result.png)                   |
| :----------------------------------------------------------: |
| **Fig. 1** 3D 弹性力学 PINN 预测的**翼型结构**位移场结果示例 |

该图展示了 PINN 模型对三维**翼型结构**位移场 $u, v, w$ 的参考解（Ref）与预测解（PINN）的对比。

## 核心功能与方法

*   **三维弹性力学求解:** 针对各向同性线性弹性材料，求解**翼型结构**在给定**体力（模拟外部载荷）**作用和**位移边界约束**下的平衡方程。
*   **物理信息神经网络 (PINN):** 将 3D 弹性力学 Navier-Cauchy 方程和**指定的位移边界条件**直接编码到神经网络的损失函数中。
*   **全连接神经网络 (FCN):** 采用多层全连接网络作为函数逼近器，从输入坐标 $(x, y, z)$ 预测位移 $(u, v, w)$。
*   **自动微分 (Automatic Differentiation):** 利用 PyTorch 的自动微分功能，计算位移场对空间坐标的各阶导数（用于构建应变、应力和 PDE 残差）。
*   **复合损失函数:** 包含 PDE 残差损失和**位移边界条件**损失，驱动网络学习满足物理定律和边界约束的解。
*   **多阶段学习率调度:** 在训练过程中采用阶梯式学习率衰减策略，以提高模型的收敛性和精度。
*   **三维可视化:** 提供直观的三维散点图，展示参考解和预测解的**翼型位移场**，便于结果对比和分析。



## 数学背景

本项目解决的是在三维域 $\Omega$ 内定义的**翼型结构**的**线性弹性力学问题**。该问题通过物理信息神经网络（PINN）实现求解，着重于在**位移边界约束**下的精确位移场预测。

### 物理约束

1.  **材料本构参数 (Material Constitutive Parameters)**
    对于各向同性线弹性材料，其力学行为由杨氏模量 $E$ 和泊松比 $\nu$ 决定。在弹性力学控制方程中，通常使用拉梅常数 $\lambda$ 和剪切模量 $G$（或 $\mu$），它们与 $E$ 和 $\nu$ 的关系如下：

    $$
    G = \frac{E}{2(1+\nu)} 
    $$
    
    $$
    \lambda = \frac{E\nu}{(1+\nu)(1-2\nu)} 
    $$
    
    在代码中，默认值设置为 $E = 100$ 和 $\nu = 0.3$。

2.  **小应变理论 (Small Strain Theory)**
    位移场 $(u, v, w)$ 在笛卡尔坐标系下的应变张量分量（工程应变）定义为：
    
    $$
    \varepsilon_{xx} = \frac{\partial u}{\partial x}, \quad \varepsilon_{yy} = \frac{\partial v}{\partial y}, \quad \varepsilon_{zz} = \frac{\partial w}{\partial z} 
    $$
    
    $$
    \gamma_{xy} = \frac{\partial u}{\partial y} + \frac{\partial v}{\partial x}, \quad \gamma_{yz} = \frac{\partial v}{\partial z} + \frac{\partial w}{\partial y}, \quad \gamma_{xz} = \frac{\partial u}{\partial z} + \frac{\partial w}{\partial x} 
    $$

3.  **线性本构关系 (Linear Constitutive Relation / Hooke's Law)**
    对于各向同性线弹性材料，应力张量分量与应变张量分量通过广义胡克定律关联。以正应力 $\sigma_{xx}$ 和剪应力 $\tau_{xy}$ 为例：
    
    $$
    \sigma_{xx} = \lambda(\varepsilon_{xx} + \varepsilon_{yy} + \varepsilon_{zz}) + 2G\varepsilon_{xx} 
    $$
    
    $$
    \tau_{xy} = G\gamma_{xy}
    $$
    
    （其他应力分量 $\sigma_{yy}, \sigma_{zz}, \tau_{yz}, \tau_{xz}$ 具有类似形式）

4.  **平衡方程 (Equilibrium Equations)**
    在存在体力 $f_x, f_y, f_z$ 的情况下，翼型结构在静力平衡时需要满足的平衡方程（或将其代入应变-位移和应力-应变关系后得到的**Navier-Cauchy 方程**）：
    **x 方向:**
    
    $$
    (2G+\lambda) \frac{\partial^2 u}{\partial x^2} + G \left( \frac{\partial^2 u}{\partial y^2} + \frac{\partial^2 u}{\partial z^2} \right) + (G+\lambda) \left( \frac{\partial^2 v}{\partial x \partial y} + \frac{\partial^2 w}{\partial x \partial z} \right) + f_x = 0 
    $$
    
    **y 方向:**
    
    $$
    (2G+\lambda) \frac{\partial^2 v}{\partial y^2} + G \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial z^2} \right) + (G+\lambda) \left( \frac{\partial^2 u}{\partial y \partial x} + \frac{\partial^2 w}{\partial y \partial z} \right) + f_y = 0 
    $$
    
    **z 方向:**
    
    $$
    (2G+\lambda) \frac{\partial^2 w}{\partial z^2} + G \left( \frac{\partial^2 w}{\partial x^2} + \frac{\partial^2 w}{\partial y^2} \right) + (G+\lambda) \left( \frac{\partial^2 u}{\partial z \partial x} + \frac{\partial^2 v}{\partial z \partial y} \right) + f_z = 0 
    $$
    
    在代码中，体力分量 $f_x, f_y, f_z$ 默认设置为 $0.1$。

5.  **位移边界条件 (Displacement Boundary Conditions)**
    在**翼型结构**的边界 $\partial\Omega_{BC}$ 上，施加 Dirichlet 位移边界条件，即直接给定位移值：
    
    $$
    u(x_b, y_b, z_b) = u_{\text{target}}(x_b, y_b, z_b)
    $$
    
    $$
    v(x_b, y_b, z_b) = v_{\text{target}}(x_b, y_b, z_b) 
    $$
    
    $$
    w(x_b, y_b, z_b) = w_{\text{target}}(x_b, y_b, z_b)
    $$
    这些目标位移值从预处理的 `.mat` 数据文件中加载，模拟了翼型在特定区域（如翼根）的固定或受限情况。

#### 物理方程参数说明

| 参数                      | 含义                                  | 默认值 |
| :------------------------ | :------------------------------------ | :----- |
| $E$                       | 杨氏模量 (Young's modulus)            | `100`  |
| $\nu$                     | 泊松比 (Poisson's ratio)              | `0.3`  |
| $G$                       | 剪切模量 (Shear modulus)              | —      |
| $\lambda$                 | 拉梅常数 (Lamé constant)              | —      |
| $u, v, w$                 | 位移分量在 **x, y, z** 方向           | —      |
| $\varepsilon_{xx}, \dots$ | 正应变分量 (e.g., $\varepsilon_{xx}$) | —      |
| $\gamma_{xy}, \dots$      | 剪应变分量 (e.g., $\gamma_{xy}$)      | —      |
| $\sigma_{xx}, \dots$      | 正应力分量 (e.g., $\sigma_{xx}$)      | —      |
| $\tau_{xy}, \dots$        | 剪应力分量 (e.g., $\tau_{xy}$)        | —      |
| $f_x, f_y, f_z$           | 体力分量                              | `0.1`  |

### PINN 损失函数

为了将物理约束嵌入到神经网络训练中，我们主要在损失函数中引入了平衡方程（Navier-Cauchy 方程）残差和位移边界条件残差。

#### PDE 残差损失 ($L_E$)

该损失项衡量神经网络预测的位移场在内部域搭配点上对 Navier-Cauchy 平衡方程的满足程度。

$$
L_E = \frac{1}{N_{\rm col}} \sum_{(x_c, y_c, z_c) \in \Omega_{\rm col}} \left( \left(\text{Eq1}\right)^2 + \left(\text{Eq2}\right)^2 + \left(\text{Eq3}\right)^2 \right) 
$$

其中，$\text{Eq1}, \text{Eq2}, \text{Eq3}$ 分别代表上述平衡方程在 $x, y, z$ 方向的左侧项。

#### 位移边界条件损失 ($L_{BC}$)

该损失项衡量神经网络预测的位移场在**翼型结构**边界上的值与给定目标位移之间的偏差。

$$
L_{BC} = \frac{1}{N_{BC}} \sum_{(x_b, y_b, z_b) \in \partial\Omega_{BC}} \left( (u_{NN}-u_{\text{target}})^2 + (v_{NN}-v_{\text{target}})^2 + (w_{NN}-w_{\text{target}})^2 \right) 
$$

#### 总损失函数 ($\mathcal{L}$)

总损失函数是 PDE 残差损失和位移边界条件损失的加权和：

$$
\mathcal{L} = \alpha_E L_E + \alpha_{BC} L_{BC}
$$

其中 $\alpha_E$ 和 $\alpha_{BC}$ 是可调节的权重系数，在代码中默认设置为 `lam_equ = 1` 和 `lam_bcs = 10`。通过最小化此总损失，神经网络被驱动学习满足所有物理约束的翼型位移场。

## 环境与依赖

-   Python ≥ 3.8
-   PyTorch ≥ 1.10 (代码中默认使用 `torch.float32`)
-   NumPy
-   Matplotlib (用于数据可视化)
-   SciPy (用于加载 `.mat` 数据文件)
-   `typing`, `collections`, `math`, `os` (标准库)
-   `scipy.stats.qmc` (此模块在代码中被导入但未直接使用，`LHSample` 函数在 `TOOLS` 模块中，但主训练流程未使用它进行数据生成。主要依赖为 `scipy.io`)

## 快速开始


### 安装依赖

建议使用虚拟环境进行安装：

```bash
conda create -n pinn_3d_elasticity_airfoil python=3.9
conda activate pinn_3d_elasticity_airfoil
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 # 根据你的CUDA版本选择
pip install numpy matplotlib scipy
```

### 运行脚本


**训练与评估 (一体化运行)**

直接运行 `main.py` 将执行完整的训练和评估流程。默认会进行 3 次独立的训练 (`for loop in range(0, 3)`)，每次训练包含多个学习率阶段和总计 240000 个 Epoch。

```bash
python main.py
```

**关键默认参数 (可在 `train` 函数中修改):**

*   `E = 100` (弹性模量)
*   `nu = 0.3` (泊松比)
*   `N_neu = 80` (神经网络隐藏层神经元数量)
*   `N_HLayer = 6` (神经网络隐藏层数量)
*   `N_f = 10000` (PDE 残差点数量)
*   `lam_bcs = 10` (边界条件损失权重)
*   `lam_equ = 1` (方程损失权重)
*   `num_epoch` (训练轮次，分为 30k, 30k, 30k, 50k, 100k Epochs，共 24 万 Epoch)
*   `lr` (学习率，逐阶段衰减)

训练过程中会周期性打印损失信息。训练完成后，模型将保存到 `./results/` 目录，并生成一个名为 `result.png` 的 3D 可视化图表，展示**翼型结构**的参考位移场与预测位移场的对比。

**单独测试已训练模型**

如果你已经有训练好的模型权重 (`.pth` 文件)，可以取消 `main.py` 文件底部 `if __name__ == "__main__":` 块中 `test_model` 部分的注释，并修改 `net_params` 指向你的模型路径，然后运行：

```bash
python main.py
```

这将加载指定模型，并对评估数据进行位移场预测和可视化。

## 模型构建

#### 代码单元 1：全连接神经网络模型 (FCNet)

这个代码块定义了 `FCNet` 类，它是一个**全连接神经网络 (Fully Connected Network, FCN)**，作为 PINN 的基础函数逼近器。它用于从输入的三维坐标 $(x, y, z)$ 预测**翼型结构**的位移场 $(u, v, w)$。


```python
class FCNet(torch.nn.Module):
    def __init__(self, num_ins=3,
                 num_outs=3,
                 num_layers=10, # 隐藏层数量
                 hidden_size=50, # 每层神经元数量
                 activation=torch.nn.Tanh):
        super(FCNet, self).__init__()

        # 构建神经网络的层结构，包括输入层、多个隐藏层和输出层
        layers = [num_ins] + [hidden_size] * num_layers + [num_outs]
        self.depth = len(layers) - 1 # 网络的深度 (层数)

        self.activation = activation # 使用 Tanh 激活函数

        layer_list = list()
        # 逐层构建线性层和激活函数
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))

        # 构建最后一层（输出层），没有激活函数
        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        # 将所有层组合成一个有序字典，再转换为 Sequential 模型
        layerDict = OrderedDict(layer_list)
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        # 前向传播，将输入 x 依次通过所有定义的层
        out = self.layers(x)
        return out
```

> 💡 **解读：**
>
> -   **功能**：实现一个可配置层数和每层神经元数量的全连接神经网络。
> -   **核心**：使用 `torch.nn.Linear` 定义线性变换层，`torch.nn.Tanh` 定义激活函数。`OrderedDict` 和 `torch.nn.Sequential` 使得网络结构清晰且易于构建。
> -   **用途**：作为 PINN 的核心近似器，接收三维坐标 `(x,y,z)` 作为输入，输出对应的**翼型**位移 `(u,v,w)`。

> 🔍 **提示：**
>
> -   `num_layers` 指的是隐藏层的数量，不包括输入层和输出层。
> -   `hidden_size` 决定了每个隐藏层的宽度。
> -   `Tanh` 是 PINN 中常用的激活函数，因其平滑性和良好的导数特性。也可以尝试 `SiLU` 或 `GELU` 等现代激活函数。
> -   模型的权重和偏置会在 `torch.nn.Linear` 初始化时自动采用 PyTorch 默认的初始化策略（如 Kaiming Uniform 或 Xavier Uniform）。


#### 代码单元 2：物理信息神经网络 (PysicsInformedNeuralNetwork) 框架

这个代码块定义了 `PysicsInformedNeuralNetwork` 类，它是整个 PINN 求解器的核心。它封装了神经网络模型、材料参数、损失函数计算逻辑、优化器和训练流程。

```python
class PysicsInformedNeuralNetwork:
    def __init__(self,
                 opt=None,
                 E = 100, # 弹性模量
                 nu = 0.3, # 泊松比
                 layers=10, # 隐藏层数量
                 hidden_size=50,  # 隐藏层神经元数量
                 learning_rate=0.001,
                 weight_decay=0.9,
                 outlet_weight=1, # 此项目未使用
                 bc_weight=1, # 边界条件损失权重
                 eq_weight=1, # 方程损失权重
                 ic_weight=1, # 初始条件损失权重，此项目未使用
                 num_ins=3, # 输入维度 (x,y,z)
                 num_outs=3, # 输出维度 (u,v,w)
                 supervised_data_weight=1, # 此项目未使用
                 training_type='unsupervised', # 此项目未使用
                 net_params=None, # 预训练模型参数路径
                 checkpoint_freq=2000,
                 checkpoint_path='./checkpoint/'):

        self.E = E
        self.nu = nu
        self.hidden_size = hidden_size

        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_path = checkpoint_path

        # 损失权重
        self.alpha_b = bc_weight # 边界条件权重
        self.alpha_e = eq_weight # 方程权重
        # 初始化损失值，用于打印
        self.loss_i = self.loss_o = self.loss_b = self.loss_e = self.loss_s = 0.0

        # 初始化神经网络 (FCNet)
        self.net = self.initialize_NN(
                num_ins=num_ins,
                num_outs=num_outs,
                num_layers=layers, # 隐藏层数量
                hidden_size=self.hidden_size).to(device)

        # 如果提供了预训练模型参数，则加载
        if net_params:
            load_params = torch.load(net_params)
            self.net.load_state_dict(load_params)

        # 初始化优化器 (Adam)
        self.opt = torch.optim.Adam(
            params=self.net.parameters(),
            lr=learning_rate,
            weight_decay=0) if not opt else opt

    def initialize_NN(self,
                      num_ins=3,
                      num_outs=3,
                      num_layers=10,
                      hidden_size=50):
        # 参数类型检查
        if not isinstance(num_layers, int):
            raise TypeError(f"num_layers must be an integer, got {type(num_layers)}")
        if not isinstance(hidden_size, int):
            raise TypeError(f"hidden_size must be an integer, got {type(hidden_size)}")

        # 返回 FCNet 实例
        return FCNet(num_ins=num_ins,
                     num_outs=num_outs,
                     num_layers=num_layers,
                     hidden_size=hidden_size,
                     activation=torch.nn.Tanh)

    def neural_net_u(self, x, y, z):
        # 将输入坐标 (x,y,z) 拼接成一个张量，送入神经网络
        X = torch.cat((x, y, z), dim=1)
        uvw = self.net(X) # 神经网络预测输出 (u,v,w)
        u = uvw[:, 0:1] # 提取 u 分量
        v = uvw[:, 1:2] # 提取 v 分量
        w = uvw[:, 2:3] # 提取 w 分量
        return u, v, w

    def predict(self, net_params, X):
        # 外部调用的预测接口
        x, y, z = X
        return self.neural_net_u(x, y, z)
```

> 💡 **解读：**
>
> -   **功能**：构建 PINN 求解器的骨架，包括神经网络模型、物理参数、损失权重和优化器。
> -   **核心**：
>     -   `initialize_NN` 负责创建 `FCNet` 实例。
>     -   `neural_net_u` 执行神经网络的前向传播，将 3D 坐标映射到 3D 位移。
>     -   初始化时可以加载预训练模型，并配置 Adam 优化器。
> -   **用途**：作为整个 PINN 训练和预测过程的统一接口，用于求解**翼型结构**的弹性问题。

> 🔍 **提示：**
>
> -   `E` 和 `nu` 是材料属性，它们在 `neural_net_equations` 方法中用于计算拉梅常数。
> -   `bc_weight` (`alpha_b`) 和 `eq_weight` (`alpha_e`) 是超参数，它们的设置对训练效果至关重要，可能需要进行调优。


## 模型训练

#### 代码单元 3：数据加载与预处理 (DataLoader)

这个代码块定义了 `DataLoader` 类，负责从 MATLAB `.mat` 文件中加载所有必要的训练和评估数据，包括**翼型结构的位移边界条件点**、PDE 搭配点和参考解数据。

```python
class DataLoader:
    def __init__(self, path=None, N_f=20000, N_b=1000):
        '''
        N_f: PDE 方程残差点数量
        N_b: 边界条件点数量 (此项目中 N_b 未直接在加载函数中使用)
        '''
        self.N_f = N_f # 方程点数量

    def loading_boundary_data(self, filename):
        # 从 .mat 文件加载边界数据
        data = scipy.io.loadmat(filename)
        # 边界点的坐标和对应的目标位移值
        bx = data['bx']; by = data['by']; bz = data['bz']
        lx = data['lx']; ly = data['ly']; lz = data['lz']
        ux = data['ux']; uy = data['uy']; uz = data['uz']
        b_u = data['b_u']; b_v = data['b_v']; b_w = data['b_w'] # 修正：原始代码这里v和w都用了'b_w'，已更正
        l_u = data['l_u']; l_v = data['l_v']; l_w = data['l_w']
        u_u = data['u_u']; u_v = data['u_v']; u_w = data['u_w']

        # 将所有边界点坐标和目标位移拼接起来
        x_b = np.concatenate([bx, lx, ux], axis=0).reshape([-1, 1])
        y_b = np.concatenate([by, ly, uy], axis=0).reshape([-1, 1])
        z_b = np.concatenate([bz, lz, uz], axis=0).reshape([-1, 1])
        x_u = np.concatenate([b_u, l_u, u_u], axis=0).reshape([-1, 1])
        y_v = np.concatenate([b_v, l_v, u_v], axis=0).reshape([-1, 1])
        z_w = np.concatenate([b_w, l_w, u_w], axis=0).reshape([-1, 1])

        N_train_bcs = x_b.shape[0]
        print('-----------------------------')
        print('N_train_bcs: ' + str(N_train_bcs))
        print('N_train_equ: ' + str(self.N_f))
        print('-----------------------------')
        return x_b, y_b, z_b, x_u, y_v, z_w

    def loading_training_data(self, filename):
        # 从 .mat 文件加载 PDE 方程训练点 (搭配点)
        data = scipy.io.loadmat(filename)
        x = data['x']
        y = data['y']
        z = data['z']
        x_train_f = x
        y_train_f = y
        z_train_f = z
        return x_train_f, y_train_f, z_train_f

    def loading_evaluate_data(self, filename):
        # 从 .mat 文件加载评估点及其参考解
        data = scipy.io.loadmat(filename)
        x_star = data['x']
        y_star = data['y']
        z_star = data['z']
        u_star = data['u_ref']
        v_star = data['v_ref']
        w_star = data['w_ref']
        return x_star, y_star, z_star, u_star, v_star, w_star
```

> 💡 **解读：**
>
> -   **功能**：封装了从外部 `.mat` 文件加载**翼型结构**所需数据的逻辑。
> -   **核心**：使用 `scipy.io.loadmat` 读取 MATLAB 文件，并将不同类型的数据（**位移边界**、方程点、评估点）分离并格式化。
> -   **用途**：为 PINN 训练提供所有必需的输入数据和验证数据。

> 🔍 **提示：**
>
> -   请确保 `data.mat` 文件中的变量名与 `loading_boundary_data`, `loading_training_data`, `loading_evaluate_data` 方法中使用的键（如 `'bx'`, `'x_ref'` 等）完全匹配。
> -   `loading_boundary_data` 中 `b_v = data['b_w']` 和 `b_w = data['b_w']` 这一行已被修正为 `b_v = data['b_v']` 和 `b_w = data['b_w']`。

#### 代码单元 4：弹性力学 PDE 残差计算

这个代码块展示了 `PysicsInformedNeuralNetwork` 类中用于计算**Navier-Cauchy 方程 PDE 残差**的方法 `neural_net_equations`。它利用自动微分计算预测位移场的各阶导数，并构建物理方程的残差项，这些方程描述了**翼型结构**在受力下的平衡状态。

```python
class PysicsInformedNeuralNetwork: # ... (部分代码省略，仅展示相关方法)
    # ...
    @torch.jit.script
    def autograd(y: torch.Tensor, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        TorchScript 函数，用于计算张量 y 对多个输入 x 的梯度。
        """
        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(y, device=y.device)]
        grad = torch.autograd.grad(
            [
                y,
            ],
            x,
            grad_outputs=grad_outputs,
            create_graph=True, # 允许创建更高阶的导数图
            allow_unused=True, # 允许不使用的输入
        )

        if grad is None:
            grad = [torch.zeros_like(xx) for xx in x]
        assert grad is not None
        grad = [g if g is not None else torch.zeros_like(x[i]) for i, g in enumerate(grad)]
        return grad

    def neural_net_equations(self, x, y, z):
        # 将输入坐标 (x,y,z) 拼接，送入神经网络预测位移 (u,v,w)
        X = torch.cat((x, y, z), dim=1)
        uvw = self.net(X)
        u = uvw[:, 0:1]
        v = uvw[:, 1:2]
        w = uvw[:, 2:3]

        # 计算位移分量 u, v, w 的一阶导数
        u_x, u_y, u_z = self.autograd(u, [x, y, z])
        v_x, v_y, v_z = self.autograd(v, [x, y, z])
        w_x, w_y, w_z = self.autograd(w, [x, y, z])

        # 计算位移分量的二阶导数 (PDE 中需要的混合导数和纯二阶导数)
        u_xx = self.autograd(u_x, [x])[0]
        u_yy = self.autograd(u_y, [y])[0]
        u_zz = self.autograd(u_z, [z])[0]
        u_xy = self.autograd(u_x, [y])[0] # 或 self.autograd(u_y, [x])[0]
        u_xz = self.autograd(u_x, [z])[0] # 或 self.autograd(u_z, [x])[0]

        v_xx = self.autograd(v_x, [x])[0]
        v_yy = self.autograd(v_y, [y])[0]
        v_zz = self.autograd(v_z, [z])[0]
        v_xy = self.autograd(v_x, [y])[0] # 或 self.autograd(v_y, [x])[0]
        v_yz = self.autograd(v_y, [z])[0] # 或 self.autograd(v_z, [y])[0]

        w_xx = self.autograd(w_x, [x])[0]
        w_yy = self.autograd(w_y, [y])[0]
        w_zz = self.autograd(w_z, [z])[0]
        w_xz = self.autograd(w_x, [z])[0] # 或 self.autograd(w_z, [x])[0]
        w_yz = self.autograd(w_y, [z])[0] # 或 self.autograd(w_z, [y])[0]

        # 计算拉梅常数 G 和 lambda (la)
        G = ((self.E/(2*(1+self.nu))))
        la = (self.E * self.nu)/((1 + self.nu)*(1 - 2 * self.nu))

        # 定义体力分量，模拟翼型所受的均布载荷
        f_1 = 0.1
        f_2 = 0.1
        f_3 = 0.1

        # 构建 Navier-Cauchy 方程的残差项 (eq1, eq2, eq3)
        # x 方向平衡方程残差
        eq1 = ((2 * G + la) * ( u_xx ) + G * (u_yy + u_zz) + ( G + la )*(v_xy + w_xz) + f_1)
        # y 方向平衡方程残差
        eq2 = ((2 * G + la) * ( v_yy ) + G * (v_xx + v_zz) + ( G + la )*(u_xy + w_yz) + f_2)
        # z 方向平衡方程残差
        eq3 = ((2 * G + la) * ( w_zz ) + G * (w_xx + w_yy) + ( G + la )*(u_xz + v_yz) + f_3)

        return eq1, eq2, eq3
```

> 💡 **解读：**
>
> -   **功能**：计算 3D 线性弹性力学 Navier-Cauchy 方程的残差。
> -   **核心**：
>     -   `autograd` 函数是 PyTorch `torch.autograd.grad` 的封装，用于高效计算高阶导数。`@torch.jit.script` 装饰器用于优化性能。
>     -   根据 Navier-Cauchy 方程的数学形式，利用链式法则和自动微分，构建三个平衡方程的左侧项。
>     -   拉梅常数 `G` 和 `la` 随材料参数 `E` 和 `nu` 动态计算。
> -   **用途**：作为 PINN 损失函数中的 PDE 残差项，强制神经网络学习满足**翼型结构**平衡方程的解。

> 🔍 **提示：**
>
> -   在自动微分中，为了计算二阶导数，对一阶导数再次调用 `autograd`。`create_graph=True` 在第一次 `autograd` 调用时是必需的，以便构建计算图用于第二次反向传播。
> -   体力 `f_1, f_2, f_3` 是常数，可以根据**翼型结构**的实际受力情况进行调整。

#### 代码单元 5：损失函数前向计算

这个代码块展示了 `PysicsInformedNeuralNetwork` 类中用于计算**总损失函数**的方法 `fwd_computing_loss_3d`。它结合了**翼型结构表面上的位移边界损失**和 PDE 残差损失。

```python
class PysicsInformedNeuralNetwork: # ... (部分代码省略，仅展示相关方法)
    # ...
    def fwd_computing_loss_3d(self, loss_mode='MSE'):
        # 1. 计算边界条件损失
        # 在边界点上预测位移 (u_pred_b, v_pred_b, w_pred_b)
        (self.u_pred_b, self.v_pred_b, self.w_pred_b) = self.neural_net_u(self.x_b, self.y_b, self.z_b)

        # 根据指定的损失模式 (MSE 或 L2) 计算边界条件损失
        if loss_mode == 'L2':
            # L2 范数 (欧几里得距离)
            self.loss_b = torch.norm((self.x_u.reshape([-1]) - self.u_pred_b.reshape([-1])), p=2) + \
                          torch.norm((self.y_v.reshape([-1]) - self.v_pred_b.reshape([-1])), p=2) + \
                          torch.norm((self.z_w.reshape([-1]) - self.w_pred_b.reshape([-1])), p=2)
        if loss_mode == 'MSE':
            # 均方误差 (Mean Squared Error)
            self.loss_b = torch.mean(torch.square(self.x_u.reshape([-1]) - self.u_pred_b.reshape([-1]))) + \
                          torch.mean(torch.square(self.y_v.reshape([-1]) - self.v_pred_b.reshape([-1]))) + \
                          torch.mean(torch.square(self.z_w.reshape([-1]) - self.w_pred_b.reshape([-1])))

        # 2. 计算 PDE 残差损失
        # 在 PDE 搭配点上计算 Navier-Cauchy 方程的残差 (eq11_pred, eq22_pred, eq33_pred)
        assert self.x_f is not None and self.y_f is not None and self.z_f is not None # 确保方程点已设置

        (self.eq11_pred, self.eq22_pred,
         self.eq33_pred) = self.neural_net_equations(self.x_f, self.y_f, self.z_f)

        # 根据指定的损失模式计算 PDE 残差损失
        if loss_mode == 'L2':
            self.loss_e = torch.norm(self.eq11_pred.reshape([-1]), p=2) + \
                          torch.norm(self.eq22_pred.reshape([-1]), p=2) + \
                          torch.norm(self.eq33_pred.reshape([-1]), p=2)
        if loss_mode == 'MSE':
            self.loss_e = torch.mean(torch.square(self.eq11_pred.reshape([-1]))) + \
                          torch.mean(torch.square(self.eq22_pred.reshape([-1]))) + \
                          torch.mean(torch.square(self.eq33_pred.reshape([-1])))

        # 3. 计算总损失 (加权求和)
        self.loss = self.alpha_b * self.loss_b + self.alpha_e * self.loss_e

        # 返回总损失和各项子损失
        return self.loss, [self.loss_e, self.loss_b]
```

> 💡 **解读：**
>
> -   **功能**：根据当前神经网络的预测和物理定律/**位移边界条件**，计算一个综合的损失值。
> -   **核心**：
>     -   调用 `neural_net_u` 计算**翼型边界**上的位移预测，并与目标值计算误差（`loss_b`）。
>     -   调用 `neural_net_equations` 计算 PDE 残差，并对其平方求均值（`loss_e`）。
>     -   通过超参数 `alpha_b` 和 `alpha_e` 对两类损失进行加权求和，得到最终的总损失。
> -   **用途**：驱动神经网络在训练过程中同时满足**翼型结构**的边界条件和物理方程。

> 🔍 **提示：**
>
> -   `loss_mode` 参数允许在 L2 范数和 MSE 之间切换损失计算方式。通常使用 MSE (均方误差)。
> -   损失权重 `alpha_b` 和 `alpha_e` 的平衡至关重要，它们直接影响模型收敛的方向和最终精度。
> -   `.reshape([-1])` 用于将张量展平为一维，便于计算范数或均方误差。


#### 代码单元 6：训练循环与优化

这个代码块展示了 `PysicsInformedNeuralNetwork` 类中用于执行**训练过程**的方法 `train` 和 `solve_Adam`。它实现了基于 Adam 优化器的迭代训练循环，并包括日志打印。

```python
class PysicsInformedNeuralNetwork: # ... (部分代码省略，仅展示相关方法)
    # ...
    def train(self,
              num_epoch=1,
              lr=1e-4,
              optimizer=None, # 可选：外部传入优化器
              scheduler=None, # 可选：学习率调度器
              batchsize=None): # 此项目当前版本不使用批次训练
        if optimizer is not None:
            self.opt = optimizer # 使用外部优化器
        else:
            self.opt = torch.optim.Adam(params=self.net.parameters(), lr=lr) # 否则创建新的 Adam 优化器
        # 调用核心求解函数
        return self.solve_Adam(self.fwd_computing_loss_3d, num_epoch, batchsize, scheduler)

    def solve_Adam(self,
                   loss_func, # 损失计算函数 (fwd_computing_loss_3d)
                   num_epoch=1000,
                   batchsize=None, # 实际未用
                   scheduler=None): # 实际未用
        for epoch_id in range(num_epoch): # 迭代训练指定轮次
            loss, losses = loss_func() # 计算总损失及各项子损失
            loss.backward() # 反向传播，计算梯度
            self.opt.step() # 更新神经网络参数
            self.opt.zero_grad() # 清零梯度，为下一轮迭代做准备
            if scheduler: # 如果有学习率调度器，则更新学习率 (此项目中未传入)
                scheduler.step()

            # 每 100 个 Epoch 打印一次日志
            if epoch_id == 0 or (epoch_id + 1)%100 == 0:
                self.print_log(loss, losses, epoch_id, num_epoch)

    def print_log(self, loss, losses, epoch_id, num_epoch):
        # 辅助函数：获取当前学习率
        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']

        print("current lr is {}".format(get_lr(self.opt)))
        # 格式化打印各项损失值和总损失
        if isinstance(losses[0], int): # 检查损失是否已计算
            eq_loss = losses[0]
        else:
            eq_loss = losses[0].detach().cpu().item() # 从 GPU 转移到 CPU 并转换为 Python 数值

        print("epoch/num_epoch: ", epoch_id + 1, "/", num_epoch,
              "loss[Adam]: %.3e" # 总损失
              %(loss.detach().cpu().item()), "eq_loss: %.3e " %(eq_loss), "bc_loss: %.3e" # 方程损失和边界损失
              %(losses[1].detach().cpu().item()))

# ============ TRAIN FUNCTION (Main call) ============
def train(net_params=None, loop=None):
    # 材料参数设置
    E = 100
    nu = 0.3

    # 神经网络和训练参数设置
    N_neu = 80  # 隐藏层神经元数量
    lam_bcs = 10  # 边界条件损失权重
    lam_equ = 1  # 方程损失权重
    N_f = 10000  # PDE 残差点数量
    N_HLayer = 6  # 隐藏层数量

    # 创建 PINN 实例
    PINN = PysicsInformedNeuralNetwork(
        E=E,
        nu=nu,
        layers=N_HLayer,
        hidden_size=N_neu,
        bc_weight=lam_bcs,
        eq_weight=lam_equ,
        net_params=net_params,
        checkpoint_path='./checkpoint/'
    )

    # 数据加载
    path = './datasets/'
    dataloader = DataLoader(path=path, N_f=N_f, N_b=1000)
    filename = './data/data.mat'

    # 设置训练数据和评估数据
    boundary_data = dataloader.loading_boundary_data(filename)
    PINN.set_boundary_data(X=boundary_data)
    training_data = dataloader.loading_training_data(filename)
    PINN.set_eq_training_data(X=training_data)
    x_star, y_star, z_star, u_star, v_star, w_star = dataloader.loading_evaluate_data(filename)

    # 多阶段训练，不同学习率和 epoch 数
    # 阶段 1
    PINN.train(num_epoch=30000, lr=1e-3)
    PINN.test(x_star, y_star, z_star, u_star, v_star, w_star, 30000, loop) # 测试并保存结果
    saved_ckpt = 'model_SM_loop%d_epoch%d.pth'%(loop, 30000) # 保存模型
    PINN.save(saved_ckpt, N_HLayer=N_HLayer, N_neu=N_neu, N_f=N_f)

    # 阶段 2, 3, 4, 5... (重复训练和测试)
    PINN.train(num_epoch=30000, lr=2e-4) # 累计 60000 epoch
    PINN.test(x_star, y_star, z_star, u_star, v_star, w_star, 60000, loop)
    PINN.train(num_epoch=30000, lr=4e-5) # 累计 90000 epoch
    PINN.test(x_star, y_star, z_star, u_star, v_star, w_star, 90000, loop)
    saved_ckpt = 'model_SM_loop%d_epoch%d.pth'%(loop, 90000)
    PINN.save(saved_ckpt, N_HLayer=N_HLayer, N_neu=N_neu, N_f=N_f)

    PINN.train(num_epoch=50000, lr=1e-5) # 累计 140000 epoch
    PINN.train(num_epoch=100000, lr=2e-6) # 累计 240000 epoch
    PINN.test(x_star, y_star, z_star, u_star, v_star, w_star, 240000, loop)
    saved_ckpt = 'model_SM_loop%d_epoch%d.pth'%(loop, 240000)
    PINN.save(saved_ckpt, N_HLayer=N_HLayer, N_neu=N_neu, N_f=N_f)
```

> 💡 **解读：**
>
> -   **功能**：实现 PINN 模型的端到端训练过程。
> -   **核心**：
>     -   `solve_Adam` 是核心优化循环，执行梯度计算 (`loss.backward()`)、参数更新 (`self.opt.step()`) 和梯度清零 (`self.opt.zero_grad()`)。
>     -   `train` 函数（外部调用）负责设置 PINN 实例、加载**翼型结构**数据，并按照预设的多个学习率阶段和 Epoch 数量进行训练。
> -   **用途**：使神经网络参数收敛到最佳状态，从而预测出满足物理约束的**翼型位移场**。

> 🔍 **提示：**
>
> -   多阶段学习率衰减是一种常见的训练策略，有助于模型在训练初期快速收敛，在后期稳定收敛到更精确的解。
> -   `batchsize` 参数在当前代码中未被使用，这意味着是全批量 (Full-batch) 训练，即所有训练点一次性参与损失计算。对于大规模数据集，可能需要实现小批量 (Mini-batch) 训练。
> -   `print_log` 提供了训练进度的实时反馈，有助于监控模型状态。
> -   模型训练中间会通过 `PINN.save` 保存检查点，方便中断后继续训练或进行中间结果分析。

## 可视化

#### 代码单元 7：3D 位移场可视化 (evaluate)

这个代码块展示了 `PysicsInformedNeuralNetwork` 类中用于**评估模型性能和进行 3D 可视化**的方法 `evaluate`。它计算预测解与参考解的相对 L2 误差，并绘制三维散点图来对比**翼型结构**的位移场。

```python
class PysicsInformedNeuralNetwork: # ... (部分代码省略，仅展示相关方法)
    # ...
    def evaluate(self, x, y, z, u, v, w):
        """ 在整个域内测试所有点，并进行可视化 """
        # 将输入 NumPy 数组转换为 PyTorch 张量并移动到设备 (CPU/GPU)
        x_test = torch.tensor(x.reshape(-1,1)).float().to(device)
        y_test = torch.tensor(y.reshape(-1,1)).float().to(device)
        z_test = torch.tensor(z.reshape(-1,1)).float().to(device)
        # 参考解也转换为 NumPy 数组，用于误差计算和绘图
        u_test = u.reshape(-1,1)
        v_test = v.reshape(-1,1)
        w_test = w.reshape(-1,1)

        # 通过神经网络预测位移 (u_pred, v_pred, w_pred)
        u_pred, v_pred, w_pred = self.neural_net_u(x_test, y_test, z_test)

        # 将预测结果从 PyTorch 张量转换回 NumPy 数组，以便进行误差计算和 Matplotlib 绘图
        x_test_np = x_test.detach().cpu().numpy().reshape(-1,1)
        y_test_np = y_test.detach().cpu().numpy().reshape(-1,1)
        z_test_np = z_test.detach().cpu().numpy().reshape(-1,1)
        u_pred_np = u_pred.detach().cpu().numpy().reshape(-1,1)
        v_pred_np = v_pred.detach().cpu().numpy().reshape(-1,1)
        w_pred_np = w_pred.detach().cpu().numpy().reshape(-1,1)

        # 计算相对 L2 误差
        error_u = np.linalg.norm(u_test-u_pred_np,2)/np.linalg.norm(u_test,2)
        error_v = np.linalg.norm(v_test-v_pred_np,2)/np.linalg.norm(v_test,2)
        error_w = np.linalg.norm(w_test-w_pred_np,2)/np.linalg.norm(w_test,2)
        print('------------------------')
        print('Error u: %e' % (error_u))
        print('Error v: %e' % (error_v))
        print('Error w: %e' % (error_w))
        print('------------------------')

        # --- 3D 可视化部分 ---
        # 准备绘图数据 (使用 _np 后缀表示 NumPy 数组)
        X_ref, Y_ref, Z_ref = x_test_np, y_test_np, z_test_np
        u_ref, v_ref, w_ref = u_test, v_test, w_test # 参考解
        u_pred_pinn, v_pred_pinn, w_pred_pinn = u_pred_np, v_pred_np, w_pred_np # PINN 预测解

        # 设置图表字体和大小
        font = 'Times New Roman'
        fig = plt.figure(figsize=(20, 10), facecolor='white') # 创建一个大图，包含多个子图

        # 绘制参考解 u, v, w (翼型结构位移)
        # ax1: U_ref
        ax1 = fig.add_subplot(2, 3, 1, projection='3d') # 2行3列的第一个子图
        scatter1 = ax1.scatter(Y_ref, X_ref, Z_ref, c=u_ref, cmap='jet') # 3D 散点图，颜色代表 u_ref 值
        ax1.set_xlabel('y [m]', fontname=font, fontsize=14) # 设置轴标签
        ax1.set_ylabel('x [m]', fontname=font, fontsize=14)
        ax1.set_zlabel('z [m]', fontname=font, fontsize=14)
        ax1.set_title('U_ref (Airfoil)', fontname=font, fontsize=20, fontweight='bold') # 设置标题
        ax1.set_box_aspect([15, 5, 5]) # 设置轴比例以美化显示

        # ax2: V_ref, ax3: W_ref (类似 ax1 的设置)
        ax2 = fig.add_subplot(2, 3, 2, projection='3d'); scatter2 = ax2.scatter(Y_ref, X_ref, Z_ref, c=v_ref, cmap='jet'); ax2.set_xlabel('y [m]'); ax2.set_ylabel('x [m]'); ax2.set_zlabel('z [m]'); ax2.set_title('V_ref (Airfoil)'); ax2.set_box_aspect([15, 5, 5])
        ax3 = fig.add_subplot(2, 3, 3, projection='3d'); scatter3 = ax3.scatter(Y_ref, X_ref, Z_ref, c=w_ref, cmap='jet'); ax3.set_xlabel('y [m]'); ax3.set_ylabel('x [m]'); ax3.set_zlabel('z [m]'); ax3.set_title('W_ref (Airfoil)'); ax3.set_box_aspect([15, 5, 5])

        # 绘制 PINN 预测解 u, v, w (翼型结构位移)
        # ax4: U_PINN
        ax4 = fig.add_subplot(2, 3, 4, projection='3d'); scatter4 = ax4.scatter(Y_ref, X_ref, Z_ref, c=u_pred_pinn, cmap='jet'); ax4.set_xlabel('y [m]'); ax4.set_ylabel('x [m]'); ax4.set_zlabel('z [m]'); ax4.set_title('U_PINN (Airfoil)'); ax4.set_box_aspect([15, 5, 5])
        # ax5: V_PINN, ax6: W_PINN (类似 ax4 的设置)
        ax5 = fig.add_subplot(2, 3, 5, projection='3d'); scatter5 = ax5.scatter(Y_ref, X_ref, Z_ref, c=v_pred_pinn, cmap='jet'); ax5.set_xlabel('y [m]'); ax5.set_ylabel('x [m]'); ax5.set_zlabel('z [m]'); ax5.set_title('V_PINN (Airfoil)'); ax5.set_box_aspect([15, 5, 5])
        ax6 = fig.add_subplot(2, 3, 6, projection='3d'); scatter6 = ax6.scatter(Y_ref, X_ref, Z_ref, c=w_pred_pinn, cmap='jet'); ax6.set_xlabel('y [m]'); ax6.set_ylabel('x [m]'); ax6.set_zlabel('z [m]'); ax6.set_title('W_PINN (Airfoil)'); ax6.set_box_aspect([15, 5, 5])

        plt.suptitle('3D Elasticity Results (Airfoil Structure)', fontname=font, fontsize=20, fontweight='bold') # 总标题
        plt.subplots_adjust(wspace=0.3, hspace=0.3) # 调整子图间距
        plt.savefig('result.png', dpi=300) # 保存结果图
        plt.show() # 显示图表
```

> 💡 **解读：**
>
> -   **功能**：对训练好的 PINN 模型进行性能评估（通过计算相对 L2 误差）和可视化。
> -   **核心**：
>     -   通过 `neural_net_u` 获取模型在评估点上的预测位移。
>     -   使用 `np.linalg.norm` 计算预测值与参考值之间的相对 L2 误差，量化模型精度。
>     -   利用 `matplotlib.pyplot` 的 3D 绘图功能 (`fig.add_subplot(projection='3d')` 和 `ax.scatter`) 绘制三维散点图，通过颜色映射展示**翼型结构**的位移场分布。
> -   **用途**：直观展示模型的预测效果，并提供量化误差指标，便于分析和比较。

> 🔍 **提示：**
>
> -   `.detach().cpu().numpy()` 是一系列重要操作，用于将 PyTorch 张量从 GPU 转移到 CPU，并转换为 NumPy 数组，这是 Matplotlib 绘图所需要的格式。
> -   `set_box_aspect` 可以调整 3D 坐标轴的比例，以获得更好的视觉效果。
> -   `cmap='jet'` 是一种常用的颜色映射方案，可以根据需要选择其他方案（如 `'viridis'`, `'plasma'` 等）。
> -   保存的 `result.png` 文件是展示**翼型位移场**模型效果的关键图形。

## 结果示例

### 翼型位移预测与参考解对比图

这些图展示了 PINN 模型预测的三维**翼型位移场** $(u, v, w)$ 与参考解（从 `data.mat` 加载）的对比。通过颜色深浅可以直观地观察到位移在空间中的分布，以及模型预测的准确性。

|         ![img](https://www.science42.tech/cases/caseMarkdown/翼型结构位移约束的物理信息神经网络求解/viz/1.jpg)         |         ![img](https://www.science42.tech/cases/caseMarkdown/翼型结构位移约束的物理信息神经网络求解/viz/2.jpg)         |         ![img](https://www.science42.tech/cases/caseMarkdown/翼型结构位移约束的物理信息神经网络求解/viz/3.jpg)         |
| :---------------------------------: | :---------------------------------: | :---------------------------------: |
| **Fig. 2** 参考解 $u(x,y,z)$ (翼型) | **Fig. 3** 参考解 $v(x,y,z)$ (翼型) | **Fig. 4** 参考解 $w(x,y,z)$ (翼型) |

|          ![img](https://www.science42.tech/cases/caseMarkdown/翼型结构位移约束的物理信息神经网络求解/viz/4.jpg)           |          ![img](https://www.science42.tech/cases/caseMarkdown/翼型结构位移约束的物理信息神经网络求解/viz/5.jpg)           |          ![img](https://www.science42.tech/cases/caseMarkdown/翼型结构位移约束的物理信息神经网络求解/viz/6.jpg)           |
| :------------------------------------: | :------------------------------------: | :------------------------------------: |
| **Fig. 5** PINN 预测 $u(x,y,z)$ (翼型) | **Fig. 6** PINN 预测 $v(x,y,z)$ (翼型) | **Fig. 7** PINN 预测 $w(x,y,z)$ (翼型) |

## 贡献与许可证

感谢您对本项目 3D 弹性力学 PINN 求解器在**翼型结构**应用中的关注和学习！您可以根据实际需求调整材料参数、神经网络结构、损失权重，或将其拓展到更复杂的弹性力学问题（如塑性、大变形、**不同翼型设计**、**气动弹性分析**等）。

**贡献者：**

-   Yang Zhang

**许可证：**  
本项目代码遵循 [GNU 通用公共许可证 第三版（GPLv3）](https://www.gnu.org/licenses/gpl-3.0.html)。您可以自由使用、修改和分发该代码，但需保留原始版权声明，并在分发时提供相同的许可证。

版权 © 硒钼科技（北京）。更多信息请访问 [Science42](https://www.science42.tech/#/index)。