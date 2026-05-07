<!-- 
text："地质勘探中的三维电阻反演"
area: "本项目聚焦于地质勘探中的三维电阻反演问题，旨在通过先进的算法和技术，准确获取地下电阻率分布情况，为地质结构分析和资源探测提供有力支持。在地质勘探领域，准确了解地下电阻率分布对于识别地质构造、探测矿产资源、评估地下水文情况等具有重要意义。项目采用了不同维度（2D、3D）的电阻反演模型。通过生成合成数据模拟实际观测情况，并利用正演模型计算理论数据，再通过反演算法根据观测数据反推地下真实的电阻率模型。同时，项目还提供了数据可视化和模型导出等功能，方便用户对结果进行分析和应用。"  
tags: ["三维电阻反演", "数据可视化", "反演算法"]
-->

# 地质勘探中的三维电阻反演系统

## 快速使用

您可跳转[使用示例](#使用示例)查看如何使用秋月白平台快速执行该类问题，以下是实现逻辑详解：
## 背景介绍

在地质勘探中，了解地下电阻率分布对于识别地下地质结构、寻找矿产资源等具有重要意义。常见任务包括：

- **目标识别**：判断地下是否存在高电阻率或低电阻率异常区域（如金属矿、地下水等）
- **参数反演**：推断地下不同区域的电阻率值和地质结构


## 原理介绍
我们实际拥有的是地表测量到的一组电位或视电阻率数据。目标是找到一个地下电阻率模型，使得该模型的正演响应与实际观测数据尽可能匹配。这是一个 “由果索因” 的逆向过程。

![caseMarkdown/地质勘探中的三维电阻反演/viz/40.png](https://www.science42.tech/cases/caseMarkdown/地质勘探中的三维电阻反演/viz/40.png)

## 创新点

- 结构耦合约束：不满足于简单的平滑约束，引入梯度支撑、聚焦反演等技术，使模型边界更清晰、更符合地质实际。
- 用神经网络训练正演模拟器，替代耗时的数值计算，实现实时或近实时反演。
- 从传统的直流电阻率扩展到复电阻率/激发极化法（CR/IP）三维反演，同时获取电阻率和充电率参数，用于区分矿物类型和评估岩石颗粒特性。



## 1、控制方程
稳态电流场方程(泊松方程)：
$$
\nabla \cdot (\sigma(\mathbf{r}) \nabla \phi(\mathbf{r})) = -I \delta(\mathbf{r} - \mathbf{r}_A) + I \delta(\mathbf{r} - \mathbf{r}_B)
$$
$$
s(\mathbf r)=I\,[g(\mathbf r,\mathbf r_A)-g(\mathbf r,\mathbf r_B)]
$$
其中：

$$
\sigma(\mathbf{r}):电导率 
$$
$$
 \phi(\mathbf{r}):电位 
$$
$$
I \delta(\mathbf{r} - \mathbf{r}_B)：狄拉克δ函数，表示点源位于 r_B
$$


### 总损失函数
$$
L = L_{\text{pde}} + L_{\text{data}}
$$

### PDE损失（均方残差）
$$
 L_{\text{pde}} = \frac{1}{N_f} \sum_{i=1}^{N_f} [ \nabla \cdot ( \sigma(r_i) \nabla \phi(r_i) ) + s(r_i) ]^2
$$

### 数据损失（电位差拟合）
$$
L_{\text{data}} =\frac{1}{N_d}  \sum_{i=1}^{N_f} \left[\phi(\mathbf r_{M_m})-\phi(\mathbf r_{N_m})-d^{\text{obs}}\right]^2
$$

其中 $d_{\text{obs}}$ 是观测电位差。


### 电位梯度
$$
\nabla\phi(\mathbf{r}) = \left( \frac{\partial\phi}{\partial x}, \frac{\partial\phi}{\partial y}, \frac{\partial\phi}{\partial z} \right)
$$

### 散度计算
$$
\nabla \cdot (\sigma\nabla\phi) = \frac{\partial}{\partial x}\left(\sigma\frac{\partial\phi}{\partial x}\right) + \frac{\partial}{\partial y}\left(\sigma\frac{\partial\phi}{\partial y}\right) + \frac{\partial}{\partial z}\left(\sigma\frac{\partial\phi}{\partial z}\right)
$$



## 2、参数、属性定义

### PINN电阻率反演参数属性表：

| 符号 | 含义 | 类型 | 单位 / 说明 |
|------|------|------|-------------|
| φ(r) | 电位分布 | 实标量场 | 单位：伏特(V) |
| ρ(r) | 电阻率分布 | 实标量场 | 单位：欧姆·米(Ω·m) |
| logρ(r) | 电阻率对数 | 实标量场 | log_e(ρ)，确保正定性 |
| σ(r) | 电导率分布 | 实标量场 | σ = 1/ρ，单位：西门子/米(S/m) |
| r | 空间位置向量 | 向量 | 2D: (x,z)，3D: (x,y,z) |
| x, z | 二维坐标 | 标量 | 2D笛卡尔坐标，单位：米(m) |
| x, y, z | 三维坐标 | 标量 | 3D笛卡尔坐标，单位：米(m) |
| A, B | 电流源电极位置 | 向量 | A:正电流源，B:负电流源 |
| M, N | 测量电位电极位置 | 向量 | M、N间电位差为观测数据 |
| ∇· | 散度算子 | 微分算子 | ∂/∂x + ∂/∂y + ∂/∂z |
| ∇ | 梯度算子 | 微分算子 | (∂/∂x, ∂/∂y, ∂/∂z) |
| I | 注入电流强度 | 标量 | 代码中隐含为1，单位：安培(A) |
| g(r,r_s) | 高斯源函数 | 实标量场 | 近似狄拉克δ函数，ε=0.5 |
| ε | 高斯源宽度参数 | 标量 | ε=0.5，控制源的空间展宽 |
| s(r) | 总源项分布 | 实标量场 | s(r)=g(r,r_A)-g(r,r_B) |
| d_obs | 观测电位差 | 标量 | φ_M - φ_N，单位：伏特(V) |
| φ_synth(r) | 合成正演电位 | 实标量场 | 用于生成模拟观测数据 |
| θ_φ | 电位网络参数 | 张量 | MLP_φ的权重和偏置 |
| θ_ρ | 电阻率网络参数 | 张量 | MLP_ρ的权重和偏置 |
| MLP_φ | 电位神经网络 | 网络结构 | 输入:坐标，输出:电位值 |
| MLP_ρ | 电阻率神经网络 | 网络结构 | 输入:坐标，输出:logρ |
| w | 网络宽度 | 整数 | 隐藏层神经元数，w=64 |
| d | 网络深度 | 整数 | 隐藏层层数，d=5 |
| tanh | 激活函数 | 函数 | 双曲正切激活函数 |
| N_f | 域内采样点数 | 整数 | PDE残差计算点数，2D:6000，3D:8000 |
| R(r) | PDE残差函数 | 实标量场 | R=∇·(σ∇φ)+s |
| L_pde | PDE损失 | 标量 | L_pde=mean(R²) |
| L_data | 数据损失 | 标量 | L_data=[(φ_M-φ_N)-d_obs]² |
| L_total | 总损失 | 标量 | L_total = L_pde + L_data |
| η | 学习率 | 超参数 | η=1e-3，Adam优化器参数 |
| epoch | 训练轮数 | 整数 | 2D:500轮，3D:6000轮 |
| Ω | 计算区域 | 几何域 | 2D: 0≤x≤50m, 0≤z≤30m<br>3D: 0≤x≤20m, 0≤y≤20m, 0≤z≤10m |
| ∇_θL | 损失梯度 | 张量 | 损失对网络参数的梯度 |
| Adam | 优化器 | 算法 | 自适应矩估计优化器 |
| device | 计算设备 | 硬件 | "cuda"或"cpu" |
| DATA_DIR | 数据目录 | 字符串 | 存储观测数据的目录 |
| MODEL_DIR | 模型目录 | 字符串 | 存储训练模型的目录 |
| RESULT_DIR | 结果目录 | 字符串 | 存储可视化结果的目录 |

### 软件版本要求：

建议Python 3.8以上


### 所需要导入的库：

```  
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

```

## 3、计算方法及代码实现

### 3.1 定义  pinn_2d
```python
def pinn_2d_abmn():

    print("Running 2D ABMN PINN inversion")

    # ---------------- Domain ----------------
    Nf = 6000
    x = torch.rand(Nf, 2, device=device)
    x[:, 0] *= 50.0
    x[:, 1] *= 30.0
    x.requires_grad_(True)

    # ---------------- Electrodes ----------------
    A = torch.tensor([10., 0.], device=device)
    B = torch.tensor([40., 0.], device=device)
    M = torch.tensor([22., 0.], device=device)
    N = torch.tensor([28., 0.], device=device)

    # ---------------- Synthetic forward φ ----------------
    def phi_forward(x):
        return torch.sin(np.pi*x[:,0]/50) * torch.exp(-x[:,1]/15)

    with torch.no_grad():
        d_obs = phi_forward(M.unsqueeze(0)) - phi_forward(N.unsqueeze(0))

    # Save data
    np.savez(
        f"{DATA_DIR}/abmn_2d_obs.npz",
        A=A.cpu().numpy(), B=B.cpu().numpy(),
        M=M.cpu().numpy(), N=N.cpu().numpy(),
        d_obs=d_obs.cpu().numpy()
    )

y scipy matplotlib plotly
```

### 3.2 训练及初始数据

```python
for ep in range(500):
        opt.zero_grad()

        phi = net_phi(x)
        rho = torch.exp(net_logrho(x))
        sigma = 1.0 / rho

        grad_phi = torch.autograd.grad(
            phi, x, torch.ones_like(phi), create_graph=True
        )[0]

        div = (
            torch.autograd.grad(sigma*grad_phi[:,0:1], x,
                                torch.ones_like(phi), create_graph=True)[0][:,0:1]
            +
            torch.autograd.grad(sigma*grad_phi[:,1:2], x,
                                torch.ones_like(phi), create_graph=True)[0][:,1:2]
        )

        src = gaussian_source(x, A) - gaussian_source(x, B)
        loss_pde = torch.mean((div + src)**2)

        loss_data = (net_phi(M.unsqueeze(0)) -
                     net_phi(N.unsqueeze(0)) - d_obs)**2

        loss = loss_data + loss_pde
        loss.backward()
        opt.step()

        if ep % 500 == 0:
            print(f"2D Epoch {ep:4d} | Data {loss_data.item():.2e} | PDE {loss_pde.item():.2e}")
```
### 3.3 2D可视化输出

```python

X,Z = np.meshgrid(np.linspace(0,50,120), np.linspace(0,30,80))
    pts = torch.tensor(np.c_[X.ravel(), Z.ravel()],
                       dtype=torch.float32, device=device)

    rho_inv = torch.exp(net_logrho(pts)).detach().cpu().numpy().reshape(Z.shape)

    plt.figure(figsize=(8,4))
    plt.imshow(rho_inv, extent=[0,50,30,0], cmap="jet")
    plt.scatter([10,40,22,28],[0,0,0,0], c="k")
    plt.text(10,1,"A"); plt.text(40,1,"B")
    plt.text(22,1,"M"); plt.text(28,1,"N")
    plt.colorbar(label="ρ (Ω·m)")
    plt.title("2D ABMN PINN Inverted Resistivity")
    plt.savefig(f"{RESULT_DIR}/rho_2d.png", dpi=200)
    plt.close()

```

### 3.4 定义  pinn_3d及loss计算

```python
def pinn_3d_abmn():

    print("Running 3D ABMN PINN inversion")

    # ---------------- Domain ----------------
    Nf = 8000
    x = torch.rand(Nf, 3, device=device)
    x[:, 0] *= 20.0
    x[:, 1] *= 20.0
    x[:, 2] *= 10.0
    x.requires_grad_(True)

    # ---------------- Electrodes ----------------
    A = torch.tensor([5.,10.,0.], device=device)
    B = torch.tensor([15.,10.,0.], device=device)
    M = torch.tensor([9.,10.,0.], device=device)
    N = torch.tensor([11.,10.,0.], device=device)

    def phi_forward(x):
        return (
            torch.sin(np.pi*x[:,0]/20) *
            torch.sin(np.pi*x[:,1]/20) *
            torch.exp(-x[:,2]/5)
        )

    with torch.no_grad():
        d_obs = phi_forward(M.unsqueeze(0)) - phi_forward(N.unsqueeze(0))

    np.savez(
        f"{DATA_DIR}/abmn_3d_obs.npz",
        A=A.cpu().numpy(), B=B.cpu().numpy(),
        M=M.cpu().numpy(), N=N.cpu().numpy(),
        d_obs=d_obs.cpu().numpy()
    )

    net_phi = MLP(3,1).to(device)
    net_logrho = MLP(3,1).to(device)

    opt = torch.optim.Adam(
        list(net_phi.parameters()) + list(net_logrho.parameters()), lr=1e-3
    )

    for ep in range(6000):
        opt.zero_grad()

        phi = net_phi(x)
        rho = torch.exp(net_logrho(x))
        sigma = 1.0 / rho

        grad_phi = torch.autograd.grad(
            phi, x, torch.ones_like(phi), create_graph=True
        )[0]

        div = sum(
            torch.autograd.grad(
                sigma*grad_phi[:,i:i+1], x,
                torch.ones_like(phi), create_graph=True
            )[0][:,i:i+1] for i in range(3)
        )

        src = gaussian_source(x, A) - gaussian_source(x, B)
        loss_pde = torch.mean((div + src)**2)

        loss_data = (net_phi(M.unsqueeze(0)) -
                     net_phi(N.unsqueeze(0)) - d_obs)**2

        loss = loss_data + loss_pde
        loss.backward()
        opt.step()

        if ep % 1000 == 0:
            print(f"3D Epoch {ep:4d} | Data {loss_data.item():.2e} | PDE {loss_pde.item():.2e}")

    torch.save(net_phi.state_dict(), f"{MODEL_DIR}/pinn_3d_phi.pth")
    torch.save(net_logrho.state_dict(), f"{MODEL_DIR}/pinn_3d_rho.pth")
```

### 3.5 3D可视化输出
```python
 n (z = mid slice)
    X,Y = np.meshgrid(np.linspace(0,20,60), np.linspace(0,20,60))
    Z = np.ones_like(X) * 5.0

    pts = torch.tensor(
        np.c_[X.ravel(),Y.ravel(),Z.ravel()],
        dtype=torch.float32, device=device
    )

    rho_inv = torch.exp(net_logrho(pts)).detach().cpu().numpy().reshape(X.shape)


    plt.figure(figsize=(5,4))
    plt.imshow(rho_inv, cmap="jet", extent=[0,20,20,0])
    plt.colorbar(label="ρ (Ω·m)")
    plt.title("3D ABMN PINN Inversion (z=5m slice)", pad=15)
    plt.xticks(np.arange(0, 21, 5))
    plt.yticks(np.arange(0, 21, 5))
    plt.tight_layout()  # 自动调整子图参数，以给定的填充适应图形区域
    plt.savefig(f"{RESULT_DIR}/rho_3d_slice.png", dpi=200)
    plt.close()
```




## 4、 预测、绘图及成果展示

这是一个典型的通过稳态电流场方程(泊松方程)来分析2D、3D地质勘探中的电阻率问题。


![caseMarkdown/地质勘探中的三维电阻反演/viz/rho_2d.png](https://www.science42.tech/cases/caseMarkdown/地质勘探中的三维电阻反演/viz/rho_2d.png)




![caseMarkdown/地质勘探中的三维电阻反演/viz/rho_3d_slice.png](https://www.science42.tech/cases/caseMarkdown/地质勘探中的三维电阻反演/viz/rho_3d_slice.png)

横坐标表示地表选取长度，左侧纵坐标表示纵深高度，右侧纵坐标表示电阻率。


## 5、常见问题（FAQ）

- 收敛性问题：损失函数停滞于局部极小值，模型更新停滞。

- 精度问题：地质体边界不清晰，产生虚假异常。

- 计算效率问题：训练好的模型对新数据泛化能力差。

## 6、工业价值

- 提升勘探精度与效率，直接驱动资源发现与储量升级：解决了传统方法“看不清、探不准、算不透”的核心痛点。在矿产资源勘探中，该技术能高精度勾勒出陡倾角矿体、深部隐伏矿体的三维形态和空间展布，有效区分矿化带与围岩的界限。例如，通过其精细刻画，一个原本在二维剖面上看似孤立的矿体，可能被证实为大型网状矿脉的一部分，从而将“点状”资源升级为“体状”资源，显著增加可采储量。

- 实现动态监测与四维表征，赋能资源开发与工程安全的全周期管理：在重大工程（如水坝、核电站、高铁路基）的建设和运营期，该技术如同埋设于地下的“透视神经网络”，能持续监测坝体渗漏、地基稳定性及污染物迁移，实现从“事后补救”到“事前预警”的根本性转变。这一层的价值不仅在于保障安全，更在于优化生产、延长资产寿命，其创造的间接经济效益和社会效益难以估量。

- 引领勘探智能化与数据融合，催生行业新范式与数据资产化：这是三维电阻率反演面向未来的战略性价值。以物理信息神经网络（PINN） 为代表的人工智能与反演技术的深度融合，正将反演从耗时数小时的高性能计算任务转变为近实时的“一键式”智能服务。这种变革使得在野外现场进行实时三维成像和决策成为可能，极大地缩短了从数据采集到地质解释的周期，改变了传统的工作流程。

## 使用示例
#### 1. 进入秋月白页面，点击聊天框上方的**方程求解**图标
![caseMarkdown/地质勘探中的三维电阻反演/viz/222.jpg](https://www.science42.tech/cases/caseMarkdown/地质勘探中的三维电阻反演/viz/222.jpg)
#### 2、在对话框输入：地质勘探中的三维电阻反演
![caseMarkdown/地质勘探中的三维电阻反演/viz/888.jpg](https://www.science42.tech/cases/caseMarkdown/地质勘探中的三维电阻反演/viz/888.jpg)

#### 3、模型将先进行分析与规划，随后自动给出完整的示例
![caseMarkdown/地质勘探中的三维电阻反演/viz/999.jpg](https://www.science42.tech/cases/caseMarkdown/地质勘探中的三维电阻反演/viz/999.jpg)

#### 4、运行代码并训练，得到图像
![caseMarkdown/地质勘探中的三维电阻反演/viz/rho_2d.png](https://www.science42.tech/cases/caseMarkdown/地质勘探中的三维电阻反演/viz/rho_2d.png)







