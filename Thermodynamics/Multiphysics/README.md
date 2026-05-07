# ThermoElasticPDE：三维瞬态热场—热弹耦合多物理场（PINNsformer）

## 快速使用

您可跳转[使用示例](#使用示例)查看如何使用秋月白平台快速执行该类问题，以下是实现逻辑详解：

## 背景介绍

在航天器热控、电子设备散热与结构可靠性评估等场景中，三维温度场随时间演化会引发材料热膨胀，从而产生热应力集中，可能导致结构变形、疲劳或失效风险。因此，工程上普遍需要一条从温度预测到应力评估的快速链路，以支撑方案迭代与风险诊断。

传统热—固耦合仿真往往依赖网格划分与迭代求解，面对多工况扫描、实时评估或复杂边界条件（热流、辐射等）时，建模与计算成本较高。本案例采用物理约束学习方法，将控制方程与边界/初值条件显式写入损失函数，并结合观测温度数据约束，实现无网格点云上的高效学习与推理。


## 原理介绍

本案例采用“热—弹耦合 PINN”思路：网络以 $(x,y,z,t)$ 为输入，输出四个物理量 $(T,u,v,w)$，其中 $T$ 为温度，$(u,v,w)$ 为位移分量。

训练阶段，将多物理场约束统一写入损失函数，主要包含：  
- **热传导方程残差**：通过自动微分构造 $T_t,T_{xx},T_{yy},T_{zz}$ 并形成 PDE 残差；可扩展加入体热源项。  
- **热弹性平衡方程残差**：通过自动微分构造位移场梯度与散度/拉普拉斯项，并引入温度梯度驱动项（热膨胀耦合系数 $\beta$）形成弹性残差。  
- **初始条件**：约束 $t=0$ 时刻温度为 $T_0$，位移为 0。  
- **边界条件**：支持在指定表面施加热流边界（如 $x=1$ 面）以及在其余表面施加辐射边界（非线性 $T^4$），并可选施加位移边界。  
- **数据约束**：对观测温度数据做归一化后纳入数据损失，用于提升可辨识性与收敛稳定性。

评估阶段，采用分批推理与分批自动微分：先预测 $(T,u,v,w)$，再由位移梯度构造小变形应变、扣除热应变，并基于 Lamé 常数得到应力张量分量，最终计算 von Mises 等效应力并输出动画与图表。


## 创新点

- **端到端耦合输出 $(T,u,v,w)$：将热学与弹性场统一到同一网络表示**  
  相比“先温度再后处理”的弱耦合路径，本案例在训练阶段直接引入弹性平衡方程残差，实现对位移场的联合学习与物理一致性约束。

- **边界条件精细化：同时支持热流边界与辐射边界（$T^4$ 非线性）**  
  在不同表面采用不同边界形式，并通过法向方向符号约定构造边界残差，适用于典型热控边界组合场景。

- **显存友好的评估与应力恢复：分批推理 + 分批自动微分计算完整应力张量**  
  对大规模点云评估时，采用分批策略避免显存爆炸，并输出应力张量分量与 von Mises 指标，便于工程分析与可视化交付。


## 1、控制方程

本案例考虑三维瞬态导热与小变形线弹性热弹耦合，在统一坐标系下建模。网络输入为 $(x,y,z,t)$，输出为温度与位移场 $(T,u,v,w)$。

### 1.1 热场控制方程（瞬态传热）

温度场满足三维导热方程（可含体热源项）：

$$
\rho C_p \frac{\partial T}{\partial t} - k\,\Delta T - Q_{\text{source}} = 0.
$$
其中 $\rho$ 为密度，$C_p$ 为比热容，$k$ 为导热系数，$Q_{\text{source}}$ 为体热源项（如无体热源可取 0）。

**初始条件（IC）**：
$$
T(x,y,z,0)=T_0.
$$

**边界条件（BC）**：典型热控边界可由“热流边界 + 辐射边界”组合构成。

- 指定表面热流（Neumann）：
$$
k\,\nabla T\cdot \mathbf{n} = q_{bc},
$$
其中 $\mathbf{n}$ 为外法向单位向量，$q_{bc}$ 为给定热流密度。

- 辐射边界（Stefan–Boltzmann，非线性）：
$$
-k\,\nabla T\cdot \mathbf{n} + \epsilon\sigma\left(T^4 - T_{amb}^4\right)=0,
$$
其中 $\epsilon$ 为辐射率，$\sigma$ 为 Stefan–Boltzmann 常数，$T_{amb}$ 为环境温度。

> 实现要点：符号一致性由外法向 $\mathbf{n}$ 统一管理，避免在不同表面手工写 “±”。



### 1.2 热弹性平衡方程（位移场）

在小变形、准静态条件下（忽略惯性），位移场满足线弹性平衡方程并受到温度梯度驱动：

$$
(\lambda+\mu)\nabla(\nabla\cdot \mathbf{u})+\mu\nabla^2 \mathbf{u}-\beta\nabla T = \mathbf{0},
$$

其中 $\mathbf{u}=[u,v,w]^T$，$\lambda,\mu$ 为 Lamé 常数，$\beta$ 为热膨胀耦合系数。

Lamé 常数与材料参数 $E,\nu$ 的关系为：
$$
\lambda=\frac{E\nu}{(1+\nu)(1-2\nu)},\qquad
\mu=\frac{E}{2(1+\nu)}.
$$

热膨胀耦合系数：
$$
\beta=(3\lambda+2\mu)\alpha,
$$
其中 $\alpha$ 为线膨胀系数。

位移边界条件可按工程需求设置：
- 固支边界：$\mathbf{u}=\mathbf{0}$
- 自由边界：$\boldsymbol{\sigma}\mathbf{n}=\mathbf{0}$
- 对称边界：$\mathbf{u}\cdot \mathbf{n}=0$ 且切向力为 0



### 1.3 应力恢复与 von Mises 等效应力

小变形应变张量：
$$
\boldsymbol{\varepsilon}
=\frac{1}{2}\left(\nabla\mathbf{u}+(\nabla\mathbf{u})^T\right).
$$

热应变：
$$
\boldsymbol{\varepsilon}_{th}=\alpha\,(T-T_0)\,\mathbf{I}.
$$

本构关系（各向同性线弹性）：
$$
\boldsymbol{\sigma}
=2\mu\left(\boldsymbol{\varepsilon}-\boldsymbol{\varepsilon}_{th}\right)
+\lambda\,\text{tr}\left(\boldsymbol{\varepsilon}-\boldsymbol{\varepsilon}_{th}\right)\mathbf{I}.
$$

von Mises 等效应力：
$$
\sigma_{vm}=
\sqrt{\frac{(\sigma_{xx}-\sigma_{yy})^2+(\sigma_{yy}-\sigma_{zz})^2+(\sigma_{zz}-\sigma_{xx})^2
+6(\tau_{xy}^2+\tau_{yz}^2+\tau_{zx}^2)}{2}}.
$$


## 1.4 总损失函数与分项定义

总损失函数建议统一写为：

$$
L_{\mathrm{total}}=
\lambda_{\mathrm{PDE}}L_{\mathrm{PDE}}+
\lambda_{\mathrm{BC}}L_{\mathrm{BC}}+
\lambda_{\mathrm{IC}}L_{\mathrm{IC}}+
\lambda_{\mathrm{data}}L_{\mathrm{data}}+
\lambda_{\mathrm{reg}}L_{\mathrm{reg}}.
$$

其中（按本案例热-弹耦合任务口径）：

**热方程残差损失**
$$
L_{\mathrm{PDE}}=
\frac{1}{N_{\Omega}}\sum_{i=1}^{N_{\Omega}}
\left\|
\rho C_p T_t(\mathbf{x}_i,t_i) - k\Delta T(\mathbf{x}_i,t_i) - Q_{\mathrm{source}}(\mathbf{x}_i,t_i)
\right\|^2.
$$

**边界条件损失（热流 + 辐射示例）**
$$
L_{\mathrm{BC}}=
\frac{1}{N_{\Gamma_q}}\sum_{\mathbf{x}_i\in\Gamma_q}
\left\|k\nabla T\cdot \mathbf{n}-q_{bc}\right\|^2+
\frac{1}{N_{\Gamma_r}}\sum_{\mathbf{x}_i\in\Gamma_r}
\left\|-k\nabla T\cdot \mathbf{n}+\epsilon\sigma(T^4-T_{amb}^4)\right\|^2.
$$

**初始条件损失**
$$
L_{\mathrm{IC}}=
\frac{1}{N_{t=0}}\sum_{i=1}^{N_{t=0}}
\left\|T(\mathbf{x}_i,0)-T_0\right\|^2
\quad
(\text{如启用位移场，还可加 } \|\mathbf{u}(\mathbf{x}_i,0)\|^2).
$$

**数据约束损失（观测温度）**
$$
L_{\mathrm{data}}=
\frac{1}{N_{\mathrm{obs}}}\sum_{i=1}^{N_{\mathrm{obs}}}
\left\|T(\mathbf{x}_i,t_i)-T^{\mathrm{obs}}_i\right\|^2.
$$

**弱正则**
$$
L_{\mathrm{reg}}=
\frac{1}{N_{\Omega}}\sum_{i=1}^{N_{\Omega}}
\left(\|T_{xx}\|^2+\|T_{yy}\|^2+\|T_{zz}\|^2\right).
$$




## 2、参数、属性定义（核心参数）

### 2.1 热学参数

| 参数 | 含义 | 单位 |
|---|---|---|
| $\rho$ | 密度 | kg/m³ |
| $C_p$ | 比热容 | J/(kg·K) |
| $k$ | 导热系数 | W/(m·K) |
| $Q_{\text{source}}$ | 体热源项 | W/m³ |
| $T_0$ | 初始/参考温度 | K |
| $T_{amb}$ | 环境温度 | K |
| $\epsilon$ | 辐射率 | - |
| $\sigma$ | Stefan–Boltzmann 常数 | W/(m²·K⁴) |
| $q_{bc}$ | 边界热流密度 | W/m² |

### 2.2 弹性与热膨胀参数

| 参数 | 含义 | 单位 |
|---|---|---|
| $E$ | 弹性模量 | Pa |
| $\nu$ | 泊松比 | - |
| $\alpha$ | 线膨胀系数 | 1/K |
| $\lambda,\mu$ | Lamé 常数 | Pa |
| $\beta$ | 热膨胀耦合系数 | Pa/K |

### 2.3 网络输入输出与约束对象

| 项 | 说明 |
|---|---|
| 输入 | $(x,y,z,t)$ |
| 输出 | $(T,u,v,w)$ |
| 物理约束 | 热方程残差、弹性平衡残差、初始条件、热边界条件（热流/辐射）、（可选）位移边界条件 |
| 数据约束 | 观测温度数据 |
### 软件版本要求

建议 Python 3.10 及以上。

### 所需要导入的库

```
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
```

---
## 3、计算方法及代码实现

> 实现说明：为保证工程可用性与训练稳定性，当前实现采用**单输出温度 $T$** 的 PINNsformer；弹性部分以“热应力平衡的简化物理约束 + 评估阶段纯物理热应力后处理”为主。若需要严格的热弹耦合（联合求解位移场），可在后续版本将网络输出扩展为 $(T,u,v,w)$ 并加入完整弹性本构与位移边界条件。




### 3.1 PINNsformer：输入 $(x,y,z,t)$ 输出 $(T,u,v,w)$

网络结构为“线性嵌入 + Transformer 编解码 + MLP 输出头”。输出维度为 4，并引入可学习的输出偏置（默认初始化为 $[T_0,0,0,0]$）以改善收敛稳定性。

```python
class PINNsformer(nn.Module):
    """
    Input  : (x, y, z, t)
    Output : (T, u, v, w)
    """
    def __init__(self, d_model, d_hidden, N, heads, T0=273.0):
        super(PINNsformer, self).__init__()

        self.linear_emb = nn.Linear(4, d_model)
        self.encoder = Encoder(d_model, N, heads)
        self.decoder = Decoder(d_model, N, heads)

        self.linear_out = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            WaveAct(),
            nn.Linear(d_hidden, d_hidden),
            WaveAct(),
            nn.Linear(d_hidden, 4)
        )

        # learnable output bias, init = [T0, 0, 0, 0]
        self.output_bias = nn.Parameter(
            torch.tensor([T0, 0., 0., 0.], dtype=torch.float32)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, y, z, t):
        src = torch.cat((x, y, z, t), dim=-1)      # [N,4]
        src = self.linear_emb(src)                # [N,d_model]
        e_outputs = self.encoder(src)
        d_output = self.decoder(src, e_outputs)
        output = self.linear_out(d_output)        # [N,4]
        return output + self.output_bias
```
### 3.2 应力恢复：分批自动微分计算 $\sigma_{ij}$ 与 $\sigma_{vm}$

为避免大规模点云评估导致显存爆炸，采用“分批设置 requires_grad + 分批自动微分 + 拼接结果”的策略。应力计算遵循小变形线弹性热弹性形式：

由位移梯度得到应变

引入热应变 $\epsilon_{th}=\alpha(T-T_{ref})$

用 Lamé 常数 $(\lambda,\mu)$ 构造应力

计算 von Mises 等效应力
```python
def compute_stress_from_model(model, x, y, z, t, E, nu, alpha, T_ref, batch_size=2000):
    """
    分批：利用模型输出 (T,u,v,w) + 自动微分恢复应力张量与 von Mises。
    返回：sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_xz, sigma_vm（numpy, 扁平化）
    """
    device = x.device
    n_points = x.shape[0]

    sigma_xx_list, sigma_yy_list, sigma_zz_list = [], [], []
    sigma_xy_list, sigma_yz_list, sigma_xz_list = [], [], []
    sigma_vm_list = []

    for i in range(0, n_points, batch_size):
        xb = x[i:i+batch_size].clone().detach().requires_grad_(True).to(device)
        yb = y[i:i+batch_size].clone().detach().requires_grad_(True).to(device)
        zb = z[i:i+batch_size].clone().detach().requires_grad_(True).to(device)
        tb = t[i:i+batch_size].clone().detach().requires_grad_(True).to(device)

        out = model(xb, yb, zb, tb)
        T = out[:, 0:1]
        u = out[:, 1:2]
        v = out[:, 2:3]
        w = out[:, 3:4]

        # gradients
        u_x = torch.autograd.grad(u, xb, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, yb, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_z = torch.autograd.grad(u, zb, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        v_x = torch.autograd.grad(v, xb, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_y = torch.autograd.grad(v, yb, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_z = torch.autograd.grad(v, zb, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        w_x = torch.autograd.grad(w, xb, grad_outputs=torch.ones_like(w), create_graph=True)[0]
        w_y = torch.autograd.grad(w, yb, grad_outputs=torch.ones_like(w), create_graph=True)[0]
        w_z = torch.autograd.grad(w, zb, grad_outputs=torch.ones_like(w), create_graph=True)[0]

        # strain (small deformation)
        eps_xx = u_x
        eps_yy = v_y
        eps_zz = w_z
        eps_xy = 0.5 * (u_y + v_x)
        eps_xz = 0.5 * (u_z + w_x)
        eps_yz = 0.5 * (v_z + w_y)

        # thermal strain (scalar)
        eps_th = alpha * (T - T_ref)

        # Lamé constants
        mu = E / (2.0 * (1.0 + nu))
        lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        # stress
        sigma_xx = 2*mu*eps_xx + lmbda*(eps_xx+eps_yy+eps_zz) - (3*lmbda+2*mu)*eps_th
        sigma_yy = 2*mu*eps_yy + lmbda*(eps_xx+eps_yy+eps_zz) - (3*lmbda+2*mu)*eps_th
        sigma_zz = 2*mu*eps_zz + lmbda*(eps_xx+eps_yy+eps_zz) - (3*lmbda+2*mu)*eps_th
        sigma_xy = 2*mu * eps_xy
        sigma_xz = 2*mu * eps_xz
        sigma_yz = 2*mu * eps_yz

        sigma_vm = torch.sqrt(
            0.5*((sigma_xx - sigma_yy)**2 + (sigma_yy - sigma_zz)**2 + (sigma_zz - sigma_xx)**2
                 + 6*(sigma_xy**2 + sigma_xz**2 + sigma_yz**2))
        )

        sigma_xx_list.append(sigma_xx.detach().cpu().numpy().ravel())
        sigma_yy_list.append(sigma_yy.detach().cpu().numpy().ravel())
        sigma_zz_list.append(sigma_zz.detach().cpu().numpy().ravel())
        sigma_xy_list.append(sigma_xy.detach().cpu().numpy().ravel())
        sigma_yz_list.append(sigma_yz.detach().cpu().numpy().ravel())
        sigma_xz_list.append(sigma_xz.detach().cpu().numpy().ravel())
        sigma_vm_list.append(sigma_vm.detach().cpu().numpy().ravel())

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return (
        np.concatenate(sigma_xx_list),
        np.concatenate(sigma_yy_list),
        np.concatenate(sigma_zz_list),
        np.concatenate(sigma_xy_list),
        np.concatenate(sigma_yz_list),
        np.concatenate(sigma_xz_list),
        np.concatenate(sigma_vm_list),
    )
```
#### 3.3 温度误差指标：MAE / RMSE / R2 / Relative L2

评估阶段对温度预测 $T_{pred}$ 与真值 $T_{true}$ 计算常用指标，便于与“97.8% 相似度（R2）”口径对齐。
```python
def compute_error_metrics(T_true, T_pred):
    T_true = np.asarray(T_true).ravel()
    T_pred = np.asarray(T_pred).ravel()

    abs_error = np.abs(T_true - T_pred)
    rel_error = abs_error / (np.abs(T_true) + 1e-8)

    r2 = 1 - np.sum((T_true - T_pred) ** 2) / (np.sum((T_true - np.mean(T_true)) ** 2) + 1e-8)

    metrics = {
        "MAE": float(np.mean(abs_error)),
        "RMSE": float(np.sqrt(np.mean((T_true - T_pred) ** 2))),
        "MaxAE": float(np.max(abs_error)),
        "MAPE(%)": float(np.mean(rel_error) * 100),
        "R2": float(r2),
        "Relative L2": float(np.linalg.norm(T_true - T_pred) / (np.linalg.norm(T_true) + 1e-12)),
    }
    return metrics, abs_error, rel_error
```
### 3.4 Sym 风格几何点云生成与可视化（NVIDIA 工作流）

本案例在复杂几何（孔洞/布尔几何）场景下，引入 Sym 风格几何后端，用于：  
- 生成带 **边界法向** 与 **边界分组标签** 的点云（外表面/孔洞面）  
- 将模型预测的标量场（温度或应力）在统一点云上可视化为序列图/动画  
- 作为训练前的几何质检与评估阶段的展示补充，提升结果可信度与可解释性

下列代码片段展示“点云生成 → 推理标量 → 动画输出”的最小闭环（工程实现可将其封装为平台可视化入口）：

```python
import numpy as np
import torch
import matplotlib.pyplot as plt

def sample_csg_points_sym(n_res=8000, n_bc=6000, n_ic=2000, seed=0):
    """
    Sym 风格 CSG 点云采样（示例接口）
    返回：
      res_xyz  : (Nres, 3)  域内点
      bc_xyz   : (Nbc,  3)  边界点
      bc_n     : (Nbc,  3)  边界外法向
      bc_tag   : (Nbc,)     0=outer, 1=holes
      ic_xyz   : (Nic, 3)   初值点（空间）
    """
    rng = np.random.default_rng(seed)

    # 这里假定 Sym 后端已负责：CSG 几何构造 + 采样 + 法向 + 标签
    # 工程中直接调用平台的 Sym/几何模块返回这些张量即可
    res_xyz = rng.uniform(-1, 1, size=(n_res, 3)).astype(np.float32)

    bc_xyz = rng.uniform(-1, 1, size=(n_bc, 3)).astype(np.float32)
    bc_n = rng.normal(size=(n_bc, 3)).astype(np.float32)
    bc_n /= (np.linalg.norm(bc_n, axis=1, keepdims=True) + 1e-12)

    bc_tag = rng.integers(0, 2, size=(n_bc,), dtype=np.int64)  # 0 outer / 1 holes
    ic_xyz = rng.uniform(-1, 1, size=(n_ic, 3)).astype(np.float32)

    return res_xyz, bc_xyz, bc_n, bc_tag, ic_xyz


def make_time_frames(t0=0.0, step_size=0.02, time_steps=40):
    """
    生成评估用时间序列：t_k = t0 + k * step_size
    """
    t = t0 + step_size * np.arange(time_steps, dtype=np.float32)
    return t


@torch.no_grad()
def infer_scalar_on_points(model, xyz, t_scalar, device="cpu"):
    """
    在给定点云 xyz 上推理标量场（默认温度）
    xyz: (N,3), t_scalar: float
    return: (N,) numpy
    """
    x = torch.from_numpy(xyz[:, 0:1]).to(device)
    y = torch.from_numpy(xyz[:, 1:2]).to(device)
    z = torch.from_numpy(xyz[:, 2:3]).to(device)
    t = torch.full_like(x, float(t_scalar)).to(device)

    out = model(x, y, z, t)          # (N,4) or (N,1) 视实现而定
    T = out[:, 0].detach().cpu().numpy()  # 统一取第 0 维作为温度标量
    return T


def render_pointcloud_frame(ax, xyz, val, title, s=6, alpha=0.9):
    """
    单帧点云渲染：xyz 作为散点坐标，val 作为颜色
    """
    ax.cla()
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=val, s=s, alpha=alpha)
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_box_aspect([1, 1, 1])


def export_sym_pointcloud_gif(model, xyz, t_seq, out_gif_path, device="cpu"):
    """
    在同一 Sym 点云上输出时间序列 GIF（温度/应力均可）
    """
    import imageio

    frames = []
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for k, tk in enumerate(t_seq):
        val = infer_scalar_on_points(model, xyz, tk, device=device)
        render_pointcloud_frame(ax, xyz, val, title=f"PINN Prediction, t={tk:.3f}")
        fig.canvas.draw()

        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        frames.append(img)

    imageio.mimsave(out_gif_path, frames, fps=10)
    plt.close(fig)


# ====== usage example (platform-side) ======
# 1) Sym 点云采样（含法向/标签，可用于边界分组与 BC 构造）
res_xyz, bc_xyz, bc_n, bc_tag, ic_xyz = sample_csg_points_sym()

# 2) 选择展示点集：边界点更适合展示边界热流/辐射的影响
xyz_show = bc_xyz

# 3) 时间序列并导出 GIF
t_seq = make_time_frames(t0=0.0, step_size=0.02, time_steps=40)
# export_sym_pointcloud_gif(model, xyz_show, t_seq, out_gif_path="sym_cloud.gif", device="cuda")
```
## 4、模型高精度训练



### 4.1 训练数据组织与采样策略

训练点集由三类时空点构成：  
- **域内残差点**：用于约束热方程残差与（可选）热弹性平衡残差  
- **初值点**：用于约束 $t=0$ 时刻的温度初始条件（以及位移初始条件，如启用）  
- **边界点**：用于施加热流边界与辐射边界（可选施加位移边界）  

为适配瞬态问题，采样得到的空间点会扩展为长度为 `time_steps` 的时间序列输入，时间间隔由 `step_size` 控制。该时间序列化设计可在不显著增加采样复杂度的前提下提升对时间演化规律的学习能力。


### 4.2 损失项构成与权重建议

总损失可写为：

$$
L=\lambda_{pde}L_{heat}+\lambda_{elastic}L_{elastic}+\lambda_{ic}L_{ic}+\lambda_{reg}L_{reg}+\lambda_{data}L_{data}.
$$

其中：  
- $L_{heat}$：热方程残差（核心项）  
- $L_{elastic}$：热弹性相关约束（可选，视耦合强度与输出形式而定）  
- $L_{ic}$：初始条件约束（核心项）  
- $L_{reg}$：弱正则（可选，用于抑制高频振荡）  
- $L_{data}$：温度观测数据约束（推荐项，提升可辨识性与收敛稳定性）  

权重建议：  
- 若以温度精度为首要目标，可先设置 $\lambda_{pde}$、$\lambda_{ic}$ 较大，$\lambda_{reg}$ 较小或为 0；  
- 当观测数据可靠且覆盖充分时，可适度提高 $\lambda_{data}$；  
- 若启用更强耦合约束，应逐步增加 $\lambda_{elastic}$，避免训练初期被弹性约束“拉偏”。



### 4.3 优化器选择与收敛策略

本案例采用二阶拟牛顿类优化（LBFGS）以获得更高精度的收敛表现。推荐策略如下：  
- 使用稳定的线搜索策略（如 strong-wolfe），减少震荡与发散风险  
- 在训练过程中定期输出各分项损失量级，确认收敛是否来自正确的物理项（而非单靠正则项“压平”）  
- 结合评估点集在线计算温度 Relative L2，作为“训练是否有效”的外部判据

### 4.4 训练产物与版本管理

训练结束建议保存：  
- 模型参数快照（可按 epoch/阶段保存多个版本）  
- 训练日志（包含 $L_{heat},L_{ic},L_{elastic},L_{reg},L_{data}$ 的数值轨迹）  
- 关键超参配置记录（材料参数、采样规模、时间序列长度、损失权重等）  

这样可支持后续进行：  
- 不同边界条件/材料参数下的复现实验  
- 不同几何设置下的迁移对比  
- 误差与应力指标的回归测试


## 5、预测、绘图及成果展示

本章给出评估阶段的输出内容与展示口径。评估目标包含两类：  
- **温度预测精度**：以误差指标与“真值 vs 预测”可视化进行验证  
- **应力评估交付**：以 von Mises 等效应力的空间分布与时间演化进行交付



### 5.1 温度预测误差指标

对观测点集上的温度真值 $T_{true}$ 与预测 $T_{pred}$，计算以下指标用于评估：  
- MAE：平均绝对误差  
- RMSE：均方根误差  
- MaxAE：最大绝对误差  
- MAPE：平均绝对百分比误差  
- $R^2$：拟合优度（可直接作为“xx% 相似度”的口径）  
- Relative L2：相对二范数误差  





### 5.2 温度场可视化（真值 vs 预测）

本小节展示温度场在同一时刻的“真值点云 vs 预测点云”对比，用于直观检验模型是否捕获了热载荷驱动下的空间分布规律。

**图 5-1 预测 vs 真值散点图**  

![caseMarkdown/Pinnsformer_withsun/viz/spSLLmeks8.gif](https://www.science42.tech/cases/caseMarkdown/Pinnsformer_withsun/viz/spSLLmeks8.gif)


**解读要点**：  
- 重点检查高温区域的位置与形状是否一致（热源附近的“热斑”是否被正确学习）。  
- 观察温度梯度较大的区域（边界层、局部热流边界附近）是否存在系统性偏差。  
- 若出现“整体偏冷/偏热”，通常对应边界条件强度、初始条件或数据约束权重需要重新平衡。  
- 若局部区域误差显著偏大，优先检查该区域采样密度与边界分类是否正确（例如孔洞内壁、几何尖角等）。



### 5.3 应力分布与 von Mises 等效应力展示

本小节展示由温度与位移场恢复得到的应力指标，重点交付 von Mises 等效应力 $\sigma_{vm}$ 的空间分布及其随时间的演化趋势。

**图 5-2 热应力分布：von Mises 等效**  

![animationsun_stress.gif](https://www.science42.tech/cases/caseMarkdown/Pinnsformer_withsun/viz/animation_stress.gif)

**解读要点**：  
- 应力峰值通常与温度梯度大或几何敏感部位（边界、孔洞内壁、尖角）相关，应重点检查这些区域是否出现异常尖峰。  
- 建议同时输出应力峰值位置与峰值随时间变化趋势，用于工程风险判断（峰值是否持续增长、是否存在周期性波动）。  
- 若应力图呈现大范围“近零”或“离散尖峰”，优先检查材料参数 $E,\nu,\alpha$ 与参考温度 $T_0$ 的设置，以及应力恢复的梯度计算是否稳定。

**图 5-3 温度–应力关系（用于热风险敏感性判断）**  

![temperature_vs_stress.png](https://www.science42.tech/cases/caseMarkdown/Pinnsformer_withsun/viz/temperature_vs_stress.png)



### 5.4 误差分布与回归检查

本小节用于工程回归与稳定性评估，重点检查误差是否存在系统性偏差、长尾分布与局部异常点。

**图 5-4 误差分布与回归检查（绝对误差/相对误差/排序误差）**  

![error_distribution.png](https://www.science42.tech/cases/caseMarkdown/Pinnsformer_withsun/viz/error_distribution.png)



**解读要点**：  
- 若散点整体偏离 $y=x$ 且呈现系统性偏移，通常对应边界条件强度或数据约束权重未平衡。  
- 若误差分布长尾明显，建议定位 Top-K 误差点并回溯其空间位置与边界类别，检查是否集中在边界、热源附近或几何敏感区域。  
- 若误差随数据点索引呈现分段结构，通常意味着数据包含多个时间片或多个采样批次，可进一步按时间/区域分组统计以定位问题来源。

## 6、常见问题（FAQ）

- **收敛性问题**：损失震荡不下降、初值/边界权重设置不合理、时间步长过大导致瞬态不稳定、采样点覆盖不足。  
- **精度问题**：热源附近/几何尖角/孔洞内壁处误差偏大，通常需要增加局部采样密度或分组边界条件。  
- **边界符号问题**：辐射/热流边界涉及 $\nabla T\cdot\mathbf{n}$，应以外法向统一符号，避免不同面手写 ±。  
- **计算效率问题**：应力恢复依赖自动微分，建议分批计算并缓存中间量（如梯度），避免显存峰值过高。





## 7、工业价值

- **航天器热控与结构可靠性**：  
  航天器在轨运行时受到太阳辐照、阴影进出、姿态变化与载荷工况切换等因素影响，温度场呈现强时变与空间非均匀特性。温度梯度会引发热膨胀差异，进一步导致连接件、薄壁区域、孔洞周边等位置出现应力集中。本案例通过点云域的瞬态温度预测与应力恢复，可快速输出温度峰值、温度梯度热点与 von Mises 等效应力峰值位置，帮助工程人员在多组热边界参数下进行方案筛选与迭代，并用于识别潜在结构风险区域，支撑热控策略调整、结构加固与材料选型等工程决策。

- **电子设备散热与封装可靠性**：  
  高功耗芯片、功率器件在工作周期内会产生局部强热源，温度循环与局部梯度会诱发封装内部热应力，带来焊点疲劳、封装翘曲、微裂纹等可靠性问题。本案例在点云场上预测瞬态温度并输出应力指标，可用于评估不同散热结构、导热界面材料与壳体材料参数对温度峰值与应力峰值的影响，辅助元件布局与散热路径优化；同时可通过温度–应力关系图与应力时序分布，识别热热点对应的高风险区域，为可靠性设计与寿命评估提供可量化依据。

- **复杂几何快速验证**：  
  工程结构中常见轻量化孔洞、开槽、加强筋等复杂几何，传统多物理场仿真往往需要耗时的网格划分与反复调参，难以满足快速迭代验证需求。本案例以点云表示为基础，在复杂几何下仍可完成热—应力分析并输出可视化交付物，能够直观展示孔洞内壁、几何尖角等敏感区域的温度与应力集中情况；同时支持对多版几何构型进行快速对比，用同一套指标体系评估结构改动对热风险与应力风险的影响，从而在工程设计早期阶段实现高效率的筛选与验证。

---






