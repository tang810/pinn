# PINNsformer-3：三维瞬态热弹耦合参数反演
## 快速使用

您可跳转[使用示例](#使用示例)查看如何使用秋月白平台快速执行该类问题，以下是实现逻辑详解：

## 背景介绍

在航天器热控、电子设备散热与结构可靠性评估等场景中，三维温度场随时间演化会引发材料热膨胀，从而产生热应力集中，可能导致结构变形、疲劳或失效风险。然而，实际工程中往往缺乏精确的材料参数（如热导率、热源强度）或边界条件（如热流密度），需要通过观测温度数据反推这些未知参数，以支撑方案迭代与风险诊断。

传统参数反演往往依赖网格划分与迭代优化，面对多工况扫描、实时评估或复杂边界条件（热流、辐射等）时，建模与计算成本较高。本案例采用物理约束学习方法，将控制方程与观测温度数据显式写入损失函数，实现无网格点云上的高效参数反演。


## 原理介绍

本案例采用“热—弹耦合参数反演 PINN”思路：网络以 $(x,y,z,t)$ 为输入，输出温度场 T 和位移场 (u,v,w)，通过最小化观测数据与物理预测的误差，实现参数估计。未知参数（如热导率 k、体热源 Q 等）作为损失函数中的可训练参数，通过梯度下降优化得到。

训练阶段，将多物理场约束统一写入损失函数，主要包含：  
- **热传导方程残差**：通过自动微分构造 $T_t,T_{xx},T_{yy},T_{zz}$ 并形成 PDE 残差；参数由网络输出驱动。  
- **热弹性平衡方程残差**：通过自动微分构造位移场梯度与散度/拉普拉斯项，并引入温度梯度驱动项（热膨胀耦合系数 $\beta$）形成弹性残差。  
- **初始条件**：约束 $t=0$ 时刻温度为 $T_0$，位移为 0。  
- **边界条件**：支持在指定表面施加热流边界（如 $x=1$ 面）以及在其余表面施加辐射边界（非线性 $T^4$），并可选施加位移边界。  
- **数据约束**：对观测温度数据做归一化后纳入数据损失，用于驱动参数反演。

评估阶段，采用分批推理与分批自动微分：先预测参数，再由参数驱动温度与位移场恢复，最终计算 von Mises 等效应力并输出动画与图表。

**参数反演流程示意图**

![caseMarkdown/case_c9f86444/viz/inverse_pinnsformer_schematic_strict_fixed.png](https://www.science42.tech/cases/caseMarkdown/case_c9f86444/viz/inverse_pinnsformer_schematic_strict_fixed.png)

该图对应本案例的反演闭环：网络以 $(x,y,z,t)$ 为输入预测物理场，未知参数（如 $k$）以可训练变量形式进入损失函数；通过联合优化 PDE 残差、边界/初值约束与观测数据误差，迭代更新参数并同步校正温度-位移预测。训练收敛后，可同时输出参数估计结果、温度误差分析及应力后处理结果，用于工程可解释评估。

## 创新点

- **端到端参数反演 $(k, Q_{\mathrm{source}}, ...)$：将未知参数统一到损失函数的可训练参数中**  
  相比固定参数的正向模拟路径，本案例在训练阶段直接学习参数场，实现对材料非均匀性或边界不确定性的联合反演。

- **边界条件精细化：同时支持热流边界与辐射边界（$T^4$ 非线性）**  
  在不同表面采用不同边界形式，并通过法向方向符号约定构造边界残差，适用于典型热控边界组合场景。

- **显存友好的评估与应力恢复：分批推理 + 分批自动微分计算完整应力张量**  
  对大规模点云评估时，采用分批策略避免显存爆炸，并输出应力张量分量与 von Mises 指标，便于工程分析与可视化交付。


## 1、控制方程

本案例考虑三维瞬态导热与小变形线弹性热弹耦合参数反演，在统一坐标系下建模。网络输入为 $(x,y,z,t)$，输出为未知参数场（如 $k(x,y,z,t), Q_{\mathrm{source}}(x,y,z,t)$）。

### 1.1 热场控制方程（瞬态传热）

温度场满足三维导热方程（可含体热源项）：

$$
\rho C_p \frac{\partial T}{\partial t} - k(x,y,z,t)\,\Delta T - Q_{\text{source}}(x,y,z,t) = 0.
$$
其中 $\rho, C_p$ 为已知参数，$k, Q_{\mathrm{source}}$ 为网络输出的未知参数。

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


### 1.4 总损失函数与分项定义

总损失函数建议统一写为：

$$
L_{\mathrm{total}}=
\lambda_{\mathrm{PDE}}L_{\mathrm{PDE}}+
\lambda_{\mathrm{BC}}L_{\mathrm{BC}}+
\lambda_{\mathrm{IC}}L_{\mathrm{IC}}+
\lambda_{\mathrm{data}}L_{\mathrm{data}}+
\lambda_{\mathrm{reg}}L_{\mathrm{reg}}.
$$

其中（按本案例热-弹耦合参数反演任务口径）：

**热方程残差损失**
$$
L_{\mathrm{PDE}}=
\frac{1}{N_{\Omega}}\sum_{i=1}^{N_{\Omega}}
\left\|
\rho C_p T_t(\mathbf{x}_i,t_i) - k(\mathbf{x}_i,t_i)\Delta T(\mathbf{x}_i,t_i) - Q_{\mathrm{source}}(\mathbf{x}_i,t_i)
\right\|^2.
$$

**边界条件损失（热流 + 辐射示例）**
$$
L_{\mathrm{BC}}=
\frac{1}{N_{\Gamma_q}}\sum_{\mathbf{x}_i\in\Gamma_q}
\left\|k(\mathbf{x}_i,t_i)\nabla T\cdot \mathbf{n}-q_{bc}\right\|^2+
\frac{1}{N_{\Gamma_r}}\sum_{\mathbf{x}_i\in\Gamma_r}
\left\|-k(\mathbf{x}_i,t_i)\nabla T\cdot \mathbf{n}+\epsilon\sigma(T^4-T_{amb}^4)\right\|^2.
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
\left(\|k_{xx}\|^2+\|k_{yy}\|^2+\|k_{zz}\|^2 + \|Q_{xx}\|^2 + \dots\right).
$$




## 2、参数、属性定义（核心参数）

### 2.1 热学参数

| 参数 | 含义 | 单位 |
|---|---|---|
| $\rho$ | 密度 | kg/m³ |
| $C_p$ | 比热容 | J/(kg·K) |
| $k$ | 热导率（未知，反演目标） | W/(m·K) |
| $Q_{\text{source}}$ | 体热源项（未知，反演目标） | W/m³ |
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
| 输出 | $(T, u, v, w)$ |
| 物理约束 | 热方程残差、弹性平衡残差、初始条件、热边界条件（热流/辐射）、（可选）位移边界条件 |
| 数据约束 | 观测温度数据 |
| 可训练参数 | 未知物理参数（如 k, Q_source, alpha_t 等）|
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

> 实现说明：为保证工程可用性与训练稳定性，当前实现采用**网络输出物理场 (T, u, v, w)，未知参数在损失函数中作为可训练 Parameter** 的 PINNsformer；弹性部分以“热应力平衡的简化物理约束 + 评估阶段纯物理热应力后处理”为主。若需要严格的热弹耦合（联合求解位移场），可在后续版本将网络输出扩展为 $(k, Q, u, v, w)$ 并加入完整弹性本构与位移边界条件。




### 3.1 PINNsformer：输入 $(x,y,z,t)$ 输出 $(T, u, v, w)$

网络结构为“线性嵌入 + Transformer 编解码 + MLP 输出头”。输出维度为 4（温度 T 和位移 u,v,w），并引入可学习的输出偏置（默认初始化为 $[T_0, 0, 0, 0]$）以改善收敛稳定性。

```python
import torch
import torch.nn as nn
from utils import get_clones

class WaveAct(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.Tanh()  # 稳定激活函数

    def forward(self, x):
        return self.act(x)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=256):
        super(FeedForward, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(d_model, d_ff),
            WaveAct(),
            nn.Linear(d_ff, d_ff),
            WaveAct(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.linear(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super(EncoderLayer, self).__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=heads,
            batch_first=True
        )
        self.ff = FeedForward(d_model)
        self.act1 = WaveAct()
        self.act2 = WaveAct()

    def forward(self, x):
        x2 = self.act1(x)
        x = x + self.attn(x2, x2, x2)[0]
        x2 = self.act2(x)
        x = x + self.ff(x2)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super(DecoderLayer, self).__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=heads,
            batch_first=True
        )
        self.ff = FeedForward(d_model)
        self.act1 = WaveAct()
        self.act2 = WaveAct()

    def forward(self, x, e_outputs):
        x2 = self.act1(x)
        x = x + self.attn(x2, e_outputs, e_outputs)[0]
        x2 = self.act2(x)
        x = x + self.ff(x2)
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, N, heads):
        super(Encoder, self).__init__()
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.act = WaveAct()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.act(x)

class Decoder(nn.Module):
    def __init__(self, d_model, N, heads):
        super(Decoder, self).__init__()
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.act = WaveAct()

    def forward(self, x, e_outputs):
        for layer in self.layers:
            x = layer(x, e_outputs)
        return self.act(x)

class PINNsformer(nn.Module):
    """
    PINNsformer for thermo-elastic problems (fully coupled)
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

        # 可学习的输出偏置，初始化为给定值
        self.output_bias = nn.Parameter(
            torch.tensor([T0, 0., 0., 0.], dtype=torch.float32)
        )

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, y, z, t):
        src = torch.cat((x, y, z, t), dim=-1)
        src = self.linear_emb(src)
        e_outputs = self.encoder(src)
        d_output = self.decoder(src, e_outputs)
        output = self.linear_out(d_output)
        output = output + self.output_bias
        return output
```
### 3.2 ThermoElasticPINNLoss：热弹性耦合损失函数（支持参数反演）

损失函数通过自动微分构造 PDE 残差，支持将未知参数注册为可训练 Parameter，实现参数反演。

```python
class ThermoElasticPINNLoss(nn.Module):
    def __init__(self, args, trainable_params=None, T_mean=None, T_std=None):
        """
        热弹性耦合 PINN 损失函数（支持反演）
        :param args: 配置对象，包含所有物理常数和损失权重
        :param trainable_params: 需要反演的参数名列表，如 ['k', 'alpha_t', 'q_bc_x1']
        :param T_mean, T_std: 温度归一化参数（可选）
        """
        super().__init__()
        if trainable_params is None:
            trainable_params = []

        # ---------- 注册可训练参数（Parameter）----------
        for name in trainable_params:
            if hasattr(args, name):
                init_val = getattr(args, name)
                self.register_parameter(name, nn.Parameter(torch.tensor(init_val, dtype=torch.float32)))
            else:
                raise ValueError(f"Unknown parameter name: {name}")

        # ---------- 注册所有物理常数（不可训练 buffer）----------
        # 定义所有可能用到的常数（包括可能被反演的，但已注册过则跳过）
        all_constants = {
            'k': args.k,
            'rho': args.rho,
            'Cp': args.c_p,          # 注意命名与 forward 中一致
            'T0': args.t_0,
            'Q_source': args.q_sum,
            'E': args.e,
            'nu': args.nu,
            'alpha_t': args.alpha_t,
            'T_ref': args.t_ref,
            'eps': args.eps,
            'sigma': args.sigma,
            'T_amb': args.t_amb,
            'q_bc_x1': args.q_bc_x1,
        }
        for name, val in all_constants.items():
            if not hasattr(self, name):   # 避免覆盖已注册的 Parameter
                self.register_buffer(name, torch.tensor(val, dtype=torch.float32))

        # ---------- 损失权重（普通属性，不可训练）----------
        self.lambda_pde = args.lambda_pde
        self.lambda_elastic = args.lambda_elastic
        self.lambda_ic = args.lambda_ic
        self.lambda_ic_disp = args.lambda_ic_disp
        self.lambda_bc = args.lambda_bc
        self.lambda_bc_disp = args.lambda_bc_disp
        self.lambda_data = args.lambda_data
        self.lambda_heatflux = args.lambda_heatflux
        self.lambda_rad = args.lambda_rad

        # ---------- 温度归一化参数 ----------
        self.T_mean = T_mean
        self.T_std = T_std

    # =========================================================
    # 安全梯度（防 None / NaN / Inf）
    # =========================================================
    def _grad(self, y, x):
        g = torch.autograd.grad(
            y,
            x,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]

        if g is None:
            return torch.zeros_like(x)

        g = torch.nan_to_num(g, nan=0.0, posinf=1e6, neginf=-1e6)
        return g

    # =========================================================
    # 前向计算损失
    # =========================================================
    def forward(
        self,
        model,
        res_tensor=None,
        ic_tensor=None,
        bc_tensors=None,
        data_tensor=None,
        T_data=None,
    ):
        device = next(model.parameters()).device

        # 初始化各项损失
        loss_heat = torch.zeros(1, device=device)
        loss_elastic = torch.zeros(1, device=device)
        loss_ic = torch.zeros(1, device=device)
        loss_ic_disp = torch.zeros(1, device=device)
        loss_bc_temp = torch.zeros(1, device=device)
        loss_bc_disp = torch.zeros(1, device=device)
        loss_data = torch.zeros(1, device=device)
        loss_heatflux = torch.zeros(1, device=device)
        loss_rad = torch.zeros(1, device=device)

        # ---------- 计算依赖于可训练参数的派生量（每次 forward 重新计算）----------
        mu = self.E / (2.0 * (1.0 + self.nu))
        lmbda = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        beta = (3.0 * lmbda + 2.0 * mu) * self.alpha_t

        # =====================================================
        # PDE 残差
        # =====================================================
        if res_tensor is not None and res_tensor.numel() > 0:
            res_tensor = res_tensor.clone().detach().requires_grad_(True)

            x = res_tensor[:, 0:1]
            y = res_tensor[:, 1:2]
            z = res_tensor[:, 2:3]
            t = res_tensor[:, 3:4]

            out = model(x, y, z, t)
            T = out[:, 0:1]
            u = out[:, 1:2]
            v = out[:, 2:3]
            w = out[:, 3:4]

            # 热方程
            T_t = self._grad(T, t)
            T_x = self._grad(T, x)
            T_y = self._grad(T, y)
            T_z = self._grad(T, z)

            T_xx = self._grad(T_x, x)
            T_yy = self._grad(T_y, y)
            T_zz = self._grad(T_z, z)

            heat_res = (
                self.rho * self.Cp * T_t
                - self.k * (T_xx + T_yy + T_zz)
                - self.Q_source
            )

            heat_res = torch.nan_to_num(heat_res, nan=0.0, posinf=1e6, neginf=-1e6)
            loss_heat = torch.mean(heat_res ** 2)

            # 弹性方程
            if self.lambda_elastic > 0:
                u_x = self._grad(u, x)
                u_y = self._grad(u, y)
                u_z = self._grad(u, z)

                v_x = self._grad(v, x)
                v_y = self._grad(v, y)
                v_z = self._grad(v, z)

                w_x = self._grad(w, x)
                w_y = self._grad(w, y)
                w_z = self._grad(w, z)

                div_u = u_x + v_y + w_z

                div_u_x = self._grad(div_u, x)
                div_u_y = self._grad(div_u, y)
                div_u_z = self._grad(div_u, z)

                lap_u = self._grad(u_x, x) + self._grad(u_y, y) + self._grad(u_z, z)
                lap_v = self._grad(v_x, x) + self._grad(v_y, y) + self._grad(v_z, z)
                lap_w = self._grad(w_x, x) + self._grad(w_y, y) + self._grad(w_z, z)

                res_x = (lmbda + mu) * div_u_x + mu * lap_u - beta * T_x
                res_y = (lmbda + mu) * div_u_y + mu * lap_v - beta * T_y
                res_z = (lmbda + mu) * div_u_z + mu * lap_w - beta * T_z

                elastic_res = res_x ** 2 + res_y ** 2 + res_z ** 2
                elastic_res = torch.nan_to_num(elastic_res, nan=0.0, posinf=1e6, neginf=-1e6)

                loss_elastic = torch.mean(elastic_res)

        # =====================================================
        # 初始条件
        # =====================================================
        if ic_tensor is not None and ic_tensor.numel() > 0:
            x_ic = ic_tensor[:, 0:1]
            y_ic = ic_tensor[:, 1:2]
            z_ic = ic_tensor[:, 2:3]
            t_ic = ic_tensor[:, 3:4]

            out_ic = model(x_ic, y_ic, z_ic, t_ic)

            T_ic_pred = out_ic[:, 0:1]
            u_ic_pred = out_ic[:, 1:2]
            v_ic_pred = out_ic[:, 2:3]
            w_ic_pred = out_ic[:, 3:4]

            loss_ic = torch.mean((T_ic_pred - self.T0) ** 2)
            loss_ic_disp = torch.mean(
                u_ic_pred ** 2 + v_ic_pred ** 2 + w_ic_pred ** 2
            )

        # =====================================================
        # 边界条件
        # =====================================================
        if bc_tensors is not None and isinstance(bc_tensors, dict):
            for face_name, bc_pts in bc_tensors.items():
                if bc_pts.numel() == 0:
                    continue
                bc_pts = bc_pts.clone().detach().requires_grad_(True)
                x_bc = bc_pts[:, 0:1]
                y_bc = bc_pts[:, 1:2]
                z_bc = bc_pts[:, 2:3]
                t_bc = bc_pts[:, 3:4]

                out_bc = model(x_bc, y_bc, z_bc, t_bc)

                T_bc_pred = out_bc[:, 0:1]
                u_bc_pred = out_bc[:, 1:2]
                v_bc_pred = out_bc[:, 2:3]
                w_bc_pred = out_bc[:, 3:4]

                if face_name == 'x1':   # x=1 面施加热流边界
                    T_x = self._grad(T_bc_pred, x_bc)
                    # 直接使用 self.q_bc_x1（已注册为 buffer 或 Parameter）
                    heat_flux_res = self.k * T_x - self.q_bc_x1
                    loss_heatflux = loss_heatflux + torch.mean(heat_flux_res ** 2)

                else:
                    # 其他面：辐射边界（根据面确定法向导数的符号）
                    if face_name == 'x0':
                        T_n = self._grad(T_bc_pred, x_bc)   # 法向为 -x
                        sign = 1.0
                    elif face_name == 'y0':
                        T_n = self._grad(T_bc_pred, y_bc)   # 法向为 -y
                        sign = 1.0
                    elif face_name == 'y1':
                        T_n = self._grad(T_bc_pred, y_bc)   # 法向为 +y
                        sign = -1.0
                    elif face_name == 'z0':
                        T_n = self._grad(T_bc_pred, z_bc)   # 法向为 -z
                        sign = 1.0
                    elif face_name == 'z1':
                        T_n = self._grad(T_bc_pred, z_bc)   # 法向为 +z
                        sign = -1.0
                    else:
                        continue

                    # 辐射残差：sign * k * T_n + εσ (T_amb⁴ - T⁴) = 0
                    rad_res = sign * self.k * T_n + self.eps * self.sigma * (self.T_amb**4 - T_bc_pred**4)
                    loss_rad = loss_rad + torch.mean(rad_res ** 2)

                    if self.lambda_bc_disp > 0:
                        loss_bc_disp = loss_bc_disp + torch.mean(
                            u_bc_pred ** 2 + v_bc_pred ** 2 + w_bc_pred ** 2
                        )

        # =====================================================
        # 数据损失
        # =====================================================
        if data_tensor is not None and T_data is not None and data_tensor.numel() > 0:
            x_d = data_tensor[:, 0:1]
            y_d = data_tensor[:, 1:2]
            z_d = data_tensor[:, 2:3]
            t_d = data_tensor[:, 3:4]

            T_pred = model(x_d, y_d, z_d, t_d)[:, 0:1]
            if self.T_mean is not None and self.T_std is not None:
                T_pred_norm = (T_pred - self.T_mean) / self.T_std
                loss_data = torch.mean((T_pred_norm - T_data) ** 2)
            else:
                loss_data = torch.mean((T_pred - T_data) ** 2)

        # =====================================================
        # 加权总损失
        # =====================================================
        total_loss = (
            self.lambda_pde * loss_heat
            + self.lambda_elastic * loss_elastic
            + self.lambda_ic * loss_ic
            + self.lambda_ic_disp * loss_ic_disp
            + self.lambda_bc * loss_bc_temp
            + self.lambda_bc_disp * loss_bc_disp
            + self.lambda_data * loss_data
            + self.lambda_heatflux * loss_heatflux
            + self.lambda_rad * loss_rad
        )

        total_loss = torch.nan_to_num(total_loss, nan=1e6, posinf=1e6, neginf=-1e6)

        return {
            "total": total_loss,
            "heat": loss_heat.detach(),
            "elastic": loss_elastic.detach(),
            "ic": loss_ic.detach(),
            "ic_disp": loss_ic_disp.detach(),
            "bc_temp": loss_bc_temp.detach(),
            "bc_disp": loss_bc_disp.detach(),
            "data": loss_data.detach(),
            "heatflux": loss_heatflux.detach(),
            "rad": loss_rad.detach(),
        }
```
### 3.3 应力恢复：分批自动微分计算 $\sigma_{ij}$ 与 $\sigma_{vm}$

为避免大规模点云评估导致显存爆炸，采用“分批设置 requires_grad + 分批自动微分 + 拼接结果”的策略。应力计算遵循小变形线弹性热弹性形式：

由位移梯度得到应变

引入热应变 $\epsilon_{th}=\alpha(T-T_{ref})$

用 Lamé 常数 $(\lambda,\mu)$ 构造应力

计算 von Mises 等效应力
```python
def compute_stress_from_model(model, x, y, z, t, E, nu, alpha, T_ref, batch_size=2000):
    """
    利用模型输出和自动微分计算应力张量，分批处理避免显存爆炸。
    返回所有应力分量及 von Mises 的 numpy 数组（扁平化）。
    """
    device = x.device
    n_points = x.shape[0]

    # 预先分配结果列表
    sigma_xx_list, sigma_yy_list, sigma_zz_list = [], [], []
    sigma_xy_list, sigma_yz_list, sigma_xz_list = [], [], []
    sigma_vm_list = []

    for i in range(0, n_points, batch_size):
        # 取出当前批次，并设置 requires_grad
        xb = x[i:i+batch_size].clone().detach().requires_grad_(True).to(device)
        yb = y[i:i+batch_size].clone().detach().requires_grad_(True).to(device)
        zb = z[i:i+batch_size].clone().detach().requires_grad_(True).to(device)
        tb = t[i:i+batch_size].clone().detach().requires_grad_(True).to(device)

        # 前向传播
        out = model(xb, yb, zb, tb)
        T = out[:, 0:1]
        u = out[:, 1:2]
        v = out[:, 2:3]
        w = out[:, 3:4]

        # 位移梯度（一阶导数）
        u_x = torch.autograd.grad(u, xb, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, yb, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_z = torch.autograd.grad(u, zb, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        v_x = torch.autograd.grad(v, xb, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_y = torch.autograd.grad(v, yb, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_z = torch.autograd.grad(v, zb, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        w_x = torch.autograd.grad(w, xb, grad_outputs=torch.ones_like(w), create_graph=True)[0]
        w_y = torch.autograd.grad(w, yb, grad_outputs=torch.ones_like(w), create_graph=True)[0]
        w_z = torch.autograd.grad(w, zb, grad_outputs=torch.ones_like(w), create_graph=True)[0]

        # 应变（小变形）
        eps_xx = u_x
        eps_yy = v_y
        eps_zz = w_z
        eps_xy = 0.5 * (u_y + v_x)
        eps_xz = 0.5 * (u_z + w_x)
        eps_yz = 0.5 * (v_z + w_y)

        # 热应变
        eps_th = alpha * (T - T_ref)

        # 拉梅常数
        mu = E / (2.0 * (1.0 + nu))
        lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        # 应力
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

        # 分离结果并转移到CPU
        sigma_xx_list.append(sigma_xx.detach().cpu().numpy().flatten())
        sigma_yy_list.append(sigma_yy.detach().cpu().numpy().flatten())
        sigma_zz_list.append(sigma_zz.detach().cpu().numpy().flatten())
        sigma_xy_list.append(sigma_xy.detach().cpu().numpy().flatten())
        sigma_yz_list.append(sigma_yz.detach().cpu().numpy().flatten())
        sigma_xz_list.append(sigma_xz.detach().cpu().numpy().flatten())
        sigma_vm_list.append(sigma_vm.detach().cpu().numpy().flatten())

        # 清理中间变量以释放显存
        del xb, yb, zb, tb, out, T, u, v, w
        del u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y, w_z
        del eps_xx, eps_yy, eps_zz, eps_xy, eps_xz, eps_yz, eps_th
        del sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_xz, sigma_yz, sigma_vm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 拼接所有批次的结果
    sigma_xx = np.concatenate(sigma_xx_list)
    sigma_yy = np.concatenate(sigma_yy_list)
    sigma_zz = np.concatenate(sigma_zz_list)
    sigma_xy = np.concatenate(sigma_xy_list)
    sigma_yz = np.concatenate(sigma_yz_list)
    sigma_xz = np.concatenate(sigma_xz_list)
    sigma_vm = np.concatenate(sigma_vm_list)

    return sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_xz, sigma_vm
```
### 3.4 温度误差指标：MAE / RMSE / R2 / Relative L2

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
- 将模型预测的参数场（如热导率）在统一点云上可视化为序列图/动画  
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
    在给定点云 xyz 上推理标量场（默认热导率）
    xyz: (N,3), t_scalar: float
    return: (N,) numpy
    """
    x = torch.from_numpy(xyz[:, 0:1]).to(device)
    y = torch.from_numpy(xyz[:, 1:2]).to(device)
    z = torch.from_numpy(xyz[:, 2:3]).to(device)
    t = torch.full_like(x, float(t_scalar)).to(device)

    out = model(x, y, z, t)          # (N,2) 
    k = out[:, 0].detach().cpu().numpy()  # 取热导率
    return k


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
    在同一 Sym 点云上输出时间序列 GIF（参数场）
    """
    import imageio

    frames = []
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for k, tk in enumerate(t_seq):
        val = infer_scalar_on_points(model, xyz, tk, device=device)
        render_pointcloud_frame(ax, xyz, val, title=f"Inverse Parameter, t={tk:.3f}")
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
- 若以参数反演精度为首要目标，可先设置 $\lambda_{data}$ 较大，$\lambda_{pde}$ 作为正则项；  
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




### 5.1 训练收敛与反演参数
**图 5-1 训练损失与反演参数 k 收敛曲线**

![caseMarkdown/case_c9f86444/viz/loss_curve_final.png](https://www.science42.tech/cases/caseMarkdown/case_c9f86444/viz/loss_curve_final.png)

**代码对应解读：**
- 损失历史使用 `semilogy` 展示，便于观察跨数量级下降。
- 当前案例开启反演模式且只反演 `k`，因此右下角显示 `k` 随 epoch 的演化轨迹。
- 曲线后期趋稳表示优化进入收敛区，`k` 平台值可作为本轮反演估计结果。
### 5.2 k 参数反演对比

**图 5-2 真值 k 与反演 k 对比**

![caseMarkdown/case_c9f86444/viz/k_true_vs_inferred.png](https://www.science42.tech/cases/caseMarkdown/case_c9f86444/viz/k_true_vs_inferred.png)
## 6、常见问题（FAQ）

- **收敛性问题**：损失震荡不下降、初值/边界权重设置不合理、时间步长过大导致瞬态不稳定、采样点覆盖不足。  
- **精度问题**：热源附近/几何尖角/孔洞内壁处误差偏大，通常需要增加局部采样密度或分组边界条件。  
- **边界符号问题**：辐射/热流边界涉及 $\nabla T\cdot\mathbf{n}$，应以外法向统一符号，避免不同面手写 ±。  
- **计算效率问题**：应力恢复依赖自动微分，建议分批计算并缓存中间量（如梯度），避免显存峰值过高。





## 7、工业价值

- **航天器热控与结构可靠性参数反演**：  
  航天器在轨运行时受到太阳辐照、阴影进出、姿态变化与载荷工况切换等因素影响，温度场呈现强时变与空间非均匀特性。温度梯度会引发热膨胀差异，进一步导致连接件、薄壁区域、孔洞周边等位置出现应力集中。本案例通过点云域的参数反演，可快速从观测温度数据反推热导率、热源强度等未知参数，帮助工程人员在多组热边界参数下进行方案筛选与迭代，并用于识别潜在结构风险区域，支撑热控策略调整、结构加固与材料选型等工程决策。

- **电子设备散热与封装可靠性参数反演**：  
  高功耗芯片、功率器件在工作周期内会产生局部强热源，温度循环与局部梯度会诱发封装内部热应力，带来焊点疲劳、封装翘曲、微裂纹等可靠性问题。本案例在点云场上反演参数并输出应力指标，可用于评估不同散热结构、导热界面材料与壳体材料参数对温度峰值与应力峰值的影响，辅助元件布局与散热路径优化；同时可通过温度–应力关系图与应力时序分布，识别热热点对应的高风险区域，为可靠性设计与寿命评估提供可量化依据。

- **复杂几何快速验证参数反演**：  
  工程结构中常见轻量化孔洞、开槽、加强筋等复杂几何，传统多物理场仿真往往需要耗时的网格划分与反复调参，难以满足快速迭代验证需求。本案例以点云表示为基础，在复杂几何下仍可完成参数反演并输出可视化交付物，能够直观展示孔洞内壁、几何尖角等敏感区域的参数分布情况；同时支持对多版几何构型进行快速对比，用同一套指标体系评估结构改动对热风险与应力风险的影响，从而在工程设计早期阶段实现高效率的筛选与验证。

---



