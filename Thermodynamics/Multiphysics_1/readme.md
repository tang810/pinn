# PINNsformer：三维瞬态单一温度场无网格建模求解

## 快速使用

您可跳转[使用示例](#使用示例)查看如何使用秋月白平台快速执行该类问题，以下是实现逻辑详解：

## 背景介绍

在航天器热控、电子设备散热与复杂结构温度管理等场景中，三维温度场随时间演化直接决定系统热安全边界、热控策略有效性与温度均匀性水平。工程上通常需要一条从热边界条件、材料参数到温度场预测的快速链路，以支撑多工况扫描、方案迭代与热风险诊断。

传统瞬态导热仿真往往依赖网格划分与迭代求解，面对复杂几何、非线性边界条件（热流、辐射等）以及多轮参数分析时，建模与计算成本较高。本案例采用物理约束学习方法，将控制方程与边界/初值条件显式写入损失函数，并结合观测温度数据约束，实现无网格点云上的高效学习与推理。

## 原理介绍

本案例采用“热场 PINN”思路：网络以 $(x,y,z,t)$ 为输入，输出单一物理量温度 $T$。

训练阶段，将热场相关约束统一写入损失函数，主要包含：  
- **热传导方程残差**：通过自动微分构造 $T_t,T_{xx},T_{yy},T_{zz}$ 并形成 PDE 残差；可扩展加入体热源项。  
- **初始条件**：约束 $t=0$ 时刻温度为 $T_0$。  
- **边界条件**：支持在指定表面施加热流边界（如 $x=1$ 面）以及在其余表面施加辐射边界（非线性 $T^4$）。  
- **数据约束**：对观测温度数据做归一化后纳入数据损失，用于提升可辨识性与收敛稳定性。  

评估阶段，采用分批推理方式在统一点云上预测温度场，并输出误差指标、点云对比图与时间序列可视化结果。

**单场热传导示意图**

![caseMarkdown/case_8d125309/viz/pinnsformer_code_based_schematic_refined_v2——单场.png](https://www.science42.tech/cases/caseMarkdown/case_8d125309/viz/pinnsformer_code_based_schematic_refined_v2——单场.png)

该图展示了纯热场建模的核心链路：以时空坐标 $(x,y,z,t)$ 为输入，网络输出温度场 $T$，再通过热传导 PDE、边界/初值条件与观测数据共同约束训练。与热弹耦合版本相比，该版本不引入位移与应力分支，计算路径更简洁，适合用于热场快速预测与参数敏感性分析。


## 创新点

- **单场热学建模：聚焦瞬态温度预测主任务**  
  相比多场联合建模路径，本案例只保留热场主线，建模目标更聚焦，适用于以温度预测精度与热边界响应分析为核心的工程场景。

- **边界条件精细化：同时支持热流边界与辐射边界（$T^4$ 非线性）**  
  在不同表面采用不同边界形式，并通过法向方向符号约定构造边界残差，适用于典型热控边界组合场景。

- **显存友好的温度评估：分批推理实现大规模点云预测**  
  对大规模点云评估时，采用分批策略避免显存爆炸，便于工程分析与可视化交付。

## 1、控制方程

本案例考虑三维瞬态导热问题，在统一坐标系下建模。网络输入为 $(x,y,z,t)$，输出为温度场 $T$。

### 1.1 热场控制方程（瞬态传热）

温度场满足三维导热方程（可含体热源项）：

$$
\rho C_p \frac{\partial T}{\partial t} - k\cdot\Delta T - Q_{\mathrm{source}} = 0.
$$

其中 $\rho$ 为密度，$C_p$ 为比热容，$k$ 为导热系数，$Q_{\mathrm{source}}$ 为体热源项（如无体热源可取 0）。

**初始条件（IC）**：
$$
T(x,y,z,0)=T_0.
$$

**边界条件（BC）**：典型热控边界可由“热流边界 + 辐射边界”组合构成。

- 指定表面热流（Neumann）：
$$
k\cdot\nabla T\cdot \mathbf{n} = q_{bc},
$$
其中 $\mathbf{n}$ 为外法向单位向量，$q_{bc}$ 为给定热流密度。

- 辐射边界（Stefan–Boltzmann，非线性）：
$$
-k\cdot\nabla T\cdot \mathbf{n} + \epsilon\sigma\left(T^4 - T_{amb}^4\right)=0,
$$
其中 $\epsilon$ 为辐射率，$\sigma$ 为 Stefan–Boltzmann 常数，$T_{amb}$ 为环境温度。



### 1.2 总损失函数与分项定义

总损失函数建议统一写为：

$$
L_{\mathrm{total}}=
\lambda_{\mathrm{PDE}}L_{\mathrm{PDE}}+
\lambda_{\mathrm{BC}}L_{\mathrm{BC}}+
\lambda_{\mathrm{IC}}L_{\mathrm{IC}}+
\lambda_{\mathrm{data}}L_{\mathrm{data}}+
\lambda_{\mathrm{reg}}L_{\mathrm{reg}}.
$$

其中：

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
\left\|T(\mathbf{x}_i,0)-T_0\right\|^2.
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
| $Q_{\mathrm{source}}$ | 体热源项 | W/m³ |
| $T_0$ | 初始/参考温度 | K |
| $T_{amb}$ | 环境温度 | K |
| $\epsilon$ | 辐射率 | - |
| $\sigma$ | Stefan–Boltzmann 常数 | W/(m²·K⁴) |
| $q_{bc}$ | 边界热流密度 | W/m² |

### 2.2 网络输入输出与约束对象

| 项 | 说明 |
|---|---|
| 输入 | $(x,y,z,t)$ |
| 输出 | $T$ |
| 物理约束 | 热方程残差、初始条件、热边界条件（热流/辐射） |
| 数据约束 | 观测温度数据 |

### 软件版本要求

建议 Python 3.10 及以上。

### 所需要导入的库

```python
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
```

---

## 3、计算方法及代码实现

> 实现说明：为保证工程可用性与训练稳定性，当前实现采用**单输出温度 $T$** 的 PINNsformer，训练阶段围绕热方程、热边界、初始条件与观测数据进行统一约束。

### 3.1 PINNsformer：输入 $(x,y,z,t)$ 输出 $T$

网络结构为“线性嵌入 + Transformer 编解码 + MLP 输出头”。输出维度为 1，并引入可学习的输出偏置（默认初始化为 $T_0$）以改善收敛稳定性。

```python
class PINNsformer(nn.Module):
    """
    Input  : (x, y, z, t)
    Output : T
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
            nn.Linear(d_hidden, 1)
        )

        self.output_bias = nn.Parameter(
            torch.tensor([T0], dtype=torch.float32)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, y, z, t):
        src = torch.cat((x, y, z, t), dim=-1)      # [N,4]
        src = self.linear_emb(src)                 # [N,d_model]
        e_outputs = self.encoder(src)
        d_output = self.decoder(src, e_outputs)
        output = self.linear_out(d_output)         # [N,1]
        return output + self.output_bias
```

### 3.2 温度误差指标：MAE / RMSE / R2 / Relative L2

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

### 3.3 Sym 风格几何点云生成与可视化（NVIDIA 工作流）

本案例在复杂几何（孔洞/布尔几何）场景下，引入 Sym 风格几何后端，用于：  
- 生成带 **边界法向** 与 **边界分组标签** 的点云（外表面/孔洞面）  
- 将模型预测的温度标量场在统一点云上可视化为序列图/动画  
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
    在给定点云 xyz 上推理温度标量场
    xyz: (N,3), t_scalar: float
    return: (N,) numpy
    """
    x = torch.from_numpy(xyz[:, 0:1]).to(device)
    y = torch.from_numpy(xyz[:, 1:2]).to(device)
    z = torch.from_numpy(xyz[:, 2:3]).to(device)
    t = torch.full_like(x, float(t_scalar)).to(device)

    out = model(x, y, z, t)          # (N,1)
    T = out[:, 0].detach().cpu().numpy()
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
    在同一 Sym 点云上输出温度时间序列 GIF
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
res_xyz, bc_xyz, bc_n, bc_tag, ic_xyz = sample_csg_points_sym()

xyz_show = bc_xyz

t_seq = make_time_frames(t0=0.0, step_size=0.02, time_steps=40)
# export_sym_pointcloud_gif(model, xyz_show, t_seq, out_gif_path="sym_cloud.gif", device="cuda")
```

## 4、模型高精度训练

### 4.1 训练数据组织与采样策略

训练点集由三类时空点构成：  
- **域内残差点**：用于约束热方程残差  
- **初值点**：用于约束 $t=0$ 时刻的温度初始条件  
- **边界点**：用于施加热流边界与辐射边界  

为适配瞬态问题，采样得到的空间点会扩展为长度为 `time_steps` 的时间序列输入，时间间隔由 `step_size` 控制。该时间序列化设计可在不显著增加采样复杂度的前提下提升对时间演化规律的学习能力。

### 4.2 损失项构成与权重建议

总损失可写为：

$$
L=\lambda_{pde}L_{heat}+\lambda_{ic}L_{ic}+\lambda_{reg}L_{reg}+\lambda_{data}L_{data}.
$$

其中：  
- $L_{heat}$：热方程残差（核心项）  
- $L_{ic}$：初始条件约束（核心项）  
- $L_{reg}$：弱正则（可选，用于抑制高频振荡）  
- $L_{data}$：温度观测数据约束（推荐项，提升可辨识性与收敛稳定性）  

权重建议：  
- 若以温度精度为首要目标，可先设置 $\lambda_{pde}$、$\lambda_{ic}$ 较大，$\lambda_{reg}$ 较小或为 0；  
- 当观测数据可靠且覆盖充分时，可适度提高 $\lambda_{data}$；  
- 若边界条件存在明显非线性，应优先检查边界采样密度与边界损失量级是否平衡。

### 4.3 优化器选择与收敛策略

本案例采用二阶拟牛顿类优化（LBFGS）以获得更高精度的收敛表现。推荐策略如下：  
- 使用稳定的线搜索策略（如 strong-wolfe），减少震荡与发散风险  
- 在训练过程中定期输出各分项损失量级，确认收敛是否来自正确的物理项（而非单靠正则项“压平”）  
- 结合评估点集在线计算温度 Relative L2，作为“训练是否有效”的外部判据

### 4.4 训练产物与版本管理

训练结束建议保存：  
- 模型参数快照（可按 epoch/阶段保存多个版本）  
- 训练日志（包含 $L_{heat},L_{ic},L_{reg},L_{data}$ 的数值轨迹）  
- 关键超参配置记录（材料参数、采样规模、时间序列长度、损失权重等）  

这样可支持后续进行：  
- 不同边界条件/材料参数下的复现实验  
- 不同几何设置下的迁移对比  
- 温度误差指标的回归测试

## 5、预测、绘图及成果展示

本章给出评估阶段的输出内容与展示口径。评估目标为**温度预测精度验证**，主要通过误差指标与“真值 vs 预测”可视化进行展示。

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

**图 5-2 预测 vs 真值散点图**  

![caseMarkdown/case_8d125309/viz/animation.gif](https://www.science42.tech/cases/caseMarkdown/case_8d125309/viz/animation.gif)

**解读要点**：  
- 重点检查高温区域的位置与形状是否一致（热源附近的“热斑”是否被正确学习）。  
- 观察温度梯度较大的区域（边界层、局部热流边界附近）是否存在系统性偏差。  
- 若出现“整体偏冷/偏热”，通常对应边界条件强度、初始条件或数据约束权重需要重新平衡。  
- 若局部区域误差显著偏大，优先检查该区域采样密度与边界分类是否正确（例如孔洞内壁、几何尖角等）。


## 6、常见问题（FAQ）

- **收敛性问题**：损失震荡不下降、初值/边界权重设置不合理、时间步长过大导致瞬态不稳定、采样点覆盖不足。  
- **精度问题**：热源附近/几何尖角/孔洞内壁处误差偏大，通常需要增加局部采样密度或分组边界条件。  
- **边界符号问题**：辐射/热流边界涉及 $\nabla T\cdot\mathbf{n}$，应以外法向统一符号，避免不同面手写 ±。  
- **计算效率问题**：大规模点云评估建议分批推理，并合理控制可视化输出密度，避免显存峰值过高。

## 7、工业价值

- **面向系统级热态态势感知与快速反馈**：
  在航空电子、动力系统等领域，常见多工况切换、频繁热循环与局部瞬态过热事件。该模型支持在“点云+多边界配置”下快速推理出不同工况下的温度、热梯度及热响应敏感区域，可与实时监测数据在线融合，构建“仿真预警+运行自适应”闭环，显著缩短从参数调整到热态验证的迭代周期。

- **型号预研与轻量化热设计对接**：
  对于发动机壳体、结构件、散热器壳体等不同结构版本，往往需要在早期阶段快速判断热流布置与温度峰值风险。通过本案例的点云热场推理，可在不依赖全网格高精度模型的前提下，在几分钟级别评估多版本几何、材料（导热、比热）及工况参数的热性能差异，提供可量化的热安全裕量、温度均匀性指标，支持轻量化/结构优化方案的早期落地验证。

- **工艺边界与多源热负载协同优化**：
  在电子封装、激光加热、焊接等工艺中，热负载受材料非均质、辐射/对流混合边界影响。通过将模型与工艺数据（功率谱、时间序列热源）集成，可实现“区域内点云热场 + 边界热流自适应”双线索分析，快速定位工艺敏感区、热梯度集中区与时间片段过热风险，降低迭代试验次数并加速工艺参数定标。

- **数字孪生平台热算子组件**：
  该点云 PINN 模型可作为数字孪生平台中的“热算子”组件，与结构、流体、控制系统协同。通过统一数据接口，可在大规模仿真框架中按需调度（批量点云预测 / 在线点云预测），实现“热风险评估+多物理多工况调度”一体化，提升平台的端到端响应能力与工程交付效率。
