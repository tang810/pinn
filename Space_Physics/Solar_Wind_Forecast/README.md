<!-- 
text："基于PINN的太阳风多变量时间序列预测"
area: "本案例展示我们在太阳风预测中的一套“物理约束 + 深度时序模型”实现：将电场E、速度(Vx,Vy,Vz)、磁场(Bx,By,Bz)组成的7维小时级序列作为输入窗口（span=24h），预测提前量prior=12h后的目标状态。核心做法是在常规数据拟合损失之外，引入理想等离子体欧姆定律的松弛不等式约束 |E|−α||V×B||₂≤0，并用ReLU只惩罚违反物理约束的预测，从而在提升预测稳定性的同时减少物理不一致输出。案例包含：数据构造/预处理、窗口采样、GRU预测网络、α自适应拟合、物理损失与加权训练、以及覆盖‘精度+物理一致性’的完整可视化结果展示。"
tags: ["太阳风预测", "欧姆定律约束", "多变量时间序列", "GRU", "物理一致性"]
-->

# 基于PINN的太阳风多变量时间序列预测

## 快速使用

您可跳转[使用示例](#使用示例)查看如何使用秋月白平台快速执行该类问题，以下是实现逻辑详解：

---

## 背景介绍

太阳风是近地空间环境最重要的外部驱动之一。电场 $E$、速度 $\mathbf{V}$ 与磁场 $\mathbf{B}$ 的变化会通过磁层—电离层耦合过程影响地磁扰动、辐射带粒子通量、卫星充放电与姿轨控误差等风险评估链路。在工程应用中，许多模型与指标（例如某些地磁活动代理量、能量输入项、传播/到达时间估计等）都需要稳定、连续且物理一致的太阳风输入，因此对太阳风关键参数的提前预测具有直接价值。

从数据角度看，太阳风时间序列具有三类典型特征，使得“只追求误差最小”的方法往往不够可靠：

1. **强时序性 + 多尺度结构**
   太阳风包含小时到天尺度的周期与准周期变化，也包含更长周期的背景漂移，同时叠加了随机扰动与突变事件。模型需要同时捕捉短期惯性、日周期结构以及更慢的背景变化，否则容易出现滞后或过度平滑。

2. **强耦合性（变量之间不是独立的）**
   电场、速度和磁场之间并非自由组合。特别是电场与 $\mathbf{V}\times\mathbf{B}$ 存在基本物理联系——这意味着在同一时刻，$(E,\mathbf{V},\mathbf{B})$ 并不是任意的 7 维向量，而应落在一个“物理可行域”附近。

3. **工程风险来自“不一致输出”而不只是“误差”**
   纯数据驱动模型在短期预测上可能获得不错的 RMSE/R²，但仍可能输出物理上矛盾的组合，例如给出一个看似合理的电场幅值，却与预测的速度/磁场叉乘幅值尺度严重不匹配。这样的输出在后续链路中会被放大：它可能导致能量输入估计偏差、传播到达判断异常，或者在阈值判断类任务中触发错误告警/漏报。

因此，我们的目标不仅是把预测“做准”，还要把预测“做稳、做可信”——具体到方法层面，就是把已知的物理耦合关系显式纳入训练目标，使模型在学习时间依赖的同时，能够自动压制物理不可行的输出，提升预测的可解释性与可用性。

---

## 原理介绍

我们把问题统一表述为一个“**带物理约束的多变量时间序列预测**”任务：用过去一段时间的 7 维观测序列，预测未来提前量后的 7 维状态，并在训练时引入欧姆定律启发的物理一致性约束，作为额外的优化目标。

### 1）数据形式：7维小时级序列，既做预测也做物理一致性判断

每个时间点的观测向量定义为：

$$
\mathbf{x}_t=[E_t,Vx_t,Vy_t,Vz_t,Bx_t,By_t,Bz_t]\in\mathbb{R}^7
$$

其中：

* $E_t$：电场标量（工程数据中常以标量形式提供）
* $\mathbf{V}_t=(Vx_t,Vy_t,Vz_t)$：速度三分量
* $\mathbf{B}_t=(Bx_t,By_t,Bz_t)$：磁场三分量

这 7 个量不仅是预测对象，也是物理约束计算所需的最小集合：因为我们希望用 $\mathbf{V}\times\mathbf{B}$ 来构造对 $E$ 的物理约束，所以速度与磁场的三分量必须同时存在，才能在输出端计算 $\|\mathbf{V}\times\mathbf{B}\|$。

### 2）要预测什么：窗口输入 span 与提前量 prior

我们把时间序列预测写成窗口监督学习：

* 输入窗口长度：`span=24` 小时
* 预测提前量：`prior=12` 小时
* 训练样本定义：

$$
\mathbf{X}_t=[\mathbf{x}_{t},\mathbf{x}_{t+1},...,\mathbf{x}_{t+span-1}]\in\mathbb{R}^{span\times 7}
$$

$$
\mathbf{y}_t=\mathbf{x}_{t+span+prior-1}\in\mathbb{R}^7
$$

直观理解：模型看到过去 24 小时的演化轨迹，输出“提前 12 小时后”的状态向量。这样做的好处是：

* 预测目标是一个“点”，便于与实际观测对齐做监督学习；
* 模型输入保留完整窗口，可学习滞后效应与多尺度依赖；
* 适用于滚动预测与在线更新（工程部署友好）。

### 3）用什么算法：GRU 编码 + MLP 输出（多变量联合预测）

我们采用 GRU（门控循环单元）来编码输入窗口序列。GRU 适合处理这种小时级、多变量、具有长短期混合依赖的序列：

* GRU 通过门控结构在一定程度上缓解长序列的梯度消失问题；
* 对于 span=24 这样的窗口，GRU 能有效学习“过去一天”内的惯性、日周期趋势与突变响应；
* 与简单线性/AR 模型相比，GRU 能表达非线性耦合与跨变量的交互影响。

模型输出端使用 MLP head，将最后时刻隐藏状态映射为 7 维预测向量：

$$
\hat{\mathbf{y}}_t=f_\theta(\mathbf{X}_t)\in\mathbb{R}^7
$$

这里强调“联合预测”的意义：我们不把每个变量拆开独立预测，而是让模型在同一个隐藏空间里同时学习 $E$、$\mathbf{V}$、$\mathbf{B}$ 的共同动态结构。这样更容易让输出在统计层面保持一致，也为下一步引入物理一致性约束打下基础。

### 4）物理理论如何进入模型：欧姆定律提供“可行域”，用不等式做鲁棒约束

在理想等离子体近似下，欧姆定律可写作：

$$
\mathbf{J}=\sigma(\mathbf{E}+\mathbf{V}\times\mathbf{B})
$$

当电流密度 $\mathbf{J}\approx 0$ 时有：

$$
\mathbf{E}\approx -\mathbf{V}\times\mathbf{B}
$$

如果我们能获得电场三分量，那么可以直接对向量等式建约束；但在工程数据中，电场通常是标量形式（幅值或某方向投影），因此我们用一个更鲁棒的松弛形式：用叉乘幅值给出电场幅值的可解释尺度，从而构造不等式约束：

$$
|E|-\alpha\|\mathbf{V}\times\mathbf{B}\|_2 \le 0
$$

其中：

* $\|\mathbf{V}\times\mathbf{B}\|_2$ 表示速度与磁场叉乘向量的幅值，反映了电场应具有的“物理尺度”
* $\alpha$ 用于单位/尺度对齐（使右侧能与 $|E|$ 在同一尺度比较）

**为什么这一步重要？**
因为它把“预测输出空间”从任意的 7 维组合，缩小到一个更合理的物理可行域附近。纯数据驱动方法可能在误差上看起来不错，但在这个约束下会暴露“物理不一致”的输出；引入该约束后，模型会被训练成“尽量不跨出可行域边界”。

### 5）如何惩罚违反：只惩罚 $g>0$ 的部分（ReLU物理损失）

定义违反量：

$$
g = |E|-\alpha\|\mathbf{V}\times\mathbf{B}\|_2
$$

我们只对违反部分施加惩罚：

$$
L_{\text{phy}}=\mathbb{E}[\max(0,g)]
$$

这样做的意义是：

* 当预测已经满足物理边界（$g\le 0$）时，不施加额外压力，避免影响正常拟合；
* 当预测跨越物理边界（$g>0$）时，梯度会把 $|E|$ 往下拉、或推动 $\|\mathbf{V}\times\mathbf{B}\|$ 往合理方向调整，使输出回到可行域附近；
* 训练过程更稳定，也更符合“约束纠偏”的直觉。

最终训练目标是“数据拟合 + 物理一致性”两目标加权：

$$
L=(1-\lambda)L_{\text{data}}+\lambda L_{\text{phy}}
$$

其中 $\lambda$ 控制精度与物理一致性的权衡。


---

### 2）算法推导与设计（从物理关系到可训练目标）

这一节的目标是把物理关系变成一个**可优化、可训练、可度量**的目标函数，并解释每一步为什么这么做。

#### 2.1 物理出发点：理想等离子体欧姆定律

在等离子体中，一个常见的近似形式为：

$$
\mathbf{J}=\sigma(\mathbf{E}+\mathbf{V}\times\mathbf{B})
$$

在太阳风的很多情形下，可以把电中性/弱电流视为近似成立，即 $\mathbf{J}\approx 0$，于是得到

$$
\mathbf{E}\approx -\mathbf{V}\times\mathbf{B}
$$

这句关系很关键：它告诉我们 **电场不是“随便的一个数”**，它与速度、磁场存在结构性耦合。

#### 2.2 标量电场带来的现实限制：从“向量等式”到“标量不等式”

工程数据中电场常以 **标量**（幅值或某一方向分量）提供，而 $\mathbf V\times\mathbf B$ 是向量。为了在标量观测条件下仍能利用耦合关系，我们不强行使用三分量等式，而采用一个更稳健的“松弛约束”：

* 用 $\|\mathbf V\times\mathbf B\|_2$ 表示叉乘向量的幅值；
* 用 $|E|$ 表示电场标量的幅值（或绝对值）；
* 引入 $\alpha$ 用于单位/尺度对齐（速度、磁场、电场的单位体系通常不同）。

于是构造约束：

$$
|E| \le \alpha \|\mathbf{V}\times\mathbf{B}\|_2
$$

等价写成

$$
g = |E|-\alpha\|\mathbf{V}\times\mathbf{B}\|_2 \le 0
$$

直观解释：**右侧给出了由 $\mathbf V$ 与 $\mathbf B$ 决定的“可解释电场上界”**。如果模型预测出的 $|E|$ 超过这个上界，我们认为它更可能是“物理不一致”。

> 为什么用“不等式”而不是“等式”？
> 因为标量观测与理想近似都会引入偏差，硬等式会过强、导致训练不稳定；不等式更鲁棒，更适合作为工程约束。

#### 2.3 把约束变成可训练的物理损失：只罚违反者（ReLU）

我们希望训练时“尽量满足 $g\le 0$”，最直接的方式是设计一个 penalty：当 $g\le 0$ 时损失为 0，当 $g>0$ 时损失随违反程度增大而增大。最自然、可微且稳定的写法就是 ReLU：

$$
L_{\text{phy}} = \mathbb{E}\left[\max(0, g)\right]
=\mathbb{E}\left[\max\left(0,\ |E|-\alpha\|\mathbf{V}\times\mathbf{B}\|_2\right)\right]
$$

它有两个重要性质：

1. **不干扰“已经物理正确”的样本**：当预测满足约束，物理损失为 0，不会给优化器额外压力。
2. **把学习信号集中到“物理错误”的区域**：梯度只来自违反样本，训练过程更稳定、更像“纠偏器”。

#### 2.4 数据损失与物理损失的耦合：两目标权衡

如果只做物理损失，模型可能学会“把 E 压得很小”来避免违反，但预测会失真。所以必须保留数据拟合目标：

$$
L_{\text{data}}=\frac{1}{N}\sum_i \|\hat{\mathbf y}_i-\mathbf y_i\|^2
$$

最终总损失采用线性权衡：

$$
L=(1-\lambda)L_{\text{data}}+\lambda L_{\text{phy}}
$$

* $\lambda$ 越大：越强调物理一致性（更“保守”、更少违反），但可能牺牲部分拟合精度。
* $\lambda$ 越小：越强调预测精度，但物理一致性改善有限。

这种设计的优点是工程上可控：可以用验证集同时监控 **R² / RMSE** 和 **physics penalty / violation rate**，选择在业务目标下最合适的 $\lambda$。

#### 2.5 为什么必须在“原始单位”计算物理项

时间序列预测通常会做 Z-score：

$$
x_{norm} = \frac{x-\mu}{\sigma}
$$

但注意：$|E|$ 与 $\|\mathbf V\times \mathbf B\|$ 的比较依赖真实量纲与尺度。如果直接在归一化空间计算
$|E_{norm}|-\alpha\|\mathbf V_{norm}\times \mathbf B_{norm}\|$，那么：

* 归一化对每个维度使用不同 $\sigma$，会改变 “叉乘幅值” 的相对尺度；
* $\alpha$ 将失去“单位换算/尺度对齐”的物理含义，变成一个不可解释的数；
* 可能出现“归一化空间满足约束，但原始空间严重违反”的假象。

因此我们的实现流程是：

1. 模型输出在归一化空间：$\hat{\mathbf y}_{norm}$
2. 反归一化回原始单位：$\hat{\mathbf y}_{raw}=\hat{\mathbf y}_{norm}\cdot\sigma+\mu$
3. 在原始单位计算 $g=|E|-\alpha\|\mathbf V\times\mathbf B\|$
4. 用 ReLU 得到 $L_{phy}$


---

### 3）参数与数据（详细表）

> 下表按“代码里真实使用”的变量/超参数整理，便于复现实验与排查问题。

| 符号/变量                | 含义          | 代码对应                            | 典型取值/形状                               | 说明                     |   |   |   |     |   |             |
| -------------------- | ----------- | ------------------------------- | ------------------------------------- | ---------------------- | - | - | - | --- | - | ----------- |
| `FEATURES`           | 7维特征集合      | `FEATURES`                      | `["E","Vx","Vy","Vz","Bx","By","Bz"]` | 预测目标与输入维度一致            |   |   |   |     |   |             |
| `span`               | 观测窗口长度      | `WindowDataset(span=24, ...)`   | 24（小时）                                | 输入序列长度                 |   |   |   |     |   |             |
| `prior`              | 预测提前量       | `WindowDataset(..., prior=12)`  | 12（小时）                                | 预测未来第 `prior` 小时的状态    |   |   |   |     |   |             |
| `x`                  | 输入窗口        | `__getitem__`                   | `(span,7)`                            | `arr[t:t+span]`        |   |   |   |     |   |             |
| `y`                  | 预测目标        | `__getitem__`                   | `(7,)`                                | `arr[t+span+prior-1]`  |   |   |   |     |   |             |
| `mu`,`sig`           | Z-score统计量  | `normalize_z`                   | `(7,)`                                | 用训练集均值方差归一化            |   |   |   |     |   |             |
| `alpha` ($\alpha$)   | 物理约束尺度常数    | `fit_alpha_raw`                 | 标量                                    | \`median(              | E | / |   | V×B |   | )\`，原始单位拟合  |
| `lam` ($\lambda$)    | 物理损失权重      | `train_one_epoch(..., lam=0.2)` | 0\~1                                  | 平衡拟合与物理一致性             |   |   |   |     |   |             |
| `hidden`             | GRU隐藏维      | `GRUForecast(hidden=64)`        | 64                                    | 序列编码能力                 |   |   |   |     |   |             |
| `num_layers`         | GRU层数       | `GRUForecast(num_layers=1)`     | 1                                     | >1 可用 dropout          |   |   |   |     |   |             |
| `dropout`            | GRU dropout | `GRUForecast(dropout=0.1)`      | 0.1                                   | `num_layers>1`时有效      |   |   |   |     |   |             |
| `clip`               | 梯度裁剪        | `clip_grad_norm_`               | 1.0                                   | 稳定训练                   |   |   |   |     |   |             |
| 指标 `rmse`            | 均方根误差       | `eval_metrics`                  | 标量                                    | 评估预测误差                 |   |   |   |     |   |             |
| 指标 `r2_macro`        | R²宏平均       | `eval_metrics`                  | 标量                                    | 评估整体解释度                |   |   |   |     |   |             |
| 指标 `r2_per_feature`  | 分特征R²       | `eval_metrics`                  | dict                                  | 查看各变量可预测性差异            |   |   |   |     |   |             |
| 指标 `physics_penalty` | 物理违反惩罚      | `eval_metrics`                  | 标量                                    | 在原始单位下评估物理一致性          |   |   |   |     |   |             |

---

### 4）计算方法与代码实现（关键代码片段）

#### 4.1 窗口采样（span / prior）

```python
class WindowDataset(Dataset):
    def __init__(self, arr, span=24, prior=12):
        self.arr = torch.as_tensor(arr, dtype=torch.float32)
        self.span = span
        self.prior = prior
        self.T, self.D = self.arr.shape
        self.idxs = list(range(self.T - (span + prior)))

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        t = self.idxs[i]
        x = self.arr[t : t + self.span]                 # (span, D)
        y = self.arr[t + self.span + self.prior - 1]    # (D,)
        return x, y
```



#### 4.2 GRU 预测网络（GRU + MLP head）

```python
class GRUForecast(nn.Module):
    def __init__(self, input_dim=7, hidden=64, num_layers=1, dropout=0.1, output_dim=7):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.head(last)
```



#### 4.3 α 的自适应拟合（原始单位）

```python
def fit_alpha_raw(df_raw: pd.DataFrame, eps=1e-6):
    """
    用原始单位拟合 alpha: median(|E| / ||VxB||)
    """
    arr = df_raw[FEATURES].to_numpy(dtype=np.float64)
    E = np.abs(arr[:, 0])
    V = arr[:, 1:4]
    B = arr[:, 4:7]
    cross_norm = np.linalg.norm(np.cross(V, B), axis=1)
    ratio = E / (cross_norm + eps)
    ratio = ratio[np.isfinite(ratio)]
    return float(np.median(ratio)) if ratio.size else 1.0
```



#### 4.4 物理惩罚项（先反归一化，再算 ReLU(g)）

```python
def physics_penalty_original_units(yhat_norm, alpha, mu_t, sig_t):
    yhat_raw = inverse_z_torch(yhat_norm, mu_t, sig_t)  # back to original units

    E = yhat_raw[:, 0].abs()
    V = yhat_raw[:, 1:4]
    B = yhat_raw[:, 4:7]
    cross = torch.cross(V, B, dim=1)
    cross_norm = torch.linalg.vector_norm(cross, ord=2, dim=1)

    g = E - alpha * cross_norm
    return torch.relu(g).mean()
```



#### 4.5 训练：数据损失 + 物理损失加权

```python
def train_one_epoch(model, loader, opt, alpha, mu_t, sig_t, lam=0.2, device="cpu", clip=1.0):
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        yhat = model(x)

        # 主损失：用 MSE 更稳定（RMSE 梯度有时会抖）
        mse = torch.mean((yhat - y) ** 2)
        phy = physics_penalty_original_units(yhat, alpha, mu_t, sig_t)
        loss = (1 - lam) * mse + lam * phy

        loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()

        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)
```



---

### 5）结果展示

![caseMarkdown/case_beebbd4a/viz/loss.png](https://www.science42.tech/cases/caseMarkdown/case_beebbd4a/viz/loss.png)

训练损失曲线在前期快速下降，随后逐步趋缓并稳定收敛，表明模型参数优化有效、训练过程总体稳定。后期下降幅度变小，说明模型已接近当前结构与数据条件下的可达最优区域。


![caseMarkdown/case_beebbd4a/viz/penalty.png](https://www.science42.tech/cases/caseMarkdown/case_beebbd4a/viz/penalty.png)

这条曲线反映验证集上的 物理违反惩罚 随训练的变化。前几个 epoch 惩罚迅速降低，说明模型很快学会避免明显的物理不一致；后期在较低水平附近小幅波动，代表物理约束总体被保持，但在少量复杂样本或噪声段仍会出现轻微违反。


![caseMarkdown/case_beebbd4a/viz/physics.png](https://www.science42.tech/cases/caseMarkdown/case_beebbd4a/viz/physics.png)

这张图用于检查 物理一致性：横轴是由速度与磁场计算得到的标度项，纵轴是预测电场的绝对值（均为原始单位）。大多数点落在参考直线（边界）附近且主要位于其下方，说明预测结果整体符合“电场不应超过由 V×B 决定的物理上界”的约束；少量偏离点对应仍存在的物理违反样本。

![caseMarkdown/case_beebbd4a/viz/vx.png](https://www.science42.tech/cases/caseMarkdown/case_beebbd4a/viz/vx.png)

这张散点图展示了测试集中 Vx 的真实值与预测值 的对应关系（已归一化）。点云整体沿对角线分布，说明模型能较好地捕捉 Vx 的变化趋势与幅度；少量离群点反映了在极端波动或噪声较强的区间，预测仍存在偏差。

---

## 常见问题

1. **为什么物理项要在“原始单位”计算？**
   物理不等式 $|E|-\alpha\|V\times B\|_2\le 0$ 依赖量纲与尺度。若直接在归一化空间比较，$\alpha$ 的意义会被破坏，因此实现中先反归一化再计算 `g` 并 ReLU 惩罚。

2. **$\alpha$ 为什么用 median 拟合？**
   `median(|E|/||V×B||)` 对异常点更鲁棒，能更稳定地给出尺度常数，避免少量极端样本主导 $\alpha$。

3. **lam 取值怎么选？**
   `lam` 决定“拟合数据”与“服从物理”的权衡：`lam` 太大可能牺牲拟合精度，太小则物理一致性改善有限。训练时可监控 `val_r2` 与 `val_phys` 的共同趋势来调参。

---

## 研究价值

* **可信预测**：在不牺牲主要预测能力的前提下，将物理一致性作为优化目标，减少违反基本物理耦合的输出，使预测结果更可用、更可解释。
* **可迁移方法论**：把“物理约束 → 可微 penalty → 与数据损失加权融合”的范式迁移到其他空间物理/地球物理/工程系统的时序预测任务中。

* **把“准确”升级为“可信”**
  传统评估往往只关注 RMSE/R² 等统计误差，但在空间天气链路中，更关键的是输出是否物理自洽。我们将欧姆定律约束显式纳入训练目标，使模型在保持预测能力的同时，显著降低物理不一致输出的概率，从而让预测结果更适合作为下游模型与风险评估系统的输入。

* **提升极端/异常场景下的稳健性**
  太阳风序列包含突变、扰动与噪声。在这些场景中，纯数据驱动模型容易产生“发散式”预测（例如电场异常放大、与 $\|\mathbf V\times\mathbf B\|$ 脱钩）。物理可行域约束相当于为模型提供了一个“安全边界”，在异常段落起到纠偏与抑制作用，减少不合理峰值与不稳定振荡，提高工程可用性。

* **提供可审计、可解释的质量控制机制**
  我们不仅输出误差指标，还引入物理一致性诊断：$|E|$ 与 $\alpha\|\mathbf V\times\mathbf B\|$ 的散点、违反量 $g$ 的分布与违反率等。这些诊断使模型输出的质量能够被量化、被追踪、被审计，便于模型迭代、上线验收与长期监控。

* **形成可迁移的方法范式**
  该框架将“物理约束 → 可微惩罚项 → 与数据损失联合优化”的路径工程化落地，具备通用性：只要一个系统存在明确的守恒律、约束不等式或变量耦合关系（等离子体、气象水文、地球物理、工程控制等），就可以用类似方式把物理先验融入时序预测或回归任务，提升泛化与可靠性。




