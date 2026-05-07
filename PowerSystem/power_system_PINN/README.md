<!-- 
  "text": "电力系统频率扰动：基于 PINN 物理约束的 FNO 时空建模（10 节点）",
  "area": "本案例聚焦潮流背景下的多节点电力系统频率扰动传播。以摆动方程和潮流耦合为物理先验，在无标签情形下通过物理残差训练 Fourier Neural Operator（FNO），并采用硬初值嵌入保证波到达前的频率/角度一致性。目标是用极少的先验与参数，稳定复现节点间的扰动传播、阻尼与合拍特性，并与数值解对照评估误差。",
  "tags": ["电力系统", "频率稳定", "FNO", "摆动方程", "潮流耦合", "时空建模"],
  "search": ["PINN（基础物理信息神经网络）","波动方程"，“电磁学”]
-->

# 基于 PINN 约束的 FNO：10 节点电力系统频率扰动传播

## 项目背景

在输电网络中，任一节点的有功扰动都会沿电气拓扑在各母线间传播并引起频率偏移。传统做法依赖数值积分（如 RK4）逐步求解摆动方程；本项目采用 **Physics-Informed** 思路：不使用数据标签，直接把**摆动方程 + 潮流耦合**作为训练目标，用 **Fourier Neural Operator (FNO)** 建立**时间—空间**上的函数映射，学习 10 节点系统中频率扰动的传播与衰减规律。

> **目标**：最小化物理残差（PDE），并通过**硬初值**确保故障前系统处于给定潮流稳态，使学习到的解物理一致、数值平滑且对网格分辨率不敏感。

---

## 物理建模（核心先验）

### 1) 单机等效/多机摆动方程（逐节点）

对第 $i$ 个节点，引入电角度 $\theta_i(t)$、频率偏移 $f_i(t) = \dot \theta_i / (2\pi)$。摆动方程写为

$$
\ddot \theta_i
= \frac{f_b}{2H_i} \Big(P_{m,i} - P_{e,i} - D_i \frac{\dot \theta_i}{2\pi}\Big),
$$

其中 $f_b$ 为基准频率，$H_i$ 惯性常数，$D_i$ 阻尼，$P_{m,i}$ 机械输入，$P_{e,i}$ 电气输出（见下）。

### 2) 潮流耦合（链式等值）

使用简单的正弦功角关系表征相邻母线耦合（代码中给出线性链拓扑示例）：

$$
P_{line,i\to i+1} = K \sin(\theta_i - \theta_{i+1}), \qquad
K=\frac{U^2}{X_{line}},
$$

令 $P_{load,i}(t)$ 为节点负荷，得到

$$
\begin{aligned}
P_{e,1} &= P_{load,1} + P_{line,1\to 2},\\
P_{e,i} &= P_{load,i} + P_{line,i\to i+1} - P_{line,i-1\to i}, \quad i=2,\dots,N-1,\\
P_{e,N} &= P_{load,N} - P_{line,N-1\to N}.
\end{aligned}
$$

扰动通过在某节点 $b\_\text{event}$ 对 $P_{load,b}$ 施加阶跃 $\Delta P$ 引入。

### 3) 初值与“硬约束”嵌入（Hard IC）

训练时不直接回归 $\theta$，而是让网络输出一个校正项 $g(t,x)$，并通过时间窗权重 $m(t)$ 硬性嵌入初值：

$$
\theta(t,x)=\theta_0(x) + m(t)\,g(t,x), \qquad m(t)=\Big(\tfrac{t}{T}\Big)^2.
$$

* $\theta_0(x)$：扰动前的潮流稳态角（示例中设置为从 1→N 的线性降角，代表故障前有潮流）。
* 当 $t\to 0$ 时，$m(t)\to 0$，自动满足 $\theta(0,x)=\theta_0(x)$；这也保证**波未到达前频率即不是随便的“50 Hz 常值”，而是与给定潮流一致的稳态**。

---

## 结果产物

运行脚本后生成：

* `pinn_timeseries.png`：静态时间序列（各母线频率）
* `pinn_timeseries.gif`：时间演化动图（带扰动时刻标记）
* `pinn_space_profile.gif`：固定时刻的空间散点（频率 vs 节点）
* `fno_vs_numerical.png`：与 RK4 数值解对照（含 MAE/RMSE）

---

## 代码结构与解析

### 1) 关键参数与网格

```python
N = 10; H = 3.0; D = 0.0
X_line = 0.1; U = 1.0; fb = 50.0
K = U*U/X_line         # 线路等值耦合强度
Pload_base = 0.5; dP = 0.5; event_bus = 0
t_step = 0.1           # 扰动时刻
T_plot_end = 1.0       # 可视化时窗
T_train_end = 1.1      # 训练时窗（略宽于可视化）
T = 440                # 时间离散点数
```

* **N**：节点数（示例 10）。
* **H, D**：惯量与阻尼（均可扩展为逐节点）。
* **K**：线路等值功角系数。
* **扰动**：在 `event_bus` 处、`t_step` 时刻将负荷加一个阶跃 `dP`。
* **时间/空间网格**：`t_grid`、`x_grid`。

### 2) 初始潮流与机械功率 $P_m$

```python
theta0_np = np.linspace(0.1, 0.01, N).astype(np.float32)  # 预故障潮流角
theta0 = torch.tensor(theta0_np, device=device)

with torch.no_grad():
    Pload0 = torch.full((N,), Pload_base, device=device)
    Pm = pe_from_theta(theta0, Pload0)  # 以初值解出 Pm，用作摆动方程的机械功率
```

* 先用初始角 $\theta_0$ 与静态负荷 $P_{load,0}$ 求一次 $P_e$，令其等于 $P_m$，保证扰动前满足平衡。

### 3) 负荷随时间的阶跃

```python
def make_Pload_time():
    P = torch.full((T, N), Pload_base, device=device)
    mask = (t_grid >= t_step).float().view(T, 1)
    P[:, event_bus] = P[:, event_bus] + mask.squeeze(-1)*dP
    return P
Pload_t = make_Pload_time()
```

* 形成时间维度上的 $P_{load}(t)$，仅在故障母线加阶跃。

### 4) FNO2d（频域卷积的时空模型）

**为何用 FNO？**

* 频域卷积适于捕捉**长程依赖**与**平滑时空结构**，在 PDE 型任务上比常规 CNN/MLP 更稳。
* 不以离散步进方式预测，而是学习**整体时空映射** $(t,x)\to g(t,x)$。

**结构要点**：

* 输入通道：$[t\_\text{norm}, x\_\text{norm}, \text{step\_mask}, \text{event\_mask}, 1]$
* 多层 SpectralConv2d（$\text{rfft2}$ → 频域乘权 → $\text{irfft2}$）
* 残差型 1×1 卷积支路 + GELU
* 输出：校正项 $g(t,x)$，再经硬初值得到 $\theta(t,x)$

### 5) 硬初值嵌入（Hard IC）

```python
m_time = (t_grid/T_train_end).view(1, T, 1) ** 2
theta0_b = theta0.view(1, 1, N)
def theta_from_raw_g(raw_g): return theta0_b + m_time * raw_g
```

* $m(t)$ 二次时间窗，确保 $t=0$ 瞬间强制回到 $\theta_0$，**避免初值漂移**，训练更稳。

### 6) 物理残差（摆动方程）

```python
def pde_loss(theta_pred):
    dth_dt, d2th_dt2, th_mid = finite_diff_time(theta_pred, dt)    # 中心差分
    df  = dth_dt/(2*math.pi); ddf = d2th_dt2/(2*math.pi)
    Pe  = chain_Pe_theta(th_mid, Pload_t[1:-1, :])                 # 中时刻 Pe
    swing = ddf - (fb/(2*H))*(Pm.view(1,1,-1) - Pe - D*df)
    return (swing**2).mean()
```

* 在时间维度上做二阶/一阶差分近似。
* 将 $\theta$ → $P_e(\theta)$，装配成摆动方程残差 `swing`。
* **无标签训练**：只靠残差下降来逼近物理解。

### 7) 评估与可视化

* 基于 $\theta$ 的时间差分得到 $f=\dot\theta/(2\pi)$，叠加 $f_b$ 得到**实际频率**。
* 与 **RK4** 数值解对照，输出 **MAE/RMSE**。
* 生成四种图像/GIF，帮助直观看到**扰动注入、传播、衰减**。

---

## 使用方法

### 环境

* Python 3.9+
* PyTorch、NumPy、Matplotlib

### 运行

1. 根据需要在脚本顶部修改：

   * **规模**：`N`（节点数）、`T`（时间点数）、`T_plot_end/T_train_end`
   * **物理量**：`H, D, X_line, U, Pload_base, dP, event_bus, t_step`
   * **网络**：`width, modes_t, modes_x, nlayers, EPOCHS`
2. `python your_script.py`
3. 在查看输出图/GIF。

---

## 常见问题（FAQ）

* **Q：波未到达前频率不是 50 Hz，如何保证初值正确？**
  **A**：我们把**潮流稳态角 $\theta_0$** 直接嵌入并通过 $m(t)$ 作硬约束，$t\to 0$ 时 $\theta\to \theta_0$。这等效于把“有潮流的稳态”作为边界起点，避免“零潮流”或“50 Hz 常值”假设带来的偏差。

* **Q：后段频率不降反升/边界约束失效？**
  **A**：优先检查三点：
  (1) **时间差分分辨率**（`T` 是否过小）；
  (2) **modes\_t/modes\_x** 是否太小（FNO 频域带宽不够会“漏”高阶；适当增加 `modes_t`/`modes_x` 与 `width`）；
  (3) **PDE 残差权重**已是唯一训练信号，必要时可加**轻微 IC 项**（代码已有 `w_ic_tiny`）稳定早期训练。

* **Q：是否要把初值当作硬性边界条件？**
  **A**：已通过 $m(t)$ 做到了“硬嵌入”（Hard IC）。相比 L2 惩罚，这更稳，并能保证零时刻严格满足潮流前状态。

---

## 与数值解对照

`fno_vs_numerical.png` 采用相同的扰动设置，用 RK4 积分作为“数值参考”。图中**虚线**为数值解，**实线**为 FNO-PINN 预测，图例只区分线型（避免节点过多造成图例冗余），并给出全局 MAE/RMSE（单位 mHz）。

---

## 代码片段（关键模块一览）

**(1) FNO2d 主干**

```python
class FNO2d(nn.Module):
    def __init__(self, in_ch, out_ch=1, width=64, modes_t=32, modes_x=8, nlayers=4):
        super().__init__()
        self.fc0  = nn.Linear(in_ch, width)
        self.convs = nn.ModuleList([SpectralConv2d(width, width, modes_t, modes_x) for _ in range(nlayers)])
        self.ws    = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(nlayers)])
        self.act   = nn.GELU()
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_ch)

    def forward(self, x):
        x = x.permute(0,3,1,2)                              # (B,C,T,X)
        x = self.fc0(x.permute(0,2,3,1)).permute(0,3,1,2)   # -> width
        for conv, w in zip(self.convs, self.ws):
            x = self.act(conv(x) + w(x))                    # 频域核 + 1×1 残差
        x = x.permute(0,2,3,1)                              # (B,T,X,width)
        x = self.act(self.fc1(x))
        x = self.fc2(x)                                     # (B,T,X,1)
        return x.squeeze(-1)                                # -> g(t,x)
```

**(2) 硬初值**

```python
m_time = (t_grid/T_train_end).view(1, T, 1) ** 2
theta0_b = theta0.view(1, 1, N)
def theta_from_raw_g(raw_g):
    return theta0_b + m_time * raw_g
```

**(3) 摆动方程残差**

```python
def pde_loss(theta_pred):
    dth_dt, d2th_dt2, th_mid = finite_diff_time(theta_pred, dt)
    df  = dth_dt/(2*math.pi); ddf = d2th_dt2/(2*math.pi)
    Pe  = chain_Pe_theta(th_mid, Pload_t[1:-1, :])
    swing = ddf - (fb/(2*H))*(Pm.view(1,1,-1) - Pe - D*df)
    return (swing**2).mean()
```

---

## 适配到更大系统

* **节点数**：只需改 `N` 与初值 `theta0_np`（或由潮流解产生）。
* **网络带宽**：同步增大 `width / modes_t / modes_x / nlayers`；
* **训练时窗**：增大 `T_train_end` 与 `T`（更长历史 + 更细时间差分）。



1. `pinn_timeseries.png`（PINN-FNO 时序预测）
2. `fno_vs_numerical.png`（数值对比）
3. `pinn_timeseries.gif`（时域传播动画）
4. `pinn_space_profile.gif`（空间传播动画）

并且保持学术化风格，方便直接放到 README：

---

## 结果与讨论

本研究基于 **FNO（Fourier Neural Operator）** 的物理信息建模框架，针对 10 节点电网系统的频率扰动传播进行了时空预测。为验证模型效果，我们给出数值解与 PINN-FNO 预测的对比结果，并通过动画刻画扰动的动态传播过程。

---

### 1. PINN-FNO 时序预测

![Timeseries PINN](https://www.science42.tech/cases/caseMarkdown/电网频率扰动传播时空建模分析/viz/pinn_timeseries.png)

图中展示了 10 节点系统在 0–1 s 内的 PINN-FNO 预测频率曲线。结果表明：

* 模型能够准确捕捉各节点的动态轨迹；
* 扰动发生后（0.1 s），频率偏离额定 50 Hz，并伴随小幅振荡；
* 系统逐步趋于稳定，体现了电网的惯性与耦合特征。

---

### 2. 数值解对比验证

![FNO vs Numerical](https://www.science42.tech/cases/caseMarkdown/电网频率扰动传播时空建模分析/viz/fno_vs_numerical.png)

图中彩色实线为 **PINN-FNO 预测结果**，虚线为 **数值积分解**。可以看到：

* 两者在扰动后的响应趋势高度一致；
* 模型预测误差仅为毫赫兹量级（MAE ≈ 0.84 mHz, RMSE ≈ 1.11 mHz）；
* 证明了 PINN-FNO 在电网动力学建模中的物理一致性与高精度。

---

### 3. 时域传播动画

![Time Series Propagation](https://www.science42.tech/cases/caseMarkdown/电网频率扰动传播时空建模分析/viz/pinn_timeseries.gif)

该动图展示了扰动在时间维度上的传播过程：

* 扰动施加后，节点频率依次偏离 50 Hz；
* 各节点响应呈现振荡衰减特征；
* 动态演示了 **扰动由局部逐步扩散至全网** 的传播机制。

---

### 4. 空间传播动画

![Space Profile](https://www.science42.tech/cases/caseMarkdown/电网频率扰动传播时空建模分析/viz/pinn_space_profile.gif)

该动画展示了扰动在空间维度上的分布：

* 扰动首先出现在 1 号节点；
* 随后沿网络逐步向远端节点传播；
* 幅值随距离衰减，直观体现了电力网络中的 **耦合与波动传输效应**。

---


## 平台使用示例

1. 进入 **硒钼Science**，点击聊天框上方 **方程求解** 图标（红框）

   <!-- 截图占位：后续替换为真实图片 -->
   ![](https://www.science42.tech/cases/caseMarkdown/电网频率扰动传播时空建模分析/viz/figure1.png)


2. 提问与回答示例

   在对话框中输入 **"能源系统电力仿真"** 或类似问题  

   ![](https://www.science42.tech/cases/caseMarkdown/电网频率扰动传播时空建模分析/viz/figure2.png)  

3. 模型会先进行分析和规划，然后给出完整的 PINN 建模实现示例的 Python 代码  

   ![](https://www.science42.tech/cases/caseMarkdown/电网频率扰动传播时空建模分析/viz/figure3.png)  

4. 查看结果

   - 前端将自动展示并可下载以下可视化与指标：  
     - **时域传播动画**  
     - **空间传播动画**  

 
    ![](https://www.science42.tech/cases/caseMarkdown/电网频率扰动传播时空建模分析/viz/figure4.png) 






### 总结

通过数值对比与动画可视化，PINN-FNO 模型在 **时域** 和 **空间域** 均展现出较强的拟合精度与物理合理性。其对扰动传播方向性、能量衰减规律的刻画，为 **电力系统动态稳定性分析与快速预测** 提供了有力工具。

---






## 许可证与致谢

**贡献者**：

* Di Liu

**许可证**：本文档及配套代码版权归 **硒钼科技（北京）** 所有。更多信息参见 Science42 平台。


