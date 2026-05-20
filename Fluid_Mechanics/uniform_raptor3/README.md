# Raptor 3 发动机反应流求解器

SpaceX Raptor 3 CH4/LOX 推进系统：准一维等熵喷管 + 解析羽流模型。输出三区结构化网格 VTK，供 ParaView 可视化。

---

## 1. 环境准备 & 运行

```bash
# 创建 conda 环境（Python 3.10+）
conda create -n raptor python=3.10 -y

# 激活环境
conda activate raptor

# 安装依赖
conda install numpy scipy matplotlib cantera vtk pillow -c conda-forge -y
```

```bash
# 运行
conda activate raptor
python main.py
```

输出文件在 `results/` 下，`raptor3_full.vtm` 拖入 ParaView 即可查看。

---

## 2. 配置

```python
# config.py
class Config:
    # 发动机
    P_CHAMBER = 35e6           # 350 bar
    T_CHAMBER = 3550.0         # 火焰温度 [K]
    P_AMBIENT = 101325.0       # 海平面大气压
    M_CH4 = 170.0              # 甲烷质量流量 [kg/s]
    M_LOX = 630.0              # 液氧质量流量 [kg/s]
    M_TOTAL = 800.0
    OF_RATIO = 630.0 / 170.0   # ≈ 3.71（富燃状态）

    # 气体
    GAMMA = 1.18               # 常比热比（硬编码模式）
    R      = 380.0             # 气体常数 [J/(kg·K)]

    # Cantera 模式
    USE_CANTERA = False        # True → 变比热比 + 刚性 ODE
    GAMMA_ITER_MAX = 5         # γ-流动耦合最大迭代次数
    GAMMA_ITER_TOL = 1e-3

    # 网格
    NX_INT   = 200             # 喷管轴向
    NX_EXT   = 300             # 羽流轴向
    L_PLUME  = 15.0            # 羽流域长度 [m]
    R_PLUME  = 5.0             # 羽流径向范围 [m]
    NY_CORE  = 60              # 核心区径向 (S2+S3)
    NY_OUTER = 40              # 外层径向 (S4)
```

---

## 3. 壁面几何 & 喉部校准

20 个控制点定义的拉瓦尔喷管轮廓，用三次样条插值为 `r = wall_spline(x)`。

```python
# geometry.py
_BASE_WALL = np.array([
    [0.00, 0.650], [0.20, 0.650], [0.40, 0.650], [0.60, 0.650],
    [0.75, 0.620], [0.85, 0.560], [0.92, 0.480], [1.00, 0.380],  # ← 收缩至喉部
    [1.08, 0.310], [1.12, 0.300],                                 # ← 喉部最小截面
    [1.18, 0.320], [1.28, 0.375], [1.42, 0.470], [1.60, 0.600],
    [1.80, 0.750], [2.05, 0.940], [2.35, 1.140], [2.70, 1.340],
    [3.10, 1.540], [3.50, 1.750],                                 # ← 出口
])
```

**校准**：由目标质量流量反推喉部面积 `A*`，全局缩放半径使等熵质量流量公式匹配：

```python
def init_geometry(cfg):
    A_star_target = (
        cfg.M_TOTAL /
        (cfg.P_CHAMBER * np.sqrt(cfg.GAMMA / (cfg.R * cfg.T_CHAMBER)) *
         (2.0 / (cfg.GAMMA + 1.0)) ** ((cfg.GAMMA + 1.0) / (2.0 * (cfg.GAMMA - 1.0))))
    )
    r_throat_base = np.min(_BASE_WALL[:, 1])
    _R_SCALE = np.sqrt(A_star_target / (np.pi * r_throat_base ** 2))
    WALL_PTS[:, 1] *= _R_SCALE                       # 缩放
    wall_spline = CubicSpline(WALL_PTS[:, 0], WALL_PTS[:, 1])
```

---

## 4. 准一维等熵喷管

核心：给定面积分布 `A(x)`，反求马赫数 `M(x)`，等熵递推 `T, P, ρ, u`。

### 4.1 网格与面积

```python
class Quasi1DNozzle:
    def __init__(self, cfg):
        self.xc = np.linspace(0.01, L_ENGINE, cfg.NX_INT)  # 轴向
        r_wall = wall_spline(self.xc)
        self.A = np.pi * r_wall ** 2                        # 面积分布
        self.A_star = self.A.min()                          # 喉部面积
        self.A_ratio = np.maximum(self.A / self.A_star, 1.0)
```

### 4.2 马赫数求解

等熵面积比关系——每个网格点用 `fsolve` 反求 M：

```
A/A* = (1/M) × [ 2/(γ+1) × (1 + (γ-1)/2 × M²) ] ^ ((γ+1) / (2(γ-1)))
```

```python
def _solve_mach(self, subsonic_mask, gamma):
    M = np.zeros(self.nx)
    for i in range(self.nx):
        ar_i = self.A_ratio[i]
        if subsonic_mask[i]:                              # 收缩段：亚声速分支
            guess = [max(0.05, 1.0 / ar_i)]
            sol = fsolve(lambda m: ar(m, gamma) - ar_i, guess)
            M[i] = min(float(sol[0]), 0.99)
        else:                                              # 扩张段：超声速分支
            guess = [max(1.5, ar_i ** 0.5)]
            sol = fsolve(lambda m: ar(m, gamma) - ar_i, guess)
            M[i] = max(float(sol[0]), 1.01)
    return M
```

### 4.3 等熵递推

```python
# 有了 M(x)，利用等熵关系式 → T, P, ρ, u
T_ratio = 1.0 / (1.0 + (gamma - 1.0) / 2.0 * M ** 2)
T   = T_CHAMBER * T_ratio                                   # 温度
P   = P_CHAMBER * T_ratio ** (gamma / (gamma - 1.0))        # 压力
rho = P / (R * T)                                           # 密度
u   = M * np.sqrt(gamma * R * T)                            # 轴向速度
```

---

## 5. 化学反应

### 5.1 硬编码模式（默认，< 1s）

全局单步反应 `CH4 + 2 O2 → CO2 + 2 H2O`，追踪 `CH4, O2, CO2, H2O, N2` 五种组分。

```python
def _compute_species_hardcoded(self):
    x_comb = self.xc[int(np.argmin(self.A) * 0.7)]        # 燃烧起始位置
    stoich_O2_per_CH4 = 4.0
    Y_CH4_reactable = Y_CH4_in * (Y_O2_in / stoich_O2_per_CH4)  # 可燃烧 CH4

    for i in range(nx):
        if self.xc[i] < x_comb:                           # 上游：反应指数渐变
            frac = (tanh((T[i] - 1200) / 400) * 0.5 + 0.5)
            frac = frac * (xc[i] / x_comb) ** 2
        else:                                              # 下游：反应完全
            frac = 1.0

        Y[i, 0] = Y_CH4_in - frac * Y_CH4_reactable        # CH4
        Y[i, 1] = Y_O2_in  - frac * Y_O2_in                # O2
        Y[i, 2] = frac * Y_CH4_reactable * (44/16)         # CO2
        Y[i, 3] = frac * Y_CH4_reactable * (36/16)         # H2O
        Y[i, 4] = 1.0 - sum(Y[i, :4])                      # N2（填充）
```

O/F=3.71 > 化学计量比 4.0，属富燃——过剩 CH4 不燃尽。

### 5.2 Cantera 模式（~10-30s）

```python
def _solve_cantera(self):
    # Step 1: 硬编码解作为初始场
    # Step 2: 逐单元 IdealGasReactor 刚性 ODE 积分 (gri30.yaml, 53组分)
    for iteration in range(5):         # 最多 5 轮 γ-流动耦合
        gamma_new, R_new = self._cantera_refine(gas)   # 每个 cell: ODE + γ=cp/cv
        if np.max(|gamma_new - gamma_old|) < 1e-3:
            break
    # Step 3: 收敛 γ 最终流场
```

---

## 6. 解析羽流模型

计算域：x ∈ [3.5, 18.5] m，三区矩形网格。

```
r=5 ┌────────────────┬──────────────────────────────┐
    │                │                              │
    │   S2（壁面跟随） │        S4（羽流外层）          │
    │   0 → f(x)     │        r_exit → 5            │
    ├── f(x)─────────┼────── r = f(3.5) ────────────┤
    │                │                              │
    │   S2（壁面跟随） │        S3（羽流核心）          │
    │   0 → f(x)     │        0 → r_exit            │
r=0 └────────────────┴──────────────────────────────┘
   x=0            x=3.5                          x=18.5
```

### 6.1 网格生成

轴向与喷管出口 dx 对接，几何级数拉伸；径向双区——核心 tanh 聚类、外层均匀：

```python
class PlumeModel:
    def __init__(self, cfg, nozzle):
        # 轴向：二分法求拉伸比 r，使 Σ dx_int·r^i = L_PLUME
        self.xc = _geometric_stretch(dx_int, L_PLUME)
        # 径向 core: eta = tanh(stretch * (2t-1)) / tanh(stretch)  → 壁面加密
        # 径向 outer: 均匀 [0, 1]
        self.yc = eta_core × r_exit (core)  /  eta_outer × (5 - r_exit) (outer)
```

### 6.2 马赫盘

```python
def _compute_plume_with_phase(self, phase_shift):
    L_cell = 0.8 * D_exit * sqrt(M² - 1) * (P_exit/P_amb) ^ (1/(2γ))   # 胞间距

    for i in range(nx):
        # 波幅从 0 渐变至额定值（避免出口处跳变）
        ramp = min(1.0, x_rel / (0.55 * D_exit))

        wave     = sin(2π·x_rel / L_cell + phase)
        envelope = exp(-x_rel / L_decay)
        amplitude = 0.25 * envelope * ramp       # T,P 振幅 25%, 随衰减

        T_cl = T_exit * (1 + amplitude * wave)               # 中心线温度
        P_cl = P_exit * (1 + amplitude * 1.5 * wave)          # 中心线压力
        M_cl = M_exit * (1 - amplitude * 0.8 * wave)          # 中心线马赫数
        u_cl = M_cl * sqrt(γ·R·T_cl)
```

### 6.3 径向过渡剖面

出口平面为硬阶跃（top-hat），向下游平滑过渡到 S 型势流核心轮廓。`fade` 控制过渡速度，`width_ramp` 使径向过渡宽度也从 0 渐变起始：

```python
fade = min(1.0, x_rel / x_blend)
width_ramp = min(1.0, x_rel / (0.55 * D_exit))

if fade < 0.005 and width_ramp < 0.01:          # 极近出口：极窄阶跃
    width = r_exit * 0.015 * width_ramp
else:                                           # 下游：渐变展宽
    width = r_exit * (0.02 + 0.5 * fade) * width_ramp

if width < r_exit * 0.015:                      # 宽度太小时退化为硬阶跃
    weight = 1.0 if r < r_exit else 0.0
else:                                           # S 型势流核心
    weight = 1.0 / (1 + exp((r - r_exit) / (width * 0.3)))

# 所有物理量按权重插值到环境值
T[i,j] = T_ambient + (T_cl - T_ambient) * weight
P[i,j] = P_ambient + (P_cl - P_ambient) * weight
Mach[i,j] = M_cl * weight
u[i,j]   = u_cl * weight
```

### 6.4 组分混合

出口组分向环境空气（23%O₂, 77%N₂）按同一权重混合：

```python
def _compute_species(self):
    ambient = [0.0, 0.23, 0.0, 0.0, 0.77]  # CH4,O2,CO2,H2O,N2
    for i in range(nx):
        for j in range(ny):
            weight = _compute_weight(x_rel, r_j)     # 同上径向权重
            Y[i,j,:] = ambient + (Y_exit - ambient) * weight
```

---

## 7. 推力 & 比冲

```python
def thrust(self):
    rho_e, u_e, p_e, A_e = rho[-1], u[-1], P[-1], A[-1]
    mdot  = rho_e * u_e * A_e                    # 出口质量流量
    F_mom = mdot * u_e                           # 动量推力
    F_pre = (p_e - P_AMBIENT) * A_e               # 压力推力
    return F_mom + F_pre                          # 总推力

def isp(self):
    F, _, _, mdot = self.thrust()
    return F / (mdot * 9.81)                      # 海平面比冲
```

**结果与 Raptor 3 参考值对比：**

| 参数 | 计算结果 | Raptor 3 参考值 |
|------|----------|-----------------|
| 燃烧室压力 | 35.0 MPa | ~35 MPa |
| 燃烧室温度 | 3550 K | ~3500–3800 K |
| O/F 比 | 3.71 | ~3.6 |
| 膨胀比 | 34:1 | ~34:1 |
| 出口马赫数 | 4.02 | ~3.8–4.2 |
| 出口速度 | 3236 m/s | ~3300–3400 m/s |
| 海平面推力 | 2.61 MN (266吨) | ~2.46–2.80 MN |
| 海平面比冲 | 329 s | ~327–350 s |

---

## 8. VTK 输出

三区多块结构化网格，写入 `.vtm` + 三个 `.vts` 子文件。优先用 `vtk` 库，不可用时降级为纯 Python XML。

```python
# 标量场（每个 cell）
# density, pressure, temperature, Mach, u, v,
# Y_CH4, Y_O2, Y_CO2, Y_H2O, Y_N2, reaction_rate

# S2: nx_int × ny_core, 壁面跟随, x∈[0, 3.5], r∈[0, f(x)]
# S3: nx_ext × ny_core, 矩形,      x∈[3.5, 18.5], r∈[0, r_exit]
# S4: nx_ext × ny_outer, 矩形,     x∈[3.5, 18.5], r∈[r_exit, 5]
```

三区间边界精确对齐：S2⇔S3 同径向 η 分布逐点匹配；S3⇔S4 同轴向网格 `r=r_exit` 水平线。

---

## 9. 后处理

```python
# postprocess.py
create_summary_plots(cfg, nozzle)     # 四面板: M沿程, T+P双轴, 组分演化, 指标汇总
create_thrust_curve(cfg, nozzle)      # 推力累积曲线 + 动量/压力分量饼图
create_gif_from_vtk(cfg)             # 20 帧马赫盘相位扫描 → 温度+马赫数 GIF
```

| 文件 | 内容 |
|------|------|
| `results/raptor3_full.vtm` | 三区 VTK 多块文件 |
| `results/raptor3_full_????.vtm` | 20 帧动画序列 |
| `results/raptor3_summary.png` | 性能面板 |
| `results/raptor3_thrust.png` | 推力曲线 |
| `results/raptor3_plume.gif` | 羽流 GIF 动画 |

---

## 10. 完整执行流程

```python
from config import Config
from geometry import init_geometry
from nozzle import Quasi1DNozzle
from plume import PlumeModel
from vtk_writer_vtk import VTKWriter
from postprocess import (create_summary_plots, create_thrust_curve, create_gif_from_vtk)

cfg = Config()
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# [1] 几何校准：缩放射径使 A* 匹配目标质量流量
init_geometry(cfg)

# [2] 准一维喷管：M(x), T(x), P(x), u(x), Y(x)
nozzle = Quasi1DNozzle(cfg)
nozzle.solve()

# [3] 解析羽流：三区 2D 场 + 马赫盘
plume = PlumeModel(cfg, nozzle, nx_plume=cfg.NX_EXT, ny=cfg.NY)

# [4] VTK 输出
vtk_writer = VTKWriter(cfg, nozzle, plume)
vtk_writer.write('results/raptor3_full.vtm', frame=0)

# [5] 20 帧动画（相位扫描 0 → 4π）
for frame in range(20):
    phase = frame * pi / 10
    plume._compute_plume_with_phase(phase)    # 重置流场
    plume._compute_species()                   # 重置组分
    vtk_writer.write(f'results/raptor3_full_{frame:04d}.vtm', frame)

# [6] 后处理
create_summary_plots(cfg, nozzle)
create_thrust_curve(cfg, nozzle)
create_gif_from_vtk(cfg)
```

---

## 关键设计决策

1. **准一维喷管**：推力/比冲由燃烧室参数和面积比主导，一维等熵即可精确捕捉。完整 CFD 仅增 ~5% 修正量，计算成本却差 3–4 个数量级。

2. **解析羽流**：超声速射流马赫盘可用线化波理论很好描述。欧拉求解在同网格上需 10³–10⁴ 倍迭代，精度提升甚微。

3. **三区网格**：壁面跟随区 + 矩形羽流区分离，避免整体结构化网格的"阶梯状"表达。

4. **马赫盘渐变起始**：波幅和径向过渡宽度均从 0 ramp 起始（`x_rel/(0.55D)`），消除出口平面的物理不连续。

5. **Cantera 可选**：常比热比模式 < 1s；变比热比 + 刚性 ODE 模式 ~10-30s 提供更高保真度。

---

## 项目文件

```
uniform_raptor3/
├── main.py                # 主入口
├── README.md              # 说明文档
├── requirements.txt       # 依赖列表
├── format.json            # 项目结构化描述文件
├── src/
│   ├── __init__.py
│   ├── config.py          # 工况、网格、化学参数
│   ├── geometry.py        # 壁面轮廓、样条插值、喉部校准
│   ├── nozzle.py          # 准一维等熵喷管 + CH4/O2 燃烧
│   ├── plume.py           # 解析羽流: 马赫盘、射流扩散、组分混合
│   ├── vtk_writer.py      # 纯 Python VTM/VTS XML 写入器（备选）
│   ├── vtk_writer_vtk.py  # VTK Python 模块写入器（首选）
│   ├── postprocess.py     # 汇总图、推力曲线、GIF 动画
│   └── grid_check.py      # 网格检查工具
├── data/
│   ├── raptor_mech.yaml   # Cantera 简化反应机理
│   └── Raptor3.jpeg       # 发动机结构图
├── model/                 # 模型/中间件输出预留
└── results/               # 输出: VTM, PNG, GIF
```
