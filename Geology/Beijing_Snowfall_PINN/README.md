<!-- 
text: "北京地区降雪PINN模拟",
area: "使用物理信息神经网络模拟北京地区的降雪过程，考虑冷锋移动、地形抬升及温度融化机制。",
tags: ["PINN", "降雪模拟", "相变", "北京"],
search: ["Snowfall Simulation", "Beijing Weather", "Phase Change", "Cold Front"]
-->

# 北京降雪动力学模拟


## 1. 运行环境与依赖 (Environment & Dependencies)

本项目基于 Python 和 PyTorch 构建，主要依赖库如下：

*   **Python**: 3.8+
*   **PyTorch**: 深度学习框架，用于构建神经网络及自动微分 (Autograd)。
*   **NumPy**: 数值计算。
*   **Matplotlib**: 数据可视化及动画生成。
*   **GeoPandas & Shapely**: 处理地理空间数据 (GeoJSON) 及边界掩膜。

```bash
pip install torch numpy matplotlib geopandas shapely
```

导入 Python 库
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
```

操作系统：Linux / Windows / macOS 均可（示例环境为 Linux + CUDA）

---

## 北京降雪动力学模拟

### 2.1 问题背景
该案例旨在模拟受复杂物理机制控制的降雪过程。模型不仅要捕捉随风的平流运动，还需要处理**地形效应**（山脉抬升）、**风暴移动**（天气尺度系统）以及**温度动力学**（冷锋过境与融化）。核心思想是训练一个神经网络 $S(x, y, t)$ 来近似**雪密度场**。

### 2.2 物理参数与模型配置 (Parameters & Physics)

*   **风场 (`WIND_U`, `WIND_V`)**: 设置为西北风 ($U>0, V<0$)，作为驱动雪/风暴移动的主要**平流 (Advection)** 力。
*   **扩散 (`DIFFUSION_D`)**: 表示雪密度随时间**扩散**的程度。
*   **地形模型**: 使用合成的高斯函数近似北京地貌：
    *   **西北山区**: 使用高斯函数建模，$Height \approx 0.8 \cdot \exp(-r^2)$。
    *   **东南平原**: 地势平坦。
    *   地形高度 $H(x,y)$ 决定了温度分布和地形抬升强度。
*   **温度模型**:
    *   **冷锋**: 随时间移动的降温项，模拟冷空气南下。
    *   **递减率 (Lapse Rate)**: 温度随高度 $H$ 线性降低。
    *   **冰点线**: $T=0$ 等值线区分固态雪 ($T<0$) 与融化 ($T>0$)。


设定了一股强劲的**西北风**（从西北吹向东南），所以 $U>0, V<0$

```python
# 物理参数 (归一化后的数值)
WIND_U = 0.5      # 西北风的 U 分量 (向东吹)
WIND_V = -0.5     # 西北风的 V 分量 (向南吹)
DIFFUSION_D = 0.02 # 扩散系数：雪散开的速度
MELT_RATE = 8.0   # 融化速率：温度高时雪消失多快
```

北京的地形是“西北高，东南低”。用数学函数造一个地形：

```python
# 地形函数 (输入的 x, y 坐标已归一化到 [-1, 1] 范围)
def get_terrain_tensor(x, y):
    """
    模拟北京地形：
    西北高（燕山/太行山脉），东南低（平原）。
    返回 [0, 1] 范围内的高度值 H。
    """
    # 1. 主要的西北山脉
    # 沿西北边界的高斯山脊
    # 旋转坐标系以对齐东北-西南走向的山脉轴线
    # x_rot = (x - y) / sqrt(2)
    h_mountains = 0.8 * torch.exp(-((x + 0.5)**2 + (y - 0.5)**2) / 0.5)
    
    # 2. 特定山峰（灵山/太行山延伸）- 西部
    h_peak1 = 0.6 * torch.exp(-((x + 0.8)**2 + (y - 0.2)**2) / 0.1)
    
    # 3. 北部山脉（怀柔/密云）
    h_peak2 = 0.5 * torch.exp(-((x - 0.2)**2 + (y - 0.8)**2) / 0.15)
    
    # 4. 基础坡度（西北高 -> 东南低）
    # x=-1, y=1 (西北) -> 最高点
    # x=1, y=-1 (东南) -> 最低点
    h_slope = 0.3 * ( (1 - x) + (1 + y) ) / 4.0 
    
    raw_h = h_mountains + h_peak1 + h_peak2 + h_slope
    return torch.clamp(raw_h, 0, 1.2)
```

雪会不会化，取决于温度。如果算出来的 $T > 0$，雪会化；$T < 0$，雪能存在。

```python
def get_temperature_tensor(x, y, t, H):
    base_temp_initial = 0.5 
    cooling = 1.5 * t       # 随着时间推移，冷锋过境，气温大幅下降
    altitude_cooling = LAPSE_RATE * H # 爬山越导致气温越低
    
    T = base_temp_initial - cooling - altitude_cooling
    return T
```


### 2.3 控制方程 (PDE)

我们使用**平流-扩散-反应方程**作为 PINN 的核心物理约束：

$$
\frac{\partial S}{\partial t} + \vec{u} \cdot \nabla S - D \nabla^2 S = Q_{total}
$$

其中 $Q_{total} = Q_{storm} + Q_{oro} - Q_{melt}$：

1.  **$Q_{storm}$ (风暴源)**: 移动的高斯分布，代表天气尺度系统。
2.  **$Q_{oro}$ (地形抬升)**: $\alpha \cdot \text{ReLU}(\vec{u} \cdot \nabla H)$。当风吹向山坡（即 $\vec{u} \cdot \nabla H > 0$）时，强制产生降雪。
3.  **$Q_{melt}$ (融化项)**: $\beta \cdot \text{ReLU}(T) \cdot S$。当局部温度 $T > 0$ 时，雪密度成比例衰减。

### 2.4 PINN 实现方法

#### 网络架构
使用标准的全连接神经网络 (MLP)：
*   **输入**: $(x, y, t)$ 3维时空坐标。
*   **隐藏层**: 4 层，每层 60 个神经元。
*   **激活函数**: `Tanh` (隐藏层)，`Softplus` (输出层，保证雪密度 $S \ge 0$)。


```python
class SnowfallPinn(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入: x, y, t (你在哪里？现在几点？)
        # 输出: 1 个数值 (这里有多少雪？)
        self.net = nn.Sequential(
            nn.Linear(3, 60), nn.Tanh(),
            nn.Linear(60, 60), nn.Tanh(),
            nn.Linear(60, 60), nn.Tanh(),
            nn.Linear(60, 60), nn.Tanh(),
            nn.Linear(60, 1) # 输出层
        )

    def forward(self, x):
        # F.softplus 就像一个过滤器，把负数变成 0。
        # 因为“雪的厚度”不可能是负数。
        return F.softplus(self.net(x)) 
```

*   **解释**: 这是一个简单的“多层感知机” (MLP)。
*   **输入**: 给它一个坐标 $(x, y)$ 和时间 $t$。
*   **输出**: 它猜一个雪的密度 $S$。最开始它是瞎猜的，但我们会用物理方程通过 `Loss` 来“惩罚”它，让它越猜越准。



#### PDE 嵌入实现 (Code Analysis)

这一步是 PINN 的灵魂。我们要计算“神经网络猜的结果，符不符合物理规律？”如果不符合，`residual`（残差）就会很大。

利用 PyTorch 的 `autograd` 自动计算偏导数：


```python
def pde_residual(model, x_yt):
    x_yt.requires_grad_(True) # 告诉 PyTorch：我们要对这些坐标求导数！
    S = model(x_yt)           # 让 AI 预测现在的雪量 S
    
    # --- 1. 求导数 ---
    # 求 S 对 (x, y, t) 的一阶导数：雪随时间变了吗(S_t)？空间上陡峭吗(S_x, S_y)？
    grads = torch.autograd.grad(S, x_yt, torch.ones_like(S), create_graph=True)[0]
    S_x, S_y, S_t = grads[:, 0:1], grads[:, 1:2], grads[:, 2:3]
    
    # 求二阶导数：用于计算"扩散" (S_xx, S_yy)
    grads_x = torch.autograd.grad(S_x, x_yt, torch.ones_like(S_x), create_graph=True)[0]
    S_xx = grads_x[:, 0:1]
    ...
    
    # --- 2. 计算物理源项 ---
    # a. 地形抬升 (Orographic Uplift)
    # 算出地形的坡度 (H_x, H_y)
    grad_h = torch.autograd.grad(H, x_yt, ...)[0]
    
    # 风 (WIND) 撞在坡度 (H_grad) 上 => 产生抬升 => 造雪
    uplift = WIND_U * H_x + WIND_V * H_y
    Q_oro = 3.0 * F.relu(uplift) # 只有迎风坡(uplift>0)才造雪
    
    # b. 风暴源 (Q_storm)
    # 一个随时间 t 移动的高斯圆斑
    cx = -0.5 + WIND_U * t 
    Q_storm = 5.0 * torch.exp(...) 
    
    # c. 融化 (Q_melt)
    Q_melt = MELT_RATE * F.relu(T) * S # 温度大于0就融化
    
    # --- 3. 组装方程 (Advection-Diffusion-Reaction) ---
    # 左边 (变化与运动) = 右边 (源项)
    lhs = S_t + (WIND_U * S_x + WIND_V * S_y) - DIFFUSION_D * (S_xx + S_yy)
    rhs = Q_storm + Q_oro - Q_melt
    
    # 返回不平衡的量 (Residual)。如果不为0，说明还没学会物理！
    return lhs - rhs
```

### 2.5 训练策略 (Loss & Sampling)

**Loss 函数**: 加权均方误差 (MSE) 之和。
$$ Loss = 2.0 \cdot Loss_{PDE} + 10.0 \cdot Loss_{IC} + 5.0 \cdot Loss_{BC} $$

*   **$Loss_{PDE}$**: 域内随机采样，强制满足物理方程。权重较低，但采样点最多。
*   **$Loss_{IC}$**: 初始 ($t=0$) 雪密度为 0。权重最高，确保初始状态正确。
*   **$Loss_{BC}$**: 边界雪密度为 0。

**采样策略**: 每一轮训练 (Iteration) 均使用`Monto Carlo`方法重新随机采样，防止网络记忆特定坐标点。


```python
# ---------------------------
#  训练
# ---------------------------
def train():
    model = SnowfallPinn().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 用于微调的学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    
    ITERATIONS = 5000
    print(f"开始训练，共 {ITERATIONS} 次迭代...")
    
    start_time = time.time()
    
    for it in range(1, ITERATIONS + 1):
        optimizer.zero_grad()
        
        x_pde, x_ic, y_ic, x_bc, y_bc = get_batch(4000, 1000, 1000)
        
        res = pde_residual(model, x_pde)
        loss_pde = torch.mean(res**2)
        
        pred_ic = model(x_ic)
        loss_ic = torch.mean((pred_ic - y_ic)**2)
        
        pred_bc = model(x_bc)
        loss_bc = torch.mean((pred_bc - y_bc)**2)
        
        loss = 2.0 * loss_pde + 10.0 * loss_ic + 5.0 * loss_bc
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if it % 500 == 0:
            elapsed = time.time() - start_time
            print(f"迭代 {it} | 总损失: {loss.item():.5f} | 偏微分方程损失: {loss_pde.item():.5f} | 初始条件损失: {loss_ic.item():.5f} | 时间: {elapsed:.1f}s")
            
    torch.save(model.state_dict(), "model_snow_pinn.pt")
    print("模型已保存至 model_snow_pinn.pt")
    return model

```

### 2.6 结果可视化

GeoJSON 是一种基于 JSON（JavaScript Object Notation） 的地理空间数据交换格式。它使用 JSON 的语法和结构来编码地理数据结构，如点、线、面（多边形）以及这些几何图形的集合。

简单来说，GeoJSON 就是一种用人类和机器都易于阅读的文本格式来描述地理信息（例如“一个点在哪里”、“一条路由哪些点组成”、“一个区域的边界是什么”）的标准化方法。

一个标准的 GeoJSON 对象通常包含以下几种核心类型：

* Geometry（几何对象）：描述地理形状。
* Feature（要素）：一个几何对象加上与其相关的属性。这是最常用的 GeoJSON 对象。
* FeatureCollection（要素集合）：一组 Feature 对象的集合。

一个完整的 GeoJSON 文件大概如下格式：

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": { "name": "北京", "population": 2154 },
      "geometry": {
        "type": "Point",
        "coordinates": [116.4074, 39.9042]
      }
    },
    {
      "type": "Feature",
      "properties": { "name": "上海", "population": 2428 },
      "geometry": {
        "type": "Point",
        "coordinates": [121.4737, 31.2304]
      }
    }
  ]
}

```

#### 可视化函数

我们要画图，就得把地图分成很多小格子（像素点），逐个看“这儿有多少雪？”


```python

# ---------------------------
# 可视化
# ---------------------------
def visualize_snow(model, t_val=0.8):
    print(f"正在为 t={t_val} 进行可视化...")
    
    # 尝试加载 GeoJSON 文件
    try:
        if os.path.exists(GEOJSON_FILE):
            gdf = gpd.read_file(GEOJSON_FILE)
            minx, miny, maxx, maxy = gdf.total_bounds
        else:
            raise FileNotFoundError
    except:
        print("未找到 GeoJSON 文件，使用通用边界框。")
        # 大致的北京坐标
        minx, miny, maxx, maxy = 115.4, 39.4, 117.5, 41.1
        gdf = None

    # 创建网格
    N = 300
    x = np.linspace(minx, maxx, N)
    y = np.linspace(miny, maxy, N)
    X, Y = np.meshgrid(x, y)
    
    # 将 X, Y 归一化到 [-1, 1] 范围以供模型查询
    X_norm = 2.0 * (X - minx) / (maxx - minx) - 1.0
    Y_norm = 2.0 * (Y - miny) / (maxy - miny) - 1.0
    
    # 展平数组
    inp_x = X_norm.flatten()
    inp_y = Y_norm.flatten()
    inp_t = np.full_like(inp_x, t_val)
    
    inp_tensor = torch.tensor(np.stack([inp_x, inp_y, inp_t], axis=1), dtype=torch.float32).to(DEVICE)
    
    # 模型预测
    with torch.no_grad():
        # 雪深
        S_pred = model(inp_tensor).cpu().numpy().flatten()
        
        # 辅助物理量：地形和温度
        # 为 numpy/CPU 重新手动计算
        t_ten = torch.tensor([t_val]).to(DEVICE)
        # 我们需要对 H 和 Temp 进行逐元素计算
        # 在 `inp_tensor` 坐标上通过张量操作更快，但如果可能，我们重用 Physics 函数
        # 需要重塑 inp_tensor 的列
        tx = inp_tensor[:, 0:1]
        ty = inp_tensor[:, 1:2]
        H_pred = get_terrain_tensor(tx, ty).cpu().numpy().flatten()
        Temp_pred = get_temperature_tensor(tx, ty, t_val, torch.tensor(H_pred).to(DEVICE).unsqueeze(1)).cpu().numpy().flatten()

    # 重塑为网格
    S_grid = S_pred.reshape(N, N)
    H_grid = H_pred.reshape(N, N)
    T_grid = Temp_pred.reshape(N, N)
    
    # 遮罩处理
    if gdf is not None:
        # 创建遮罩
        # 这可能很慢，简单的边界框裁剪已经完成
        # 为了速度，我们直接在地图上绘图而不对像素进行硬遮罩，
        # 或者如果边界复杂，使用简单的边界检查
        pass

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 1. 背景：地形（灰色）
    # 使用 'gist_earth' 或 'terrain' 色彩映射
    ax.imshow(H_grid, extent=[minx, maxx, miny, maxy], origin='lower', 
              cmap='terrain', alpha=0.5, vmin=-0.1, vmax=1.5)
    
    # 2. 主要数据：积雪（白色/蓝色）
    # S_grid 通常 > 0
    if np.max(S_grid) > 0.01:
        # 创建一个从透明到白色再到蓝色的自定义色彩映射
        # 或者直接使用带透明度的 'Blues'
        # 我们使用 `PuBu` (紫-蓝) 或 `Blues`
        levels = np.linspace(0.01, np.max(S_grid), 40)
        cs = ax.contourf(X, Y, S_grid, levels=levels, cmap='PuBu', alpha=0.9, extend='max')
        cbar = plt.colorbar(cs, ax=ax, fraction=0.035, pad=0.04)
        cbar.set_label('模拟雪深 (归一化)', rotation=270, labelpad=15)
    else:
        print("警告：检测到少量或无积雪。")

    # 3. 物理叠加：冻结线（温度 = 0）
    # T_grid = 0 的等高线
    cs_temp = ax.contour(X, Y, T_grid, levels=[0], colors='red', linewidths=2.5, linestyles='--')
    ax.clabel(cs_temp, inline=True, fmt='冻结线 (0°C)', fontsize=10)
    
    # 4. 地理边界
    if gdf is not None:
        gdf.boundary.plot(ax=ax, color='black', linewidth=1.5, alpha=0.6)
        
    ax.set_title(f"北京降雪模拟 (物理信息神经网络)\n时间 = {t_val:.2f} (归一化) | 风向: 西北", fontsize=15)
    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")
    
    # 添加风向箭头
    ax.arrow(115.6, 40.8, 0.2, -0.2, head_width=0.05, head_length=0.08, fc='k', ec='k', label='Wind')
    ax.text(115.6, 40.85, "风向 (西北)", fontsize=12)

    plt.tight_layout()
    save_path = f"snow_sim_beijing_t{int(t_val*100)}.png"
    plt.savefig(save_path, dpi=300)
    print(f"图表已保存至 {save_path}")
    # plt.show() # 在无头环境中已禁用

```


![](https://www.science42.tech/cases/caseMarkdown/case_5a73f515/viz/snow_sim_beijing_t80.png)

#### 为了展示更清楚，可以制作gif动画


```python
def create_snow_gif(model, filename='...'):
    # ...
    # 创建网格 (Grid)
    x_grid = np.linspace(minx, maxx, N_GRID) 
    y_grid = np.linspace(miny, maxy, N_GRID)
    X, Y = np.meshgrid(x_grid, y_grid) # 生成所有像素点的坐标矩阵
    
    # 归一化：把经纬度变成 [-1, 1]，因为模型只吃这个范围的数据
    X_norm = ... 
    
    # 准备 GeoJSON 掩膜：
    # 我们只想看北京里面的，不想看外面的。
    # 使用 shapely 库判断每个点是不是在多边形里 (contains)。
    mask_flat = np.array([map_boundary.contains(p) for p in points])
```


这是制作 GIF 动画的关键函数：

```python
    def update(frame):
        # 1. 算出当前时间 t
        t_val = (frame / total_frames) * T_MAX
        
        # 2. 让模型预测现在的雪 (S) 和温度 (T)
        pred_s = model(x_in)...
        
        # 3. 涂上“隐形药水” (Masking)
        # 把北京外面的数据设为 NaN (Not a Number)，matplotlib 就不会画这部分
        pred_s[~mask] = np.nan 
        
        ax.clear() # 清空上一帧
        
        # 4. 画底图 (地形)
        ax.imshow(H_grid, cmap='terrain', ...)
        
        # 5. 画雪 (Snow)
        # 使用 contourf (等高线填充)
        # alpha=0.8 表示半透明，能透出下面的地形
        if np.nanmax(pred_s) > 0.01:
            ax.contourf(X, Y, pred_s, cmap='PuBu', ...)
            
        # 6. 画冰点线 (Freezing Line, T=0)
        # 这是一条警戒线，红色的虚线
        ax.contour(X, Y, pred_t, levels=[0], colors='red', linestyles='--')
```

动画展示了以下过程：
1.  **掩膜**: 使用 `北京.json` 掩盖城市外区域。
2.  **可视化**: 蓝色填充表示雪密度，灰色背景为地形。
3.  **动态**: 红色虚线表示 $0^\circ C$ 线随冷锋南移；雪团经过山区时明显增强。

![北京降雪模拟](https://www.science42.tech/cases/caseMarkdown/case_5a73f515/viz/snowfall_pinn_beijing.gif)

