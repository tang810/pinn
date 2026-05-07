<!-- 
text: "地形强迫降雨PINN模拟",
area: "基于物理信息神经网络(PINN)模拟北京地区受地形强迫影响的降雨过程，结合Advection-Diffusion方程与地形源项。",
tags: ["PINN", "降雨模拟", "地形强迫", "北京"],
search: ["Orographic Rainfall", "Physics-Informed Neural Network", "Terrain Forcing"]
-->

# 地形强迫降雨预测


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

```python
import os
import math
import numpy as np              # 数值计算基础库
import matplotlib.pyplot as plt # 绘图基础库
import geopandas as gpd         # 读取 GeoJSON 地理数据 ('北京.json')
from shapely.geometry import Point # 处理地理点坐标
import torch                    # PyTorch 深度学习框架
import torch.nn as nn           # 神经网络模块
import torch.nn.functional as F # 激活函数 (如 ReLU, Tanh)
```

*   **GeoPandas & Shapely**: 用于这一行 `gdf = gpd.read_file(GEOJSON_FILE)`，读取北京的地理边界，用于绘图时的掩膜 (Masking)。
*   **PyTorch**: 核心计算引擎，利用 `torch.autograd` 实现自动微分，这是计算 PDE 残差的关键。

## 2. 地形强迫降雨预测 (Orographic Rainfall)

### 2.1 问题背景
在复杂地形下，气象预报的难点在于**地形雨**。当湿润气流遇到山脉阻挡被迫抬升时，空气冷却凝结，往往在迎风坡形成强降水。本案例通过 PINN 将地形坡度信息直接引入物理方程。

### 2.2 物理模型与地形模拟

*   **地形模拟**: 叠加多个二维高斯函数来模拟北京周边的**西山**和**军都山/燕山**，并添加一个整体倾斜的基底来模拟平原。
*   **控制方程**: 修正的对流-扩散-反应方程。
* **物理原理**: 使用二维高斯函数 (Gaussian Function) 模拟山峰。西山和燕山分别由两个不同位置和宽度的高斯函数表示。



这是地形模拟代码

```python
# 模拟地形高度 H(x,y). 返回值范围 [0, 1]
def get_terrain_tensor(x, y):
    # 1. 西山 (West Mountains): 中心在 (-0.6, 0.2)，较陡峭
    h1 = 0.8 * torch.exp(-((x + 0.6)**2 + (y - 0.2)**2) / 0.3)
    
    # 2. 军都山/燕山 (North Mountains): 中心在 (0.2, 0.8)，范围较广
    h2 = 0.7 * torch.exp(-((x - 0.2)**2 + (y - 0.8)**2) / 0.4)
    
    # 3. 基底 (Base): 模拟从西北向东南倾斜的平原地形
    base = 0.2 * (1.0 - (x + y) / 2.0)
    
    # 叠加并缩放
    h_total = h1 + h2 + base
    return h_total * 0.5 
```

### 2.3 地形强迫源项 ($S_{terrain}$)

#### 控制方程 (Governing Equation)

我们求解的是修正的**对流-扩散-反应方程**：
$$
\frac{\partial R}{\partial t} + \mathbf{u} \cdot \nabla R - D \nabla^2 R = S_{total}
$$
其中 $S_{total} = S_{storm} + S_{terrain}$。

#### 地形强迫源项 (Orographic Source Term)

地形雨的物理本质是：**湿润气流 $\mathbf{u}$ 爬升地形梯度 $\nabla H$**。
*   当风 **顺着山坡吹上去** ($\mathbf{u} \cdot \nabla H > 0$) -> 空气抬升冷凝 -> **降雨增加**。
*   当风 **顺着山坡吹下去** ($\mathbf{u} \cdot \nabla H < 0$) -> 空气下沉增温 -> **降雨抑制**。

**代码实现 (`rainfall_pinn_terrain.py` -> `pde_residual_terrain`)**:

首先，利用 PyTorch 的自动微分 (`autograd`) 计算地形梯度：

```python
    # 获取地形高度 H
    H = get_terrain_tensor(x, y)
    
    # 自动微分求 dH/dx, dH/dy
    # create_graph=True 允许对导数再次求导 (二阶导)
    grad_h = torch.autograd.grad(H, x_yt, torch.ones_like(H), create_graph=True)[0]
    H_x = grad_h[:, 0:1] # 地形在 x 方向的坡度
    H_y = grad_h[:, 1:2] # 地形在 y 方向的坡度
```

然后，计算地形因子并施加 ReLU 激活函数：

```python
    # 计算地形抬升因子: V dot Grad(H) = u*Hx + v*Hy
    # WIND_U, WIND_V 是定义的风速常数 (0.6, 0.6)
    orographic_factor = WIND_U * H_x + WIND_V * H_y
    
    # 关键点: F.relu 确保只有抬升气流 (结果>0) 才产生降水源
    # 如果结果<0 (下坡风)，源项为 0，这符合物理常识（背风坡通常无雨）
    S_terrain = 5.0 * F.relu(orographic_factor) 
```

#### 完整的 PDE 残差 calculation

```python
    # 左边 (LHS): R_t + (uRx + vRy) - D(Rxx + Ryy)
    lhs = R_t + (WIND_U * R_x + WIND_V * R_y) - DIFFUSION_D * (R_xx + R_yy)
    
    # 右边 (RHS): 移动的风暴源 + 地形强迫源
    rhs = S_storm + S_terrain
    
    # 返回残差，训练目标是使其趋近于 0
    return lhs - rhs
```


地形抬升导致的降水增强与风速和地形坡度成正比：

$$
S_{terrain} = \alpha \cdot \text{ReLU}(\mathbf{u} \cdot \nabla H) = \alpha \cdot \text{ReLU}(u \frac{\partial H}{\partial x} + v \frac{\partial H}{\partial y})
$$

*   **物理意义**:
    *   $\mathbf{u} \cdot \nabla H > 0$ (风吹向山坡) $\rightarrow$ 抬升 $\rightarrow$ **增强降雨**。
    *   $\mathbf{u} \cdot \nabla H < 0$ (风吹下山坡) $\rightarrow$ 下沉 $\rightarrow$ **抑制降雨** (雨影区)。


```python
    # 获取地形高度 H
    H = get_terrain_tensor(x, y)
    
    # 计算地形梯度 (dH/dx, dH/dy)
    # 注意: 这里需要再次对 H 求导，或者手动推导 H 的解析导数。
    # 为简单起见，利用 autograd
    grad_h = torch.autograd.grad(H, x_yt, torch.ones_like(H), create_graph=True)[0]
    H_x = grad_h[:, 0:1]
    H_y = grad_h[:, 1:2]

    # 计算地形抬升源项 (Orographic Lifting)
    # 公式: V dot Grad(H) = u*Hx + v*Hy
    # 只有当风吹向山坡(值为正)时才产生降水增强
    orographic_factor = WIND_U * H_x + WIND_V * H_y
    S_terrain = 5.0 * F.relu(orographic_factor) # 强度系数 5.0
```

### 2.4代码实现

在 PINN 中，我们不仅对降雨场 $R$ 求导，同样对地形场 $H$ 求导以获得坡度信息。

```python
# 1. 获取地形高度 H
H = get_terrain_tensor(x, y)

# 2. 自动计算地形梯度 (dH/dx, dH/dy)
grad_h = torch.autograd.grad(H, x_yt, torch.ones_like(H), create_graph=True)[0]
H_x = grad_h[:, 0:1]
H_y = grad_h[:, 1:2]

# 3. 计算地形强迫因子 (Wind dot Gradient)
orographic_factor = WIND_U * H_x + WIND_V * H_y
S_terrain = 5.0 * F.relu(orographic_factor) 
```

完整的PDE代码如下

```python
# ---------------------------
# PDE 残差 (含地形强迫)
# ---------------------------
def pde_residual_terrain(model, x_yt):
    """
    修正后的 PDE: R_t + Advection - Diffusion = S_storm + S_terrain
    """
    x_yt.requires_grad_(True)
    R = model(x_yt)

    # 1. 计算降雨场 R 的导数
    grads = torch.autograd.grad(R, x_yt, torch.ones_like(R), create_graph=True)[0]
    R_x, R_y, R_t = grads[:, 0:1], grads[:, 1:2], grads[:, 2:3]
    
    grads_x = torch.autograd.grad(R_x, x_yt, torch.ones_like(R_x), create_graph=True)[0]
    R_xx = grads_x[:, 0:1]
    
    grads_y = torch.autograd.grad(R_y, x_yt, torch.ones_like(R_y), create_graph=True)[0]
    R_yy = grads_y[:, 1:2]

    # 2. 计算地形相关的项
    x = x_yt[:, 0:1]
    y = x_yt[:, 1:2]
    t = x_yt[:, 2:3]
    
    # 获取地形高度 H
    H = get_terrain_tensor(x, y)
    
    # 计算地形梯度 (dH/dx, dH/dy)
    # 注意: 这里需要再次对 H 求导，或者手动推导 H 的解析导数。
    # 为简单起见，利用 autograd
    grad_h = torch.autograd.grad(H, x_yt, torch.ones_like(H), create_graph=True)[0]
    H_x = grad_h[:, 0:1]
    H_y = grad_h[:, 1:2]

    # 3. 计算地形抬升源项 (Orographic Lifting)
    # 公式: V dot Grad(H) = u*Hx + v*Hy
    # 只有当风吹向山坡(值为正)时才产生降水增强
    orographic_factor = WIND_U * H_x + WIND_V * H_y
    S_terrain = 5.0 * F.relu(orographic_factor) # 强度系数 5.0

    # 4. 动态暴雨源项 (移动的云团)
    center_x = -0.8 + 0.8 * t 
    center_y = -0.8 + 0.8 * t
    S_storm = 8.0 * torch.exp(-((x - center_x)**2 + (y - center_y)**2) / 0.08)

    # 5. 总 PDE 残差
    # R_t + (uRx + vRy) - D(Rxx + Ryy) - (S_storm + S_terrain) = 0
    lhs = R_t + (WIND_U * R_x + WIND_V * R_y) - DIFFUSION_D * (R_xx + R_yy)
    rhs = S_storm + S_terrain
    
    return lhs - rhs
```

## 3. 神经网络架构 (Network Architecture)

使用一个全连接神经网络 `STPinn` (Spatio-Temporal PINN) 来拟合降雨场 $R(x, y, t)$。

```python
class STPinn(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        
        for m in self.layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
        x = self.layers[-1](x)
        return x
```
*   **输入**: 3维向量 $(x, y, t)$。
*   **输出**: 1维标量 Rainfall Intensity $R$。

## 4. 训练策略 (Training Strategy)


训练不依赖标签数据，而是最小化物理残差。

```python
        # 1. 在时空域内随机采样点
        loss_pde = torch.mean(pde_residual_terrain(model, sample_spatiotemporal(2000)) ** 2)
        
        # 2. 初始条件 (Initial Condition): t=0 时无雨
        x_ic, y_ic = sample_initial_condition(500)
        loss_ic = torch.mean((model(x_ic) - y_ic) ** 2)
        
        # 3. 边界条件 (Boundary Condition): 区域边缘无雨 (Dirichlet BC)
        x_bc, y_bc = sample_boundary_condition(500)
        loss_bc = torch.mean((model(x_bc) - y_bc) ** 2)
        
        # 总损失: 加权求和
        loss = 2.0 * loss_pde + 5.0 * loss_ic + 5.0 * loss_bc
```


## 5. 结果可视化

### 5.1 静态图

这段代码展示了如何在一个图上叠加多层信息：地形、降雨、矢量场。

```python
    # 图层 1: 地形 (半透明背景)
    terrain_plot = ax.imshow(terrain_masked, ..., cmap='gist_earth', alpha=0.5)
    
    # 图层 2: 降雨 (等高线填充)
    # alpha=0.6 保证能透过降雨看到底下的山脉
    rain_contour = ax.contourf(Lon, Lat, pred_rain_masked, ..., cmap='jet', alpha=0.6)
    
    # 图层 3: 风场矢量 (Quiver)
    # 稀疏采样 (step=20) 避免箭头过密
    ax.quiver(q_lon, q_lat, q_u, q_v, ...)
```

该时刻展示了模型捕捉到的结构：
*   **亮色背景**: 高海拔山脉。
*   **彩色区域**: 降雨强度。降雨中心与山脉迎风坡高度重合，而背风坡降雨较弱，完美复现了“地形雨”特征。
*   **白色箭头**: 西南风场矢量。

![地形降雨信息图](https://www.science42.tech/cases/caseMarkdown/case_5a73f515/viz/rain_terrain_info_t_70.png)


### 5.2 动态 GIF 生成 (`create_rainfall_gif.py`)

为了生成动画，我们需要在每一帧更新降雨预测。

```python
    def update(frame):
        # 1. 计算当前时间 t
        t_val = frame / 20.0
        
        # 2. 模型预测当前时刻的降雨分布
        # ... model(x_in) ...
        
        # 3. 清除旧画面并重绘
        ax.clear()
        
        # 4. 重新绘制地形 (静态背景)
        ax.imshow(terrain, ...)
        
        # 5. 绘制当前时刻的降雨
        if np.any(valid_mask):
            ax.contourf(Lon, Lat, pred, ...)
```

可以观察到降雨团随风移动。当雨团经过高地形区域时，受 $S_{terrain}$ 项驱动，降雨强度自动增加。

![降雨演变动画](https://www.science42.tech/cases/caseMarkdown/case_5a73f515/viz/rainfall_highres.gif)


---

## 6. 总结与扩展

### 总结
本项目展示了 PINN 在气象领域的强大潜力：
1.  **物理融合**: 能够将流体力学与热力学方程无缝嵌入神经网络。
2.  **自动发现**: 无需大量标签数据，模型通过物理约束“自动发现”了地形对降雨的调制作用。
3.  **复杂边界处理**: 成功处理了非规则地形边界和特定物理条件的相互作用。

### 扩展方向
*   **真实数据同化**: 引入气象站观测数据 (Station Data)，在 Loss 中增加数据项 ($Loss_{Data}$)。
*   **多变量耦合**: 同时求解气压、温度、湿度等多个物理场，构建更完整的微物理过程。
*   **3D 地形**: 扩展到三维空间，使用真实 DEM 高程数据。
