<!-- 
README.md文件名统一大写
 -->


<!-- 
text: "在此填写该项目的简要标题，例如：“三维弹性力学应力与位移预测”",

area: "在此填写本项目研究的背景与意义，简要介绍研究问题、关键模型（如所用方程）、解决目标与技术方案。支持使用 <br/> 进行换行。例如：准确预测固体结构在受力情况下的内部应力与变形，对于工程结构优化设计至关重要。项目采用 **物理信息神经网络（PINN）** 方法，求解 XYZ 方程，预测目标物理量分布。 内容不要太长",

tags: [ "此处填写标签 3-5个，如：三维弹性力学", "应力应变预测", "Navier-Cauchy方程", "结构分析"  ]
-->




<!-- 项目名称为一级header ： # -->
# 地质元素异常图生成系统 
<!-- 项目概述：二级Header ：## -->
## 项目概述

GeoChatBot 地质元素异常图生成器是一款基于插值算法的地质数据可视化工具，支持通过反距离权重法（IDW）和克里金插值法生成地质元素的空间分布异常图。核心功能包括：
- 对稀疏地质元素数据（如 Cu、Au 等）进行空间插值
- 生成交互式地图可视化异常分布
- 支持地图测距、定位及区域绘制分析

### 解决的核心问题：
地质勘探中稀疏采样点的空间分布预测，通过插值算法实现元素浓度的连续场可视化，辅助地质异常识别。

### 预期输出：
- 元素浓度分布云图（IDW / 克里金）
- 交互式地图（支持缩放、测距、定位）

## 问题背景
### 场景实例化

在地质勘探领域，准确了解地下地质元素的空间分布情况对于矿产资源的勘查、地质灾害的预测以及环境评估等工作至关重要。然而，由于地质勘探工作的复杂性和高成本，实际获取的地质元素数据往往是稀疏的，仅来自有限数量的采样点。这些稀疏的采样数据难以全面、准确地反映地质元素在整个区域内的连续分布特征，给地质研究和决策带来了巨大挑战。此外，现有的地质数据可视化工具往往缺乏交互性，难以满足地质工作者在实际工作中对数据进行深入分析和探索的需求。地质工作者需要一种能够生成交互式地图的工具，通过缩放、测距、定位等功能，直观地观察地质元素的分布异常情况，从而更好地进行地质分析和决策。

|![pytorchES - Real and Imaginary](viz/12321.png)|
|:--:|
| **Fig.1** 场景图  |

GeoChatBot 地质元素异常图生成器正是为了解决上述问题而开发的。它利用先进的插值算法，能够对稀疏的地质元素数据进行空间插值，生成地质元素的连续场可视化异常图。同时，该工具还提供了交互式地图功能，支持地质工作者进行缩放、测距、定位等操作，大大提高了地质数据的分析效率和准确性。




## 环境与依赖:代码需要配置的环境： 把你代码的import部分复制 问大模型 ，按照下列的格式自动生成

<!-- 1.Python版本和需要的库 -->
- Python ≥3.8
- streamlit==1.23.1
- streamlit-chat==0.0.2
- langchain==0.1.0
- numpy==1.26.0
- pandas==2.1.1
- scipy==1.11.1
- folium==0.14.0
- scikit-learn==1.3.2
- streamlit-folium==0.12.3

<!-- 保留 -->
以上依赖可通过 `requirements.txt` 一键安装：


<!-- 
```bash
这个框里的是终端运行的代码格式
``` -->

```bash
pip install -r requirements.txt
```

## 快速安装

<!-- 
这里只需要替换成你对应的项目名称/python版本即可- or 大模型一键生成 -->
创建虚拟环境（推荐）
```bash
# 使用conda创建虚拟环境
conda create -n GeoAnomalyMap python=3.9
conda activate GeoAnomalyMap

# 或者使用venv（Python内置模块）
python3.9 -m venv GeoAnomalyMap
source GeoAnomalyMap/bin/activate  # Linux/MacOS
.\GeoAnomalyMap\Scripts\activate  # Windows
```

<!-- 
代码单元 统一为 4级Header: ####
格式按照: 代码单元+空格+序号(1,2,3)# TopoZeko：基于 PyTorch 的地形可视化工具

## 项目概述
TopoZeko 是一个专门用于地形可视化的工具，主要解决了冰川建模、地貌模拟等科学可视化领域中的核心问题，即如何直观、准确地展示地形数据。该项目旨在将床底高程和表面高程等地形数据进行可视化，支持 2D 热力图、3D 地形图以及 4D 厚度叠加图形展示，预期输出结果为直观展示地形特征的可视化图像，帮助科研人员更好地理解和分析地形数据。

| ![](./viz/topozeko_plot.png) |
|:--:|
| **Fig.1** 三维地形图可视化示例 |

## 环境与依赖
- Python ≥3.8
- PyTorch ≥1.12
- matplotlib
- h5py

以上依赖可通过 `requirements.txt` 一键安装：
```bash
pip install -r requirements.txt
```

## 快速安装
```bash
# 建议使用虚拟环境
conda create -n TopoZeko python=3.9
conda activate TopoZeko
pip install -r requirements.txt
```

#### 代码单元 1：导入必要库&设置设备
```python
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
import h5py

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
```

## 数学背景
### 物理约束
1. **厚度计算**  
   厚度 `THI` 通过表面高程 `SUR` 减去床底高程 `BED` 计算得出：
   $$
   THI = SUR - BED
   $$
2. **异常值处理**  
   当 `THI` 超出指定范围 `caxis` 时，将其裁剪至该范围内：
   - 若 $THI > caxis_{max}$，则 $THI = caxis_{max}$
   - 若 $THI < caxis_{min}$，则 $THI = caxis_{min}$

#### 物理方程参数说明：
| 参数 | 含义 | 默认值 |
|:--:|:--:|:--:|
| `BED` | 床底高程矩阵 | - |
| `SUR` | 表面高程矩阵 | - |
| `THI` | 厚度矩阵 | - |
| `caxis` | 厚度裁剪范围 | '' |

---

### 厚度计算函数
为了计算地形的厚度，我们构建了一个简单的计算逻辑，将其嵌入到 `TopoZeko` 函数中。

请在 **Jupyter Notebook** 中依次运行以下代码块：
#### 代码单元 2：厚度计算逻辑
```python
if extra_dimension == '' or extra_dimension == 'on':
    THI = SUR - BED
else:
    THI = extra_dimension
```

---

### 异常值处理逻辑
简介功能：
为了确保厚度数据在合理范围内，我们在代码中添加了异常值处理逻辑，将超出指定范围的数据进行裁剪。

#### 异常值裁剪
```math
THI = 
\begin{cases}
caxis_{min}, & \text{if } THI < caxis_{min} \\
caxis_{max}, & \text{if } THI > caxis_{max} \\
THI, & \text{otherwise}
\end{cases}
```

#### 符号说明：
| 符号 | 含义 |
|:--:|:--:|
| `THI` | 厚度矩阵 |
| `caxis_{min}` | 厚度裁剪范围的最小值 |
| `caxis_{max}` | 厚度裁剪范围的最大值 |

> 💡 **说明：**  
> - 当 `extra_dimension` 未指定或为 'on' 时，厚度 `THI` 通过 `SUR - BED` 计算得出。  
> - 当 `THI` 超出 `caxis` 范围时，将其裁剪至该范围内。  

## 网络模型章节：TopoZeko 函数
`TopoZeko` 是一个用于地形可视化的函数，支持 2D 热力图、3D 地形图以及 4D 厚度叠加图形展示。该函数的核心功能包括：
1. **数据预处理**：计算厚度 `THI`，处理异常值，裁剪数据至指定范围。  
2. **绘图分支**：根据用户输入的参数，判断绘制 2D、3D 或 4D 图形。  
3. **渲染设置**：通过 `plot_surface`、`pcolormesh`、`LightSource` 等进行纹理叠加、光照处理。  
4. **交互输出**：自动坐标控制、颜色条设定、图像保存为 PNG。

### 构造 TopoZeko 函数
在这一代码单元中，我们定义了 `TopoZeko` 函数，用于实现地形可视化的核心功能。主要特点包括：
- **可配置参数**：支持超过 40 项配置参数，用户可以根据需求自定义可视化效果。  
- **多模式绘图**：支持 2D 热力图、3D 地形图以及 4D 厚度叠加图形展示。

#### 代码单元 3：定义 TopoZeko 函数
```python
def TopoZeko(BED, SUR, **kwargs):
    # 解析输入参数，设置默认值
    axes = kwargs.get('axes', 'off')
    bed_colors = kwargs.get('bed_colors', 128)
    bed_colormap = kwargs.get('bed_colormap', 'copper')
    bed_colormap_flipud = kwargs.get('bed_colormap_flipud', 'on')
    bed_trans = kwargs.get('bed_trans', 1)
    caxis = kwargs.get('caxis', '')
    cbar_colors = kwargs.get('cbar_colors', 128)
    cbar_position = kwargs.get('cbar_position', 'northoutside')
    cbar_tick_format = kwargs.get('cbar_tick_format', '')
    D2 = kwargs.get('D2', 'off')
    D4_colormap = kwargs.get('D4_colormap', 'jet')
    D4_colormap_flipud = kwargs.get('D4_colormap_flipud', 'off')
    extra_dimension = kwargs.get('extra_dimension', '')
    label_size = kwargs.get('label_size', '')
    light_orientation = kwargs.get('light_orientation', [-90, 45])
    size_cm = kwargs.get('size_cm', [20, 20])
    size_pix = kwargs.get('size_pix', '')
    sur_color = kwargs.get('sur_color', [1, 1, 1])
    sur_material = kwargs.get('sur_material', 'dull')
    sur_trans = kwargs.get('sur_trans', 1)
    tick = kwargs.get('tick', 'on')
    tick_size = kwargs.get('tick_size', 18)
    title = kwargs.get('title', '')
    title_size = kwargs.get('title_size', 22)
    vertical_scaling = kwargs.get('vertical_scaling', 1)
    view_orientation = kwargs.get('view_orientation', [0, 45])
    xlabel = kwargs.get('xlabel', '')
    xlabel_rotation = kwargs.get('xlabel_rotation', 0)
    xlabel_size = kwargs.get('xlabel_size', 18)
    xlim = kwargs.get('xlim', '')
    xvalues = kwargs.get('xvalues', '')
    ylabel = kwargs.get('ylabel', '')
    ylabel_rotation = kwargs.get('ylabel_rotation', 0)
    ylabel_size = kwargs.get('ylabel_size', 18)
    ylim = kwargs.get('ylim', '')
    yvalues = kwargs.get('yvalues', '')
    zlabel = kwargs.get('zlabel', '')
    zlabel_rotation = kwargs.get('zlabel_rotation', 90)
    zlabel_size = kwargs.get('zlabel_size', 18)
    zlim = kwargs.get('zlim', '')

    # 检查BED和SUR的维度是否一致
    if BED.shape != SUR.shape:
        raise ValueError('Error: dimensions of bedrock elevation matrix and surface elevation matrix do not agree')

    if extra_dimension == '' or extra_dimension == 'on':
        THI = SUR - BED
    else:
        THI = extra_dimension

    # 如果厚度全为NaN，设置为0
    if torch.isnan(THI).all():
        THI.fill_(0)

    # 处理4D绘图的标志
    larger_than_flag = 0
    smaller_than_flag = 0

    if extra_dimension == '' or caxis == '' or caxis == 'off':
        THI_MIN = torch.min(THI)
        THI_MAX = torch.max(THI)
    else:
        THI_MIN = caxis[0]
        THI_MAX = caxis[1]
        i = torch.where(THI > THI_MAX)
        THI[i] = THI_MAX
        if len(i[0]) > 0:
            larger_than_flag = 1
        i = torch.where(THI < THI_MIN)
        THI[i] = THI_MIN
        if len(i[0]) > 0:
            smaller_than_flag = 1

    THI_DIF = THI_MAX - THI_MIN
    BED_MIN = torch.min(BED)
    BED_MAX = torch.max(BED)

    if extra_dimension == '' or extra_dimension == 'on':
        i = torch.where(THI < 0)
        if len(i[0]) > 0:
            raise ValueError('Error: bedrock elevation exceeds surface elevation')
    i = torch.where(THI == 0)
    THI[i] = float('nan')
    mask = torch.full(THI.shape, float('nan'))
    i = torch.where(THI > 0)
    mask[i] = float('-inf')

    a = BED.shape
    if xvalues == '':
        x1 = 1
        x2 = a[1]
    else:
        x1 = xvalues[0]
        x2 = xvalues[1]
    if yvalues == '':
        y1 = 1
        y2 = a[0]
    else:
        y1 = yvalues[0]
        y2 = yvalues[1]

    if D2 == 'on':
        # 绘制2D图
        fig, ax = plt.subplots()
        plt.pcolormesh(THI.numpy())
        plt.colorbar()
        plt.contour(THI.numpy(), levels=torch.arange(THI_MIN, THI_MAX, round(THI_DIF / 10)).numpy())
        ax.set_title(title, fontweight='bold', fontsize=16)
        ax.set_xlabel(xlabel, fontweight='bold', fontsize=14)
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=14)
        plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(view_orientation[1], view_orientation[0])
    ax.grid(True)

    if extra_dimension == '':
        # 绘制3D图
        # 修改 torch.meshgrid 调用，添加 indexing='ij' 参数
        X, Y = torch.meshgrid(torch.linspace(x1, x2, a[1]), torch.linspace(y1, y2, a[0]), indexing='ij')
        if bed_colormap_flipud == 'off':
            cmap = plt.get_cmap(bed_colormap)
            colors = [sur_color] + [cmap(i) for i in range(bed_colors)]
            cmap = plt.cm.colors.ListedColormap(colors)
        else:
            cmap = plt.get_cmap(bed_colormap)
            colors = [sur_color] + [cmap(i) for i in range(bed_colors - 1, -1, -1)]
            cmap = plt.cm.colors.ListedColormap(colors)
        # 在 plot_surface 中使用 cmap 参数
        h1 = ax.plot_surface(X.numpy(), Y.numpy(), BED.T.numpy(), alpha=bed_trans, cmap=cmap)
        # 对 plot_surface 返回的对象调用 set_clim 方法
        h1.set_clim(BED_MIN - (1 / bed_colors) * (BED_MAX - BED_MIN), BED_MAX)
        # 使用 color 参数设置单一颜色
        h2 = ax.plot_surface(X.numpy(), Y.numpy(), SUR.T.numpy(), color=sur_color, alpha=sur_trans)

        if zlim != '':
            ax.set_zlim(zlim)
        else:
            if vertical_scaling > 0 and vertical_scaling <= 1:
                ax.set_zlim(BED_MIN, BED_MIN + (torch.max(SUR) - BED_MIN) / vertical_scaling)
            else:
                raise ValueError('Vertical scaling should be between 0 and 1')

    else:
        # 绘制4D图
        # 修改 torch.meshgrid 调用，添加 indexing='ij' 参数
        X, Y = torch.meshgrid(torch.linspace(x1, x2, a[1]), torch.linspace(y1, y2, a[0]), indexing='ij')
        if D4_colormap_flipud == 'off' and bed_colormap_flipud == 'off':
            cmap1 = plt.get_cmap(D4_colormap)
            cmap2 = plt.get_cmap(bed_colormap)
            colors1 = [cmap1(i) for i in range(cbar_colors)]
            colors2 = [cmap2(i) for i in range(bed_colors)]
            colors = colors1 + colors2
            cmap = plt.cm.colors.ListedColormap(colors)
        elif D4_colormap_flipud == 'on' and bed_colormap_flipud == 'off':
            cmap1 = plt.get_cmap(D4_colormap)
            cmap2 = plt.get_cmap(bed_colormap)
            colors1 = [cmap1(i) for i in range(cbar_colors - 1, -1, -1)]
            colors2 = [cmap2(i) for i in range(bed_colors)]
            colors = colors1 + colors2
            cmap = plt.cm.colors.ListedColormap(colors)
        elif D4_colormap_flipud == 'off' and bed_colormap_flipud == 'on':
            cmap1 = plt.get_cmap(D4_colormap)
            cmap2 = plt.get_cmap(bed_colormap)
            colors1 = [cmap1(i) for i in range(cbar_colors)]
            colors2 = [cmap2(i) for i in range(bed_colors - 1, -1, -1)]
            colors = colors1 + colors2
            cmap = plt.cm.colors.ListedColormap(colors)
        elif D4_colormap_flipud == 'on' and bed_colormap_flipud == 'on':
            cmap1 = plt.get_cmap(D4_colormap)
            cmap2 = plt.get_cmap(bed_colormap)
            colors1 = [cmap1(i) for i in range(cbar_colors - 1, -1, -1)]
            colors2 = [cmap2(i) for i in range(bed_colors - 1, -1, -1)]
            colors = colors1 + colors2
            cmap = plt.cm.colors.ListedColormap(colors)
        # 在 plot_surface 中使用 cmap 参数
        h1 = ax.plot_surface(X.numpy(), Y.numpy(), bed_colors + ((BED.T - BED_MIN) / (BED_MAX - BED_MIN)) * bed_colors,
                             alpha=bed_trans, cmap=cmap)
        # 对 plot_surface 返回的对象调用 set_clim 方法
        h1.set_clim(0, cbar_colors + bed_colors)
        cbar = plt.colorbar(orientation=cbar_position)

        if cbar_tick_format == '':
            if THI_DIF > 100:
                tick1 = round(THI_MIN.item())
                tick2 = round((THI_MIN + THI_DIF / 5).item())
                tick3 = round((THI_MIN + 2 * THI_DIF / 5).item())
                tick4 = round((THI_MIN + 3 * THI_DIF / 5).item())
                tick5 = round((THI_MIN + 4 * THI_DIF / 5).item())
                tick6 = round((THI_MIN + THI_DIF).item())
            else:
                factor = 1000 * 10 ** (-int(torch.floor(torch.log10(THI_DIF))))
                tick1 = round(THI_MIN.item() * factor) / factor
                tick2 = round((THI_MIN + THI_DIF / 5).item() * factor) / factor
                tick3 = round((THI_MIN + 2 * THI_DIF / 5).item() * factor) / factor
                tick4 = round((THI_MIN + 3 * THI_DIF / 5).item() * factor) / factor
                tick5 = round((THI_MIN + 4 * THI_DIF / 5).item() * factor) / factor
                tick6 = round((THI_MIN + THI_DIF).item() * factor) / factor
        else:
            tick1 = format(THI_MIN.item(), cbar_tick_format)
            tick2 = format((THI_MIN + THI_DIF / 5).item(), cbar_tick_format)
            tick3 = format((THI_MIN + 2 * THI_DIF / 5).item(), cbar_tick_format)
            tick4 = format((THI_MIN + 3 * THI_DIF / 5).item(), cbar_tick_format)
            tick5 = format((THI_MIN + 4 * THI_DIF / 5).item(), cbar_tick_format)
            tick6 = format((THI_MIN + THI_DIF).item(), cbar_tick_format)

        if smaller_than_flag == 1 and larger_than_flag == 1:
            cbar.set_ticks(
                [0, cbar_colors / 5, 2 * cbar_colors / 5, 3 * cbar_colors / 5, 4 * cbar_colors / 5, cbar_colors])
            cbar.set_ticklabels(['<' + str(tick1), str(tick2), str(tick3), str(tick4), str(tick5), '>' + str(tick6)])
        elif smaller_than_flag == 1:
            cbar.set_ticks(
                [0, cbar_colors / 5, 2 * cbar_colors / 5, 3 * cbar_colors / 5, 4 * cbar_colors / 5, cbar_colors])
            cbar.set_ticklabels(['<' + str(tick1), str(tick2), str(tick3), str(tick4), str(tick5), str(tick6)])
        elif larger_than_flag == 1:
            cbar.set_ticks(
                [0, cbar_colors / 5, 2 * cbar_colors / 5, 3 * cbar_colors / 5, 4 * cbar_colors / 5, cbar_colors])
            cbar.set_ticklabels([str(tick1), str(tick2), str(tick3), str(tick4), str(tick5), '>' + str(tick6)])
        else:
            cbar.set_ticks(
                [0, cbar_colors / 5, 2 * cbar_colors / 5, 3 * cbar_colors / 5, 4 * cbar_colors / 5, cbar_colors])
            cbar.set_ticklabels([str(tick1), str(tick2), str(tick3), str(tick4), str(tick5), str(tick6)])

        max_dif = torch.max(SUR) - BED_MIN
        if torch.max(BED) - BED_MIN > max_dif:
            max_dif = torch.max(BED) - BED_MIN
        bed_dif = torch.max(BED) - BED_MIN

        if zlim != '':
            z_min = cbar_colors + ((zlim[0] - BED_MIN) / bed_dif) * bed_colors
            z_max = cbar_colors + ((zlim[1] - BED_MIN) / bed_dif) * bed_colors
            ax.set_zlim(z_min, z_max)
            tick_dif = z_max - z_min
            ax.set_zticks(
                [z_min, z_min + 0.2 * tick_dif, z_min + 0.4 * tick_dif, z_min + 0.6 * tick_dif, z_min + 0.8 * tick_dif,
                 z_max])
            ax.set_zticklabels([zlim[0], zlim[0] + 0.2 * (zlim[1] - zlim[0]), zlim[0] + 0.4 * (zlim[1] - zlim[0]),
                                zlim[0] + 0.6 * (zlim[1] - zlim[0]), zlim[0] + 0.8 * (zlim[1] - zlim[0]),
                                zlim[0] + (zlim[1] - zlim[0])])
        else:
            if vertical_scaling > 0 and vertical_scaling <= 1:
                z_min = cbar_colors
                z_max = cbar_colors + ((max_dif / bed_dif) * bed_colors) / vertical_scaling
                ax.set_zlim(z_min, z_max)
                tick_dif = z_max - z_min
                ax.set_zticks([z_min, z_min + 0.2 * tick_dif, z_min + 0.4 * tick_dif, z_min + 0.6 * tick_dif,
                               z_min + 0.8 * tick_dif, z_max])
                ax.set_zticklabels(
                    [BED_MIN, BED_MIN + 0.2 * max_dif / vertical_scaling, BED_MIN + 0.4 * max_dif / vertical_scaling,
                     BED_MIN + 0.6 * max_dif / vertical_scaling, BED_MIN + 0.8 * max_dif / vertical_scaling,
                     BED_MIN + max_dif / vertical_scaling])
            else:
                raise ValueError('Vertical scaling should be between 0 and 1')

    # 通用设置
    ax.tick_params(axis='both', labelsize=tick_size)
    ax.set_title(title, fontweight='bold', fontsize=title_size)
    ax.set_xlabel(xlabel, fontweight='bold', fontsize=xlabel_size)
    ax.xaxis.label.set_rotation(xlabel_rotation)
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=ylabel_size)
    ax.yaxis.label.set_rotation(ylabel_rotation)
    ax.set_zlabel(zlabel, fontweight='bold', fontsize=zlabel_size)
    ax.zaxis.label.set_rotation(zlabel_rotation)

    # 使用 LightSource 设置光照效果
    ls = LightSource(azdeg=light_orientation[0], altdeg=light_orientation[1])
    # 可根据需要对绘制的表面应用光照效果
    # 例如对 h1 应用光照效果
    rgb = ls.shade(BED.T.numpy(), cmap=cmap)
    h1 = ax.plot_surface(X.numpy(), Y.numpy(), BED.T.numpy(), facecolors=rgb, alpha=bed_trans)

    ax.set_axisbelow(True)
    ax.set_axis_off() if axes == 'off' else ax.set_axis_on()
    ax.set_xticklabels([]) if tick == 'off' else None
    ax.set_yticklabels([]) if tick == 'off' else None
    ax.set_zticklabels([]) if tick == 'off' else None

    if label_size:
        ax.tick_params(axis='both', labelsize=label_size)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    # 保存图片
    plt.savefig('topozeko_plot.png')
    plt.show()
```

> 💡 **解读：**
> - **功能**：根据输入的床底高程矩阵 `BED` 和表面高程矩阵 `SUR`，以及可选的配置参数，绘制 2D 热力图、3D 地形图或 4D 厚度叠加图形。
> - **参数**：
>   - `BED`：床底高程矩阵（torch.Tensor）
>   - `SUR`：表面高程矩阵（torch.Tensor）
>   - `**kwargs`：可选配置参数，如视角、垂直缩放比例、颜色映射等。
> - **返回值**：无，直接显示并保存可视化图像。

> 🔍 **提示：**
> - 根据需要调整配置参数，以获得不同的可视化效果。  
> - 确保输入的 `BED` 和 `SUR` 矩阵维度一致，否则会抛出 `ValueError`。  

## 模型训练与评估
由于 TopoZeko 是一个可视化工具，不涉及模型训练和评估的过程。

## 结果可视化
### 2D 热力图
当 `D2='on'` 时，将绘制 2D 热力图，展示地形厚度的分布情况。

#### 代码单元 4：绘制 2D 热力图
```python
if D2 == 'on':
    # 绘制2D图
    fig, ax = plt.subplots()
    plt.pcolormesh(THI.numpy())
    plt.colorbar()
    plt.contour(THI.numpy(), levels=torch.arange(THI_MIN, THI_MAX, round(THI_DIF / 10)).numpy())
    ax.set_title(title, fontweight='bold', fontsize=16)
    ax.set_xlabel(xlabel, fontweight='bold', fontsize=14)
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=14)
    plt.show()
```

### 3D 地形图
默认情况下，将绘制 3D 地形图，展示地形的表面和床底。

#### 代码单元 5：绘制 3D 地形图
```python
if extra_dimension == '':
    # 绘制3D图
    # 修改 torch.meshgrid 调用，添加 indexing='ij' 参数
    X, Y = torch.meshgrid(torch.linspace(x1, x2, a[1]), torch.linspace(y1, y2, a[0]), indexing='ij')
    if bed_colormap_flipud == 'off':
        cmap = plt.get_cmap(bed_colormap)
        colors = [sur_color] + [cmap(i) for i in range(bed_colors)]
        cmap = plt.cm.colors.ListedColormap(colors)
    else:
        cmap = plt.get_cmap(bed_colormap)
        colors = [sur_color] + [cmap(i) for i in range(bed_colors - 1, -1, -1)]
        cmap = plt.cm.colors.ListedColormap(colors)
    # 在 plot_surface 中使用 cmap 参数
    h1 = ax.plot_surface(X.numpy(), Y.numpy(), BED.T.numpy(), alpha=bed_trans, cmap=cmap)
    # 对 plot_surface 返回的对象调用 set_clim 方法
    h1.set_clim(BED_MIN - (1 / bed_colors) * (BED_MAX - BED_MIN), BED_MAX)
    # 使用 color 参数设置单一颜色
    h2 = ax.plot_surface(X.numpy(), Y.numpy(), SUR.T.numpy(), color=sur_color, alpha=sur_trans)

    if zlim != '':
        ax.set_zlim(zlim)
    else:
        if vertical_scaling > 0 and vertical_scaling <= 1:
            ax.set_zlim(BED_MIN, BED_MIN + (torch.max(SUR) - BED_MIN) / vertical_scaling)
        else:
            raise ValueError('Vertical scaling should be between 0 and 1')
```

### 4D 厚度叠加图形
当 `extra_dimension` 不为空时，将绘制 4D 厚度叠加图形，将厚度颜色映射叠加到三维图中。

#### 代码单元 6：绘制 4D 厚度叠加图形
```python
else:
    # 绘制4D图
    # 修改 torch.meshgrid 调用，添加 indexing='ij' 参数
    X, Y = torch.meshgrid(torch.linspace(x1, x2, a[1]), torch.linspace(y1, y2, a[0]), indexing='ij')
    if D4_colormap_flipud == 'off' and bed_colormap_flipud == 'off':
        cmap1 = plt.get_cmap(D4_colormap)
        cmap2 = plt.get_cmap(bed_colormap)
        colors1 = [cmap1(i) for i in range(cbar_colors)]
        colors2 = [cmap2(i) for i in range(bed_colors)]
        colors = colors1 + colors2
        cmap = plt.cm.colors.ListedColormap(colors)
    elif D4_colormap_flipud == 'on' and bed_colormap_flipud == 'off':
        cmap1 = plt.get_cmap(D4_colormap)
        cmap2 = plt.get_cmap(bed_colormap)
        colors1 = [cmap1(i) for i in range(cbar_colors - 1, -1, -1)]
        colors2 = [cmap2(i) for i in range(bed_colors)]
        colors = colors1 + colors2
        cmap = plt.cm.colors.ListedColormap(colors)
    elif D4_colormap_flipud == 'off' and bed_colormap_flipud == 'on':
        cmap1 = plt.get_cmap(D4_colormap)
        cmap2 = plt.get_cmap(bed_colormap)
        colors1 = [cmap1(i) for i in range(cbar_colors)]
        colors2 = [cmap2(i) for i in range(bed_colors - 1, -1, -1)]
        colors = colors1 + colors2
        cmap = plt.cm.colors.ListedColormap(colors)
    elif D4_colormap_flipud == 'on' and bed_colormap_flipud == 'on':
        cmap1 = plt.get_cmap(D4_colormap)
        cmap2 = plt.get_cmap(bed_colormap)
        colors1 = [cmap1(i) for i in range(cbar_colors - 1, -1, -1)]
        colors2 = [cmap2(i) for i in range(bed_colors - 1, -1, -1)]
        colors = colors1 + colors2
        cmap = plt.cm.colors.ListedColormap(colors)
    # 在 plot_surface 中使用 cmap 参数
    h1 = ax.plot_surface(X.numpy(), Y.numpy(), bed_colors + ((BED.T - BED_MIN) / (BED_MAX - BED_MIN)) * bed_colors,
                         alpha=bed_trans, cmap=cmap)
    # 对 plot_surface 返回的对象调用 set_clim 方法
    h1.set_clim(0, cbar_colors + bed_colors)
    cbar = plt.colorbar(orientation=cbar_position)

    if cbar_tick_format == '':
        if THI_DIF > 100:
            tick1 = round(THI_MIN.item())
            tick2 = round((THI_MIN + THI_DIF / 5).item())
            tick3 = round((THI_MIN + 2 * THI_DIF / 5).item())
            tick4 = round((THI_MIN + 3 * THI_DIF / 5).item())
            tick5 = round((THI_MIN + 4 * THI_DIF / 5).item())
            tick6 = round((THI_MIN + THI_DIF).item())
        else:
            factor = 1000 * 10 ** (-int(torch.floor(torch.log10(THI_DIF))))
            tick1 = round(THI_MIN.item() * factor) / factor
            tick2 = round((THI_MIN + THI_DIF / 5).item() * factor) / factor
            tick3 = round((THI_MIN + 2 * THI_DIF / 5).item() * factor) / factor
            tick4 = round((THI_MIN + 3 * THI_DIF / 5).item() * factor) / factor
            tick5 = round((THI_MIN + 4 * THI_DIF / 5).item() * factor) / factor
            tick6 = round((THI_MIN + THI_DIF).item() * factor) / factor
    else:
        tick1 = format(THI_MIN.item(), cbar_tick_format)
        tick2 = format((THI_MIN + THI_DIF / 5).item(), cbar_tick_format)
        tick3 = format((THI_MIN + 2 * THI_DIF / 5).item(), cbar_tick_format)
        tick4 = format((THI_MIN + 3 * THI_DIF / 5).item(), cbar_tick_format)
        tick5 = format((THI_MIN + 4 * THI_DIF / 5).item(), cbar_tick_format)
        tick6 = format((THI_MIN + THI_DIF).item(), cbar_tick_format)

    if smaller_than_flag == 1 and larger_than_flag == 1:
        cbar.set_ticks(
            [0, cbar_colors / 5, 2 * cbar_colors / 5, 3 * cbar_colors / 5, 4 * cbar_colors / 5, cbar_colors])
        cbar.set_ticklabels(['<' + str(tick1), str(tick2), str(tick3), str(tick4), str(tick5), '>' + str(tick6)])
    elif smaller_than_flag == 1:
        cbar.set_ticks(
            [0, cbar_colors / 5, 2 * cbar_colors / 5, 3 * cbar_colors / 5, 4 * cbar_colors / 5, cbar_colors])
        cbar.set_ticklabels(['<' + str(tick1), str(tick2), str(tick3), str(tick4), str(tick5), str(tick6)])
    elif larger_than_flag == 1:
        cbar.set_ticks(
            [0, cbar_colors / 5, 2 * cbar_colors / 5, 3 * cbar_colors / 5, 4 * cbar_colors / 5, cbar_colors])
        cbar.set_ticklabels([str(tick1), str(tick2), str(tick3), str(tick4), str(tick5), '>' + str(tick6)])
    else:
        cbar.set_ticks(
            [0, cbar_colors / 5, 2 * cbar_colors / 5, 3 * cbar_colors / 5, 4 * cbar_colors / 5, cbar_colors])
        cbar.set_ticklabels([str(tick1), str(tick2), str(tick3), str(tick4), str(tick5), str(tick6)])

    max_dif = torch.max(SUR) - BED_MIN
    if torch.max(BED) - BED_MIN > max_dif:
        max_dif = torch.max(BED) - BED_MIN
    bed_dif = torch.max(BED) - BED_MIN

    if zlim != '':
        z_min = cbar_colors + ((zlim[0] - BED_MIN) / bed_dif) * bed_colors
        z_max = cbar_colors + ((zlim[1] - BED_MIN) / bed_dif) * bed_colors
        ax.set_zlim(z_min, z_max)
        tick_dif = z_max - z_min
        ax.set_zticks(
            [z_min, z_min + 0.2 * tick_dif, z_min + 0.4 * tick_dif, z_min + 0.6 * tick_dif, z_min + 0.8 * tick_dif,
             z_max])
        ax.set_zticklabels([zlim[0], zlim[0] + 0.2 * (zlim[1] - zlim[0]), zlim[0] + 0.4 * (zlim[1] - zlim[0]),
                            zlim[0] + 0.6 * (zlim[1] - zlim[0]), zlim[0] + 0.8 * (zlim[1] - zlim[0]),
                            zlim[0] + (zlim[1] - zlim[0])])
    else:
        if vertical_scaling > 0 and vertical_scaling <= 1:
            z_min = cbar_colors
            z_max = cbar_colors + ((max_dif / bed_dif) * bed_colors) / vertical_scaling
            ax.set_zlim(z_min, z_max)
            tick_dif = z_max - z_min
            ax.set_zticks([z_min, z_min + 0.2 * tick_dif, z_min + 0.4 * tick_dif, z_min + 0.6 * tick_dif,
                           z_min + 0.8 * tick_dif, z_max])
            ax.set_zticklabels(
                [BED_MIN, BED_MIN + 0.2 * max_dif / vertical_scaling, BED_MIN + 0.4 * max_dif / vertical_scaling,
                 BED_MIN + 0.6 * max_dif / vertical_scaling, BED_MIN + 0.8 * max_dif / vertical_scaling,
                 BED_MIN + max_dif / vertical_scaling])
        else:
            raise ValueError('Vertical scaling should be between 0 and 1')
```

## 主函数与数据加载
#### 代码单元 7：构造主函数
下面请在你的 Notebook 中添加新的代码单元，运行主函数我们将通过交互指令执行对应的操作以获取所需的结果：
```python
def run():
    try:
        with h5py.File('example_data_Morteratsch_25m.mat', 'r') as mat_file:
            # 假设 .mat 文件中包含 'BED' 和 'SUR' 变量
            BED = torch.tensor(mat_file['BED'][:])
            SUR = torch.tensor(mat_file['SUR'][:])

        # 调用 TopoZeko 函数
        TopoZeko(BED, SUR)
    except FileNotFoundError:
        print("错误：未找到 'example_data_Morteratsch_25m.mat' 文件。")
    except KeyError:
        print("错误：.mat 文件中缺少 'BED' 或 'SUR' 变量。")
```

## 贡献与许可证
恭喜您完成本项目的全部 Tutorial 学习！您可以使用本文提供的模块化代码在您自己的 Notebook 中运行各个代码单元，并根据实际需求调整参数或代码结构。

**贡献者：**  
- 无

**许可证：**  
待添加
 -->
#### 代码单元 1：安装依赖库

<!-- 
代码块用 
```python
...
包裹，上下空出一行
``` -->


```python
# 安装依赖（CPU版本）
pip install -r requirements.txt

# 如需GPU加速（CUDA环境）
pip install -r requirements-gpu.txt
```

<!-- 第二章节：数学背景- 交代项目的数学/物理背景 - 并解释具体实现或使用了哪些公式 -->
<!-- 数学背景统一 2级 Header: ## -->
## 数学背景
地质学中常见的插值方法包括克里金插值法和空间反距离权重法（IDW）等。插值算法原理如下：

<!-- 如果是PINNS技术，要详细给出物理约束
物理约束统一3级 Header: ### -->
### 克里金插值法（Kriging）
核心思想：克里金插值法，也称空间自协方差最佳插值方法。克里金插值法考虑观测点与估计点之间的相对位置信息，利用观测点之间的空间位置信息对待求点进行无偏和最优估计。
目标点$x_0$处的插值：

$$
Z(x_0) = \sum_{i=1}^n \lambda_i Z(x_i)
$$

无偏性约束：

$$
\sum_{i=1}^n \lambda_i = 1
$$

变差函数：

$$
\gamma(h) = \frac{1}{2N(h)} \sum_{i=1}^{N(h)} [Z(x_i) - Z(x_i + h)]^2
$$

克里金系统方程：

$$
\begin{cases}
\sum_{j=1}^n \lambda_j \gamma(x_i, x_j) + \mu = \gamma(x_i, x_0), & i = 1, 2, ..., n \\
\sum_{j=1}^n \lambda_j = 1
\end{cases}
$$
### 反距离权重法（IDW）
核心思想：空间反距离权重法，也称为反距离加权法。该方法主要依赖于反距离的幂值，它的幂参数可以根据已知点和待插值点的距离来控制已知点对插值点的精度影响。该方法用周边采样点的值，估计未知点的值，以待插点与实际观测样本点之间的距离为权重，离插值点越近的样本点赋予的权重越大，其权重贡献与距离成反比。
目标点 $x$ 处的插值：

$$
Z(x) = \frac{\sum_{i=1}^n w_i Z_i}{\sum_{i=1}^n w_i}
$$

权重：

$$
w_i = \frac{1}{d_i^p}
$$
## 算法实现
利用python实现以上算法。
#### 代码单元 2：克里金插值图生成

<!-- 
代码块用 
```python
...
包裹，上下空出一行
``` -->


```python
@st.cache_data  
def create_kriging_interpolated_map(df, element):  
    rbfi = Rbf(X[:, 0], X[:, 1], y, function='gaussian')  
    interpolated_values = rbfi(xx, yy)  
    # 类似IDW的地图渲染逻辑  
    return m  
```

#### 代码单元 3：IDW 插值图生成

<!-- 
代码块用 
```python
...
包裹，上下空出一行
``` -->


```python
@st.cache_data  
def create_idw_interpolated_map(df, element):  
    # 数据预处理与插值逻辑  
    element_data = df[['longitude', 'latitude', element]].dropna()  
    idw = KNeighborsRegressor(n_neighbors=4, weights='distance')  
    # 网格划分与预测  
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))  
    interpolated_values = idw.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)  
    # 地图渲染  
    m = folium.Map(location=[lat_mean, lon_mean], zoom_start=10)  
    folium.raster_layers.ImageOverlay(...)  # 插值结果图层  
    folium.CircleMarker(...)  # 原始数据点标注  
    return m  
```
## 结果可视化

例如，我们现在已经有Cu元素的稀疏数据，可以通过插值得到以下分布：

|![pytorchES - Real and Imaginary](viz/CU.JPG)|
|:--:|
| **Fig.2** Cu元素分布  |

- 使用 Folium 生成基础地图，定位至数据中心点，支持缩放（zoom_start=10）。 
- 叠加插值结果作为栅格图层（如 IDW 和克里金法的元素含量分布云图）。
- 标注原始数据点（CircleMarker），点击显示元素含量及百分比。
图窗还有交互功能：

|![pytorchES - Real and Imaginary](viz/jiao.png)|
|:--:|
| **Fig.3** 交互  |

- 测距功能：通过 Folium 的MeasureControl插件实现两点距离测量。
- 定位功能：调用浏览器定位接口（LocateControl），显示用户当前位置。
- 图形绘制：支持绘制标记、多边形等。

实现该结果可视化的代码如下：
#### 代码单元 4：pytorchES结果可视化

```python
elif option == 'Generate Anomaly Map':
    df = pd.read_csv("Analytical_value_55K03.csv")
    st.title("Generate Anomaly Map 📊")
    # 用户输入元素名称
    search_bar = st.text_input("Enter the name of the element...", key="search_bar")
    element_name = st.session_state.search_bar
    if element_name:
        # 生成 IDW 地图
        st.subheader("IDW Interpolated Map")
        idw_map = create_idw_interpolated_map(df, element_name)
        folium_static(idw_map)
        # 生成克里金地图
        st.subheader("Kriging Interpolated Map")
        kriging_map = create_kriging_interpolated_map(df, element_name)
        folium_static(kriging_map)
```
---


## 主函数与数据加载

### 数据要求
输入文件：.csv格式，包含longitude（经度）、latitude（纬度）、元素名称列（如cu）。
例如：Analytical_value_55K03.csv
## 运行
### 代码单元 5：启动应用
```python
streamlit run main.py  
```
## 导航至生成异常图
输入元素名称（如cu），点击生成。
<!-- License章节 -->
## 贡献与许可证

恭喜您完成本项目的全部 Tutorial 学习！您可以使用本文提供的模块化代码在您自己的 Notebook 中运行各个代码单元，并根据实际需求调整参数或代码结构。

**贡献者：**  
- 例如：Yuhao Ma 
-  

**许可证：**  
<!-- 保留为空- 内容合格 添加 License Key -->