<!-- 
text: “地质研究中的地形可视化”,

area: "在地球科学与地理信息领域，冰川地形演化与物质循环研究面临多维度数据可视化挑战。冰川厚度、床底地形、表面形态等关键参数的空间分布及动态变化，亟需通过高精度建模与可视化手段呈现。现有可视化工具难以满足冰川数据多维度、动态性的展示需求，尤其在床底与表面地形的三维耦合分析、厚度数据的空间分布表征方面存在局限性，导致科研人员难以直观捕捉地形参数的空间关联与异常特征。本项目基于 PyTorch 构建地形可视化工具，通过 厚度计算模型和多维度渲染技术（2D 热力图 / 3D 地形图 / 4D 厚度叠加），实现床底高程、表面高程等数据的高精度可视化，解决冰川建模与地貌模拟中地形数据直观展示的核心问题，为冰川物质循环研究与地形动态分析提供交互式可视化支持。",

tags: [ "地形可视化", "多维度渲染", "厚度计算模型"  ]
-->

# 地质研究中的地形可视化
## 问题背景
在地球科学与地理信息领域，冰川地形的演化与物质循环研究面临着多维度数据可视化的挑战。冰川厚度、床底地形、表面形态等关键参数的空间分布及动态变化，需通过高精度建模与可视化手段呈现。

| ![pytorchES - Real and Imaginary](https://www.science42.tech/cases/caseMarkdown/地质研究中的地形可视化/viz/back.png) |
|:--:|
| **Fig.1** 背景图  |


## 项目概述
本项目一个专门用于地形可视化的工具，主要解决了冰川建模、地貌模拟等科学可视化领域中的核心问题，即如何直观、准确地展示地形数据。
该项目旨在将床底高程和表面高程等地形数据进行可视化，预期输出结果为直观展示地形特征的可视化图像，帮助科研人员更好地理解和分析地形数据。

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

$$
THI = 
\begin{cases}
caxis_{min}, & \text{if } THI < caxis_{min} \\
caxis_{max}, & \text{if } THI > caxis_{max} \\
THI, & \text{otherwise}
\end{cases}
$$

#### 符号说明：
| 符号 | 含义 |
|:--:|:--:|
| `THI` | 厚度矩阵 |
| `caxis_{min}` | 厚度裁剪范围的最小值 |
| `caxis_{max}` | 厚度裁剪范围的最大值 |

> **说明：**  
> - 当 `extra_dimension` 未指定或为 'on' 时，厚度 `THI` 通过 `SUR - BED` 计算得出。  
> - 当 `THI` 超出 `caxis` 范围时，将其裁剪至该范围内。  

## TopoZeko 函数
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

> **解读：**
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

| ![pytorchES - Real and Imaginary](https://www.science42.tech/cases/caseMarkdown/地质研究中的地形可视化/viz/2d.jpg) |
|:--:|
| **Fig.2** 二维地形图可视化|



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

|![pytorchES - Real and Imaginary](https://www.science42.tech/cases/caseMarkdown/地质研究中的地形可视化/viz/to.jpg)|
|:--:|
| **Fig.3** 三维地形图可视化 |



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

|![pytorchES - Real and Imaginary](https://www.science42.tech/cases/caseMarkdown/地质研究中的地形可视化/viz/4d.jpg)|
|:--:|
| **Fig.4** 四维地形图可视化 |


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
- Junlin X.

**许可证：**  
本项目代码遵循 [GNU 通用公共许可证 第三版（GPLv3）](https://www.gnu.org/licenses/gpl-3.0.html)。您可以自由使用、修改和分发该代码，但需保留原始版权声明，并在分发时提供相同的许可证。

版权 © 硒钼科技（北京）。更多信息请访问 [Science42](https://www.science42.tech/#/index)。