import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator
from sympy import symbols, init_printing

init_printing(use_latex=True)  # 启用 LaTeX 渲染
aa = symbols('a')  # 定义数学符号 a

# ======================
# 样式配置
# ======================
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'mathtext.fontset': 'cm',  # Computer Modern（默认数学字体）
    'mathtext.default': 'it',  # 默认斜体
    'figure.figsize': (16, 3),
    'axes.edgecolor': '#2E2E2E',
    'axes.linewidth': 1.8,
    'grid.color': '#EDEDED',
    'grid.linestyle': '--',
    'grid.alpha': 0.7,
    'font.family': 'DejaVu Sans',
    'xtick.color': '#555555',
    'ytick.color': '#555555'
})

# 设置调色板
colors = sns.color_palette("husl", 3)
param_colors = {
    'h1': colors[0],
    'rho': colors[1], 
    'a': colors[2]
}

# ======================
# 数据准备
# ======================
len_parameters = 30
len_x = 30

np.random.seed(42)
h1 = np.linspace(4, 10, len_parameters) + np.random.normal(0, 0.1, len_parameters)
rho = np.linspace(0.9, 0.99, len_parameters) + np.random.normal(0, 0.002, len_parameters)
a = np.linspace(-0.4, -0.05, len_parameters) + np.random.normal(0, 0.01, len_parameters)

# ======================
# 可视化映射系统
# ======================
class VisualMapper:
    def __init__(self, vrange, yrange, buffer=0.4):
        self.scale = (yrange[1]-yrange[0]-2*buffer)/(vrange[1]-vrange[0])
        self.offset = yrange[0] + buffer - vrange[0]*self.scale
        
    def __call__(self, values):
        return np.asarray(values) * self.scale + self.offset

mappers = {
    'a': VisualMapper((-0.4, -0.05), (0.0, 2.5)),
    'rho': VisualMapper((0.9, 0.99), (3.0, 5.5)),
    'h1': VisualMapper((4, 10), (6.0, 9.5))
}

a_mapped = mappers['a'](a)
rho_mapped = mappers['rho'](rho)
h1_mapped = mappers['h1'](h1)

# ======================
# 创建图形（修复坐标轴闭合问题）
# ======================
fig, ax = plt.subplots(figsize=(16, 9))

# 手动闭合坐标轴（关键修复）
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['top'].set_color('#2E2E2E')
ax.spines['right'].set_color('#2E2E2E')
ax.spines['top'].set_linewidth(1.8)
ax.spines['right'].set_linewidth(1.8)

# ======================
# 绘制散点图
# ======================
marker_styles = {
    'h1': {'marker': 'H', 's': 98, 'linewidths': 1.8},
    'rho': {'marker': 'o', 's': 110, 'linewidths': 1.5},
    'a': {'marker': 's', 's': 85, 'linewidths': 1.6}
}

for i, (x, y) in enumerate(zip(np.arange(len_x), h1_mapped)):
    ax.scatter(x, y, color=param_colors['h1'], alpha=0.8+0.2*(i/len_x), 
               zorder=10, **marker_styles['h1'])

for i, (x, y) in enumerate(zip(np.arange(len_x), rho_mapped)):
    ax.scatter(x, y, color=param_colors['rho'], alpha=0.7+0.3*(i/len_x),
               zorder=10, **marker_styles['rho'])

for i, (x, y) in enumerate(zip(np.arange(len_x), a_mapped)):
    ax.scatter(x, y, color=param_colors['a'], alpha=0.75+0.25*(i/len_x),
               zorder=10, **marker_styles['a'])

# ======================
# 配置坐标轴和刻度
# ======================
def setup_axes():
    # 设置主刻度
    ax.set_yticks(np.concatenate([
        np.linspace(0.5, 2.0, 4),    # a参数区域
        np.linspace(3.5, 5.0, 4),    # rho参数区域
        np.linspace(6.5, 9.0, 6)     # h1参数区域
    ]))
    
    # 设置刻度标签（显示原始值）
    yticks = ax.get_yticks()
    yticklabels = []
    for y in yticks:
        if y <= 2.5:  # a参数区域
            orig_val = (y - mappers['a'].offset)/mappers['a'].scale
            yticklabels.append(f"{orig_val:.2f}")
        elif y <= 5.5:  # rho参数区域
            orig_val = (y - mappers['rho'].offset)/mappers['rho'].scale
            yticklabels.append(f"{orig_val:.3f}")
        else:  # h1参数区域
            orig_val = (y - mappers['h1'].offset)/mappers['h1'].scale
            yticklabels.append(f"{orig_val:.1f}")
    
    ax.set_yticklabels(yticklabels)
    
    # 设置次要刻度
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis='y', which='both', length=4, labelsize=10)
    
    # 增强轴线
    ax.spines['left'].set_position(('outward', 1))
    ax.spines['bottom'].set_position(('outward', 1))

setup_axes()

# ======================
# 添加辅助元素
# ======================
# 区域标签
ax.text(0.02, 1.25, r'$a$', transform=ax.get_yaxis_transform(),
        color=param_colors['a'], fontsize=25, rotation=90, va='center')
ax.text(0.02, 4.25, r'$\rho$', transform=ax.get_yaxis_transform(),
        color=param_colors['rho'], fontsize=25, rotation=90, va='center')
ax.text(0.02, 7.75, r'$h_1$', transform=ax.get_yaxis_transform(),
        color=param_colors['h1'], fontsize=25, rotation=90, va='center')

# 分隔带
for y in [2.5, 5.5]:
    ax.axhspan(y-0.1, y+0.1, color='#000000', alpha=0.05, zorder=5)

# ======================
# 添加图例和标题
# ======================
title = ax.set_title("Parameter Space Visualization", 
                    fontsize=25, pad=25, color='#2E2E2E',
                    fontweight='semibold')

legend_elements = [
    plt.Line2D([0], [0], marker='H', color='w', label=r'$h_1 \in [4.0,10.0]$',
               markerfacecolor=param_colors['h1'], markersize=12),
    plt.Line2D([0], [0], marker='o', color='w', label=r'$\rho \in [0.90,1)$',
               markerfacecolor=param_colors['rho'], markersize=14),
    plt.Line2D([0], [0], marker='s', color='w', label=r'$a \in [-0.40,-0.05]$',
               markerfacecolor=param_colors['a'], markersize=12)
]

ax.legend(handles=legend_elements, 
          loc='upper center', 
          bbox_to_anchor=(0.33, 1),
          ncol=3,
          frameon=True,
          framealpha=0.9,
          edgecolor='#2E2E2E',
          fontsize=20)

# ======================
# 最终调整
# ======================
ax.set_xlabel("Sample Index", fontsize=25, labelpad=15, color='#555555')
ax.set_ylabel("Parameter Value", fontsize=25, labelpad=20, color='#555555')
fig.set_facecolor('#FFFFFF')
fig.tight_layout()
# plt.savefig('parameter_visualization.png', dpi=600, bbox_inches='tight')
# plt.show()

plt.savefig('parameter_visualization_closed_axes.png', dpi=600, bbox_inches='tight')
plt.show()