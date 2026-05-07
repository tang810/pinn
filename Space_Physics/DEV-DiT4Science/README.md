# AeroDiT : 基于Diffusion+Transformer的高精度翼型流场模拟

## 项目概述
实时且精确地预测机翼周围的气动流场，对于流动控制与气动优化至关重要。然而，由于流体物理的高度非线性特性以及高昂的计算成本，实现这一目标仍具挑战性。传统的计算流体力学（CFD）方法在计算效率与精度之间难以取得平衡，限制了其在实时场景中的应用。为应对这些挑战，本研究提出AeroDiT，一种将可扩展扩散模型与变换器架构相结合的新型代理模型。该模型基于高雷诺数机翼流场的雷诺平均纳维-斯托克斯（RANS）仿真数据进行训练，能够精确捕捉复杂的流动结构，并实现实时预测。模型表现出卓越的性能，压力𝑝及速度分量𝑢𝑥,𝑢𝑦的平均相对𝐿2 L 2误差分别为0.1、0.025和0.050，验证了其可靠性。为进一步增强物理一致性，我们引入了基于RANS残差的显式物理信息损失函数，包含质量与动量守恒约束。变换器架构使得预测可在数秒内完成，从而实现高效的气动仿真。本工作凸显了生成式机器学习技术在推进计算流体力学中的潜力，为高保真气动流场模拟提供了可行的解决方案。
针对传统**计算流体力学**(CFD)方法在高雷诺数翼型流场模拟中计算成本高、实时性差的问题，本项目提出 AeroDiT 创新框架。该模型首次将 **扩散概率模型**（DDPM） 与 **Transformer** 架构 结合，通过学习雷诺平均Navier-Stokes（RANS）模拟数据，实现复杂翼型流场的快速高精度预测。

## 秋月白对话版

我们的产品支持直接加载标准翼型文件并进行仿真预测，操作流程如下：

1. 前往 [UIUC Airfoil Coordinates Database](http://m-selig.ae.illinois.edu/ads/coord_database.html) 下载所需翼型的 `.dat` 文件，例如 **NACA0021**。
2. 在平台界面上传该 `.dat` 文件。
3. 选择相应的 **仿真模拟模块**（如稳态流场预测）。
4. 输入仿真条件，例如：
   - 翼型：`NACA0021`
   - 攻角 (AoA)：`10°`
   - 来流速度：`30 m/s`
5. 系统会自动运行仿真，输出压力分布、速度场、升力/阻力系数及升阻比曲线等结果。

### 示例结果
下面展示了 **NACA0021 @ AoA=10°, V=30 m/s** 的稳态流场案例：



通过该流程，用户只需几步即可完成从几何到物理的端到端预测。

##  物理背景

###  传统计算流体动力学（CFD）的挑战
计算流体动力学（CFD）通过数值方法求解流体控制方程（如Navier-Stokes方程），广泛应用于工程与科学领域。然而，传统CFD面临以下核心问题：  
- **计算成本高昂**：高分辨率模拟需细密网格划分和微小时间步长，尤其在湍流等复杂流动中，计算资源消耗巨大。
- **非线性与混沌特性**：流体方程的非线性特性（如对流项 $(\vec{u} \cdot \nabla)\vec{u}$ 导致数值求解不稳定，需复杂迭代算法）。  
- **物理先验依赖**：传统方法需精确已知边界条件、物理参数（如黏度）和方程形式，限制了其在未知或复杂场景中的应用。  

###  数据驱动方法的兴起与局限  
近年来，**深度学习**(DL)在高维非线性问题中的成功为CFD提供了新思路，但仍存在以下不足：  
- **生成对抗网络**（GAN）：虽能生成高质量流体场，但训练不稳定（如模式崩溃）、需精心设计损失函数（如物理约束）。  
- **物理信息神经网络**（PINN）：依赖已知控制方程，无法直接应用于物理规律未知的场景。  

###  扩散模型（DDPM）的潜力
扩散模型是一类基于逐步添加和去除噪声的生成模型，在图像生成、语音合成等领域取得了显著的成功。其基本思想是通过一个前向扩散过程将数据逐渐转化为噪声，然后通过一个逆向去噪过程从噪声中重建原始数据。在流体动力学中，扩散模型可以用来学习流场数据的统计分布，从而实现流场的生成和预测。与传统的深度学习模型相比，扩散模型能够更好地捕捉数据的复杂分布结构，生成更高质量、更具多样性的流场样本，并且具有较强的鲁棒性和稳定性。扩散模型在图像生成中表现卓越，其核心优势为：  
- **训练稳定性**：通过逐步加噪与去噪过程，避免GAN的对抗训练不稳定性。  
- **高维数据建模**：能有效学习复杂数据分布（如湍流场的多尺度结构）。  
- **无需物理先验**：仅依赖数据驱动，适应性强，可泛化至不同流体场景。  

###  结合**Transformer**架构的扩散模型
标准的扩散模型在处理高维流场数据时往往存在训练效率低、采样速度慢等问题，限制了其在大规模流体动力学模拟中的应用。而结合Transformer架构的扩散模型则有望克服这些局限性，充分利用Transformer在处理长序列数据和捕捉全局依赖关系方面的优势，提高模型的性能和效率，为高雷诺数翼型流场等复杂流体动力学问题的实时、准确预测提供了新的途径。

###  实际应用需求 
流体预测在工程中具有广泛应用，例如：  
- **环境模拟**：烟雾扩散、污染物传输。  
- **航空航天**：翼型绕流、发动机燃烧室设计。  
- **医疗与生物**：血液流动、药物输送。    

## AeroDiT的计算控制方程详解

 ###  模型架构  
AeroDiT由三个核心模块构成：条件编码器、函数空间解码器和基于Transformer的扩散模块。其整体框架如图1所示，具体设计如下：

| ![](https://www.science42.tech/cases/caseMarkdown/AeroDiT2/viz/main_new.png) |
|:--:|
| **Fig.1** 模型架构 |

- **条件编码器**：  
  输入：三维张量（3, 128, 128），包含初始x方向速度、y方向速度及空气翼型掩码（mask）。  
  功能：提取输入条件的特征（如翼型几何形状、来流速度），生成条件嵌入向量，用于引导扩散模块的生成过程。  
  训练：使用变分自编码器（VAE）进行预训练，保留编码器部分作为条件编码器。  

- **函数空间解码器**：  
  输入：扩散模块输出的隐空间特征。  
  功能：将隐特征映射回物理空间，生成目标流场（压力、x速度、y速度）。  
  训练：与条件编码器非对称训练，单独预训练后保留解码器。  

- **扩散模块（DiT）**：  
  核心结构：基于Transformer架构，包含多头自注意力机制和前馈神经网络。  
  扩散过程：  
    前向扩散：通过逐步添加高斯噪声将流场数据破坏为噪声分布。  
    反向去噪：利用Transformer预测噪声，逐步恢复原始流场。  
  创新点：相比传统U-Net扩散模型，DiT通过自注意力机制捕捉流场的长程依赖关系，提升对复杂湍流结构的建模能力。  

 ###  扩散过程数学描述  
扩散过程分为前向噪声添加和反向噪声预测两个阶段：  

- **前向扩散（Forward Process）**： 
  从初始流场 $\bm{x}_0$ 出发，逐步添加高斯噪声，经过 $T$ 步后得到纯噪声 $\bm{x}_T$。  
  *噪声调度*：使用预定义的噪声方差 $\beta_t$（随时间递增），控制噪声添加强度。  
  *条件扩散公式*：  
    $$
    q(\bm{x}_t|\bm{x}_{t-1}, \bm{\Psi}) = \mathcal{N}\left(\bm{x}_t; \sqrt{1-\beta_t}\bm{x}_{t-1}, \beta_t\mathbf{I}\right)
    $$ 
  *累积噪声公式*：  
    $$
    q(\bm{x}_t|\bm{x}_0, \bm{\Psi}) = \mathcal{N}\left(\bm{x}_t; \sqrt{\bar{\alpha}_t}\bm{x}_0, (1-\bar{\alpha}_t)\mathbf{I}\right)
    $$
    其中 $\bar{\alpha}_t = \prod_{i=1}^t (1-\beta_i)$，表示噪声累积系数。  
- **反向去噪（Reverse Process）**：  
 从噪声 $\bm{x}_T$ 出发，通过迭代预测噪声 $\bm{\epsilon}_\theta$，逐步恢复原始流场 $\bm{x}_0$。  
  *去噪均值公式*：  
    $$
    \tilde{\mu}_t = \frac{1}{\sqrt{\bar{\alpha}_t}} \left( \bm{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \bm{\epsilon}_\theta(\bm{x}_t, t, \bm{\Psi}) \right)
    $$ 
  *损失函数*：  
    $$
    \mathscr{L}_{MSE}(\theta) = \mathbb{E}_{t,\bm{x}_0,\bm{\Psi},\bm{\epsilon}} \left[ \| \bm{\epsilon} - \bm{\epsilon}_\theta(\bm{x}_t, t, \bm{\Psi}) \|^2 \right]
    $$ 
    目标是最小化预测噪声与真实噪声的均方误差。  

###  训练策略与流程
阶段非对称训练：  
  1. 条件编码器与解码器预训练：  
    分别训练条件编码器和函数空间解码器的VAE模型，各100轮次。  
    条件编码器输入为初始速度场和翼型掩码，解码器输出为流场物理量。  
  2. DiT模块训练：  
     固定预训练的编码器和解码器，仅训练扩散模块。  
     使用Adam优化器，迭代250万次，批量学习数据特征。  

###  数据集与实验配置 
**数据生成**：  
  来源：基于RANS模拟，采用Spalart-Allmaras湍流模型。范围：  
    雷诺数 $Re \in [0.5, 5] \times 10^6$，攻角范围 $\pm 22.5^\circ$。  
    包含1505种UIUC数据库翼型，随机组合来流条件生成22万张128×128网格数据。  
  数据集划分：  
   - Regular：标准数据集（22万张）。  
   - Shear：通过翼型剪切增强生成（22万张）。  
   - Subset：小规模数据集（6000张），用于测试泛化能力。  

**模型配置**：  
  DiT规模：  
   - DiT-B/2：1.29亿参数，显存5.5GB，训练速度23.67步/秒。  
   - DiT-L/2：4.57亿参数，显存14GB，训练速度10.89步/秒。  
   - DiT-XL/2：6.74亿参数，显存21GB，训练速度7.79步/秒。  
   
  硬件：NVIDIA RTX 6000 GPU，支持分布式训练。  


## 结果与讨论

### 定性及定量结果分析

三个典型测试案例验证模型性能，覆盖不同翼型与雷诺数范围($Re = 2.04 \times 10^6$)至 ($2.00 \times 10^7$)，具体结果如下：  
 **案例1**：Drela AG09翼型( $Re = 4.04 \times 10^6$ )
流场对比：



预测压力场与CFD结果视觉一致性高，最大绝对误差约 0.09（归一化后）。  
  - x方向速度场误差 <0.010，y方向速度场误差 <0.041，主要误差集中在翼型表面附近。  

**案例2**：AH 63-K-127/24翼型($Re = 9.11 \times 10^6$)
性能提升：  


  - 压力场误差 **<0.075**，速度场误差进一步降低($u_x <0.020$, $u_y <0.045$)。
表面参数曲线与CFD几乎重合，表明模型在中等雷诺数范围内泛化能力优异。

**案例3**：EPPLER 59翼型($Re = 1.21 \times 10^7$)


| ![](https://www.science42.tech/cases/caseMarkdown/AeroDiT2/viz/FIGALL.png) |
|:--:|
| **Fig.2** Comparison of surface pressure and velocities. |

- 表面压力与速度曲线出现局部偏离，尤其在翼型后缘分离区。  
可能原因：  
  1. 数据不足：训练集中 $Re > 1.0 \times 10^7$ 的样本稀缺，模型未充分学习高雷诺数流场的强湍流特性。  
  2. 流场突变：高雷诺数下流动分离和涡结构更剧烈，传统扩散模型难以捕捉动态演化过程。  

### 结果总结
核心优势：
1. 生成式建模：通过扩散过程学习流场统计分布，避免直接求解Navier-Stokes方程的计算开销。 
2. Transformer优势：自注意力机制有效捕捉流场中的全局依赖（如分离涡、激波）。   
3. 工程适用性：直接输出压力场与速度场，支持气动性能评估（如升力系数计算）。 采样步数减少至秒级，满足实时控制需求。  
4. 高精度与泛化性：在中等雷诺数($Re < 1.0 \times 10^7$)下，模型对多种翼型的流场预测误差低于2%，优于传统U-Net扩散模型。 
5. 条件生成能力：通过输入翼型几何与来流条件，模型可灵活适应不同工况，减少重复计算需求。  



## 使用AeroDiT模型生成翼型流场

### 翼型几何及流场生成

```python
def read_airfoil_dat(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    data = np.loadtxt(lines[1:])
    x, y = data[:, 0], data[:, 1]

    return x, y # 功能：读取UIUC翼型数据库的.dat文件，返回坐标数组


def generate_binary_mask(x_airfoil, y_airfoil, res=128):
    """
    生成 128x128 的 binary mask，使用 xf 和 yf 规则进行坐标变换
    """
    # 创建 128x128 的网格
    mask = np.zeros((res, res))

    # 计算物理坐标范围
    x_min, x_max = np.min(x_airfoil), np.max(x_airfoil)
    y_min, y_max = np.min(y_airfoil), np.max(y_airfoil)

    # 创建基于网格的物理坐标
    x_grid = np.array([(x / res - 0.5) * 2 + 0.5 for x in range(res)])
    y_grid = np.array([(y / res - 0.5) * 2 for y in range(res)])

    # 生成二维网格
    xv, yv = np.meshgrid(x_grid, y_grid)

    # 构造翼型轮廓的凸包（Delaunay 三角剖分法）
    points = np.vstack((x_airfoil, y_airfoil)).T
    tri = Delaunay(points)

    # 遍历所有网格点，判断是否在翼型内部
    for i in range(res):
        for j in range(res):
            point = np.array([xv[i, j], yv[i, j]])
            if tri.find_simplex(point) >= 0:
                mask[i, j] = 1  # 标记为翼型区域
    
    #mask = np.rot90(mask, k=1)  # k=1 表示顺时针旋转90度
    return mask.T  # 功能：生成128x128的二进制掩膜，标记翼型区域为1。

def save_and_plot_mask(mask, filename="binary_mask.png"):

    plt.imshow(mask, cmap='gray')
    plt.title("Airfoil Binary Mask")
    #plt.colorbar(label="Mask Value (0: Background, 1: Airfoil)")
    plt.savefig(filename)
    plt.show()  # 功能：显示并保存掩膜图像。

def generate_npz_from_dat(file_path, vx, vy, res=128):
    """
    读取 .dat 文件，生成 6 维 npz 文件，并按照翼型名称_vx*100_vy*100.npz 命名。
    :param file_path: dat 文件路径
    :param vx: 横向速度场 (numpy 数组，大小需与 mask 匹配)
    :param vy: 纵向速度场 (numpy 数组，大小需与 mask 匹配)
    :param res: 解析度 (默认 128)
    """
    # 获取翼型名称（去掉扩展名）
    airfoil_name = os.path.splitext(os.path.basename(file_path))[0]

    # 格式化 vx 和 vy，乘以 100 取整数
    vx_int = int(vx * 100)
    vy_int = int(vy * 100)

    # 生成输出文件名
    output_file = f"{airfoil_name}_{vx_int}_{vy_int}.npz"

    # 读取翼型数据并生成 mask
    x_airfoil, y_airfoil = read_airfoil_dat(file_path)
    mask = generate_binary_mask(x_airfoil, y_airfoil, res)

    # 计算 vx, vy 乘以 mask 取反后的结果
    mask_inv = 1 - mask  # 取反 mask
    masked_vx = vx * mask_inv
    masked_vy = vy * mask_inv

    # 生成 6 维数据
    zero_field = np.zeros_like(mask)
    output_data = np.stack([masked_vx, masked_vy, mask, zero_field, zero_field, zero_field], axis=0)

    # 指定 key 为 'a'，确保代码可以正确读取
    np.savez(output_file, a=output_data)  # 这里的 'a' 要和 np.load() 时的 key 对应
    print(f"Saved 6D npz file to {output_file} with key 'a'")
    return output_data  #功能：生成包含6个通道的NPZ文件，通道0-1：速度场Vx、Vy（翼型外部保留，内部置0）。通道2：掩膜。通道3-5：零场（占位）。
```
**运行代码示例**
```python
file_path = '文件路径'  # 请替换成 .dat 文件路径
x_airfoil, y_airfoil = read_airfoil_dat(file_path)
mask = generate_binary_mask(x_airfoil, y_airfoil)
save_and_plot_mask(mask)
generated_mask=mask
generate_npz_from_dat(file_path,20,1)
```

### 不同攻角及翼型表面法向量生成
```python
def compute_airfoil_normals(x, y):
    """
    计算翼型边界上的单位法向量，确保其方向指向外部
    """
    # 正向计算法向量
    dx_fwd = np.roll(x, -1) - x
    dy_fwd = np.roll(y, -1) - y
    nx_fwd = -dy_fwd
    ny_fwd = dx_fwd
    norm_fwd = np.sqrt(nx_fwd**2 + ny_fwd**2)
    nx_fwd /= norm_fwd
    ny_fwd /= norm_fwd

    # 逆向计算法向量
    dx_bwd = x - np.roll(x, 1)
    dy_bwd = y - np.roll(y, 1)
    nx_bwd = -dy_bwd
    ny_bwd = dx_bwd
    norm_bwd = np.sqrt(nx_bwd**2 + ny_bwd**2)
    nx_bwd /= norm_bwd
    ny_bwd /= norm_bwd

    # 平均正向和逆向法向量
    nx = (nx_fwd + nx_bwd) / 2
    ny = (ny_fwd + ny_bwd) / 2

    # 归一化
    norm = np.sqrt(nx**2 + ny**2)
    nx /= norm
    ny /= norm

    # 计算翼型中心
    xc = np.mean(x)
    yc = np.mean(y)

    # 计算从翼型点指向中心的向量
    vx = xc - x
    vy = yc - y

    # 计算点积，判断方向
    dot_product = nx * vx + ny * vy

    # 反转错误方向的法向量
    flip_mask = dot_product > 0
    nx[flip_mask] *= -1
    ny[flip_mask] *= -1

    return nx, ny

def generate_binary_mask(x_airfoil, y_airfoil, res=128):
    """
    生成 128x128 的 binary mask，使用 xf 和 yf 规则进行坐标变换
    """
    # 创建 128x128 的网格
    mask = np.zeros((res, res))

    # 计算物理坐标范围
    x_min, x_max = np.min(x_airfoil), np.max(x_airfoil)
    y_min, y_max = np.min(y_airfoil), np.max(y_airfoil)

    # 创建基于网格的物理坐标
    x_grid = np.array([(x / res - 0.5) * 2 + 0.5 for x in range(res)])
    y_grid = np.array([(y / res - 0.5) * 2 for y in range(res)])

    # 生成二维网格
    xv, yv = np.meshgrid(x_grid, y_grid)

    # 构造翼型轮廓的凸包（Delaunay 三角剖分法）
    points = np.vstack((x_airfoil, y_airfoil)).T
    tri = Delaunay(points)

    # 遍历所有网格点，判断是否在翼型内部
    for i in range(res):
        for j in range(res):
            point = np.array([xv[i, j], yv[i, j]])
            if tri.find_simplex(point) >= 0:
                mask[i, j] = 1  # 标记为翼型区域
    
    #mask = np.rot90(mask, k=1)  # k=1 表示顺时针旋转90度
    return mask.T

def generate_boundary_mask_with_normals(x_airfoil, y_airfoil, nx, ny, res=128):
    """
    生成 128x128 的边界 binary mask，并在边界处存储 nx 和 ny
    """
    # 创建 128x128 的网格
    mask = np.zeros((res, res))
    nx_mask = np.zeros((res, res))
    ny_mask = np.zeros((res, res))

    # 计算物理坐标范围
    x_min, x_max = np.min(x_airfoil), np.max(x_airfoil)
    y_min, y_max = np.min(y_airfoil), np.max(y_airfoil)

    # 创建基于网格的物理坐标
    x_grid = np.array([(x / res - 0.5) * 2 + 0.5 for x in range(res)])
    y_grid = np.array([(y / res - 0.5) * 2 for y in range(res)])
    xv, yv = np.meshgrid(x_grid, y_grid)

    # 构造翼型轮廓的凸包（Delaunay 三角剖分法）
    points = np.vstack((x_airfoil, y_airfoil)).T
    tri = Delaunay(points)

    # 遍历所有网格点，判断是否在翼型内部
    for i in range(res):
        for j in range(res):
            point = np.array([xv[i, j], yv[i, j]])
            if tri.find_simplex(point) >= 0:
                mask[i, j] = 1  # 标记为翼型区域

    # 计算边界：对 mask 进行膨胀后相减，得到边界区域
    mask_dilated = binary_dilation(mask)
    boundary_mask = mask_dilated.astype(int) - mask.astype(int)

    # 在边界处存储法向量
    for i in range(res):
        for j in range(res):
            if boundary_mask[i, j] == 1:
                # 找到最近的翼型边界点
                point = np.array([xv[i, j], yv[i, j]])
                closest_index = np.argmin(np.linalg.norm(points - point, axis=1))

                # 赋值法向量
                nx_mask[i, j] = nx[closest_index]
                ny_mask[i, j] = ny[closest_index]
    
    return boundary_mask.T, nx_mask.T, ny_mask.T

def shift_boundary_mask(mask):
    """
    对边界上的数值进行错位移动（循环前移一格）
    
    :param mask: 输入的蒙版 (nx_mask 或 ny_mask)
    :return: 旋转后的蒙版
    """
    # 找到所有非零像素的索引
    nonzero_indices = np.argwhere(mask != 0)
    
    # 获取非零像素的数值
    nonzero_values = mask[nonzero_indices[:, 0], nonzero_indices[:, 1]]
    
    # 将这些值循环前移一格
    shifted_values = np.roll(nonzero_values, -1)

    # 生成新的 mask
    shifted_mask = np.zeros_like(mask)

    # 赋值回去
    for i, (x, y) in enumerate(nonzero_indices):
        shifted_mask[x, y] = shifted_values[i]

    return shifted_mask

def generate_npz_for_angles(file_path, speed, res=128):
    """
    读取 .dat 文件，生成不同攻角下的 npz 文件，并包含 nx_mask 和 ny_mask
    :param file_path: dat 文件路径
    :param speed: 速度大小 (float)
    :param res: 解析度 (默认 128)
    """
    airfoil_name = os.path.splitext(os.path.basename(file_path))[0]
    x_airfoil, y_airfoil = read_airfoil_dat(file_path)

    # 生成蒙版和法向量蒙版
    mask = generate_binary_mask(x_airfoil, y_airfoil, res)
    mask_inv = 1 - mask  
    zero_field = np.zeros_like(mask)

    # 计算法向量
    nx, ny = compute_airfoil_normals(x_airfoil, y_airfoil)

    # 生成边界 mask
    _, nx_mask, ny_mask = generate_boundary_mask_with_normals(x_airfoil, y_airfoil, nx, ny, res)

    for angle in range(-16, 16, 1):
        rad = np.radians(angle)
        ux = speed * np.cos(rad)
        uy = speed * np.sin(rad)

        # 计算速度场，并确保只在翼型外部
        masked_ux = ux * mask_inv
        masked_uy = uy * mask_inv

        # 组合数据：包含速度场、mask 以及 nx_mask 和 ny_mask
        output_data = np.stack([masked_ux, masked_uy, mask, zero_field, zero_field, zero_field], axis=0)
        boundary_normals = np.stack([nx_mask, ny_mask], axis=0)  # nx_mask, ny_mask 作为单独的通道

        # 生成文件路径
        output_file = f"{airfoil_name}_{int(speed)}_{angle}.npz"

        # 保存数据
        np.savez(output_file, a=output_data, b=angle, c=boundary_normals)

        print(f"Saved: {output_file}")

def visualize_airfoil_normals(x,y,nx,ny,scale=0.05):
    #可视化翼型及其法向量
    
    plt.figure(figsize=(6,3))
    plt.plot(x,y,'k-',label='Airfoil Boundary')
    plt.quiver(x,y,nx,ny,color='r',angles='xy',scale_units='xy', scale=1/scale, width=0.002)
    plt.axis("equal")
    plt.title("airfoil boundary with normal vectors")
    plt.legend()
    plt.show()

def save_and_plot_mask(mask, filename="binary_mask.png", title="Binary Mask"):
    """
    显示并保存 binary mask
    """
    plt.imshow(mask, cmap='coolwarm')
    plt.title(title)
    plt.savefig(filename)
    plt.show()
''
```

**运行示例**
```python
file_path = '/data/sdb/bk/mayuan/DiT4Science-DA/test_gen/dat_files/m17.dat'  # 替换为你的翼型文件
x_airfoil, y_airfoil = read_airfoil_dat(file_path)
mask = generate_binary_mask(x_airfoil, y_airfoil)
save_and_plot_mask(mask)
```
| ![](https://www.science42.tech/cases/caseMarkdown/AeroDiT2/viz/Binary_Mask.png) |
|:--:|
| **Fig.3** Binary_Mask |

```python
#运行示例
x_airfoil, y_airfoil = read_airfoil_dat(file_path)

#计算法向量
nx, ny = compute_airfoil_normals(x_airfoil, y_airfoil)

#可视化
visualize_airfoil_normals(x_airfoil, y_airfoil, nx, ny)

boundary_mask, nx_mask, ny_mask = generate_boundary_mask_with_normals(x_airfoil, y_airfoil, nx, ny)

#保存和可视化
save_and_plot_mask(boundary_mask, filename="boundary_mask.png", title="Airfoil Boundary Mask")
save_and_plot_mask(nx_mask, filename="nx_mask.png", title="Normal X on Boundary")
save_and_plot_mask(ny_mask, filename="ny_mask.png", title="Normal Y on Boundary")
#np.savetxt(filename, results, delimiter=',', header=header)
np.savetxt('boundary_mask.csv', boundary_mask)
generate_npz_for_angles(file_path,7.4187) #生成不同攻角下的翼型数据并存储为npz格式
```
| ![](https://www.science42.tech/cases/caseMarkdown/AeroDiT2/viz/Airfoil_Boundary_with_normal_vectors.png) |
|:--:|
| **Fig.4** Airfoil_Boundary_with_normal_vectors |



###  使用AeroDiT模型计算不同攻角下的流场

```python
import os
import sys
from dit4science.inference_zhenghao import inference
import torch
from dit4science.model.dit_models import DiT_models
from dit4science.data.dataset_dit_zhibo import TurbDataset
from dit4science.download import find_model
from dit4science.model.autoencoder import AutoencoderKL
import gc

class DiTTest:
    def __init__(self, data_path, model_ckpt, vae_ckpt_cond, vae_ckpt_res, model_type,channel_idx=-1, prop=None, image_size=128, num_classes=1000, cfg_scale=40):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data_path = data_path
        self.channel_idx = channel_idx
        self.prop = prop
        self.image_size = image_size
        self.model_ckpt = model_ckpt
        self.num_classes = num_classes
        self.cfg_scale = cfg_scale
        self.latent_size = self.image_size // 8
        self.vae_ckpt_cond = vae_ckpt_cond
        self.vae_ckpt_res = vae_ckpt_res
        self.model_type = model_type

        # Load the dataset
        self.dataset = TurbDataset(
            dataProp=self.prop, dataDir=self.data_path,
            gaussian_norm=False, channel_idx=self.channel_idx
        )

        # Load the model
        self.model = self.load_model(self.model_type)

        # Load the VAE models
        self.vae_cond = self.load_vae(self.vae_ckpt_cond)
        self.vae_res = self.load_vae(self.vae_ckpt_res)

    def load_model(self,model_type):
        
        model = DiT_models[model_type](
            input_size=self.latent_size,
            num_classes=self.num_classes,
        ).to(self.device)
        state_dict = find_model(self.model_ckpt)
        model.load_state_dict(state_dict)
        return model

    def load_vae(self, vae_ckpt):
        config = {
            "monitor": "val/rec_loss",
            "embed_dim": 4,
            "lossconfig": {
                "target": "dit4science.loss.LPIPSWithDiscriminator",
                "params": {
                    "disc_start": 50001,
                    "kl_weight": 0.000001,
                    "disc_weight": 0.5
                }
            },
            "ddconfig": {
                "double_z": True,
                "z_channels": 4,
                "resolution": 256,
                "in_channels": 3,
                "out_ch": 3,
                "ch": 128,
                "ch_mult": [1, 2, 4, 4],  # num_down = len(ch_mult) - 1
                "num_res_blocks": 2,
                "attn_resolutions": [],
                "dropout": 0.0
            }
        }
        vae_model = AutoencoderKL(**config).to(self.device)
        vae_model.init_from_ckpt(vae_ckpt)
        return vae_model

    def run_inference(self, num_sampling_steps=250):
        inference(self.dataset, self.model, self.vae_cond, self.vae_res, latent_size=self.latent_size, num_sampling_steps=num_sampling_steps,flag=True,plot_tar=True)


if __name__ == "__main__":
   ####F.B: Specific settings:
   
    #model_type = "DiT-B/2"  # noqa0
    model_type= "DiT-XL/2" 
    #data_path = "/data/sda/mayuan/data/test_sep/high_Re/"
    #data_path = "/data/sda/mayuan/data/test_sep/low_Re/"
    data_path = "/data/sdb/bk/mayuan/DiT4Science-DA/test/"
    
    base_ckpt_path = "/data/sdb/bk/mayuan/results/008-DiT-XL-2-channel_idx-1/checkpoints/"
    start_ckpt =    2900000
    end_ckpt =      2900000
    step = 20000
    
    vae_ckpt_cond = "/data/sdb/bk/mayuan/logs/record/cond_reg/testtube/version_0/checkpoints/epoch=100-step=221594.ckpt"
    vae_ckpt_res  = "/data/sdb/bk/mayuan/logs/record/func_reg/testtube/version_0/checkpoints/epoch=100-step=221594.ckpt"
    #vae_ckpt_cond   = "/data/sda/mayuan/logs/record/cond_reg/testtube/version_0/checkpoints/epoch=100-step=221594.ckpt"
    #vae_ckpt_res    = "/data/sda/mayuan/logs/record/func_reg/testtube/version_0/checkpoints/epoch=100-step=221594.ckpt"
    
    for ckpt_number in range(start_ckpt, end_ckpt + 1, step):
        #domain = "model_domain1"
        #domain = "model_domain2"
        domain = ""
        ckpt = f"{base_ckpt_path}{domain}{ckpt_number:07d}.pt"
        
        print(f"Processing checkpoint: {ckpt}")
        try:
            handler = DiTTest(data_path, ckpt, vae_ckpt_cond, vae_ckpt_res, model_type)
            handler.run_inference(num_sampling_steps=100)
            
            del handler
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"Error processing checkpoint {ckpt}: {e}")
```

| ![](https://www.science42.tech/cases/caseMarkdown/AeroDiT2/viz/e59_6031_421.png) |
|:--:|
| **Fig.5** 生成示例 |

| ![](https://www.science42.tech/cases/caseMarkdown/AeroDiT2/viz/升力.png) |
|:--:|
| **Fig.6** 升力系数 |

| ![](https://www.science42.tech/cases/caseMarkdown/AeroDiT2/viz/阻力.png) |
|:--:|
| **Fig.7** 阻力系数 |

| ![](https://www.science42.tech/cases/caseMarkdown/AeroDiT2/viz/升阻比.png) |
|:--:|
| **Fig.8** 升阻比 |


## 结论
AeroDit通过将可扩展扩散模型与Transformer架构相融合，在翼型绕流预测中实现了高精度与物理一致性。其核心创新在于几何条件编码与物理场解耦生成，为空气动力学工程应用提供了高效数据驱动工具。

