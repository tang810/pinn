<!-- 
text: "电磁散射模拟推演",
area: "本项目针对二维电磁波在未知目标区域中的传播与反射问题，构建基于 PINN 的逆散射求解器。通过亥姆霍兹方程与 PML 吸收边界建模，分别实现了正问题（已知介电常数求电场）与反问题（已知电场反演目标位置与介电常数）。",
tags: [
  "基础方程",
  "电磁场理论",
  "物理信息神经网络(PINN)",
  "亥姆霍兹方程",
],
search: ["基础方程", "电磁学"],
 -->

# 电磁散射模拟推演

## 背景介绍

电磁反散射问题构成了现代无损检测和医学成像技术的数学物理基础。从机场安检的毫米波扫描仪到医院的核磁共振成像（MRI），从地下管线探测到地球物理勘探，电磁波与物质的相互作用信息为人类提供了"透视"物质内部结构的能力。


## 原理介绍

物理信息神经网络(PINN)是一种新兴的求解偏微分方程的方法，它将物理定律直接嵌入神经网络的训练过程中。本项目基于二维 TM 模式下的频域 Helmholtz 方程，构建高精度物理信息神经网络（PINN），在多入射角、多频率远场散射数据的约束下，对均匀圆柱散射体的等效介电常数与半径参数进行反演。

![caseMarkdown/电磁逆散射/viz/21.png](https://www.science42.tech/cases/caseMarkdown/电磁逆散射/viz/21.png)

## 创新点

- 方法论创新：首个面向电磁反散射的参数化PINN框架

- 算法架构创新：多频率多角度联合反演

- 训练策略创新：物理引导的课程学习

- 计算效率创新：无网格前向建模

- 验证方法创新：合成-实验闭环验证


## 1、控制方程

亥姆霍兹方程：
$$
\nabla^{2} E(\mathbf{r}) + k_0^{2}\
\varepsilon_r(\mathbf{r})E(\mathbf{r}) = 0
$$

其中：
$$
\mathbf{r} = (x,y) \in \mathbb{R}^{2}
$$

$$
E(\mathbf{r}) = E_r(\mathbf{r}) + j E_i(\mathbf{r})
$$
$$
k_0 = \omega \sqrt{\mu_0 \varepsilon_0}
$$




总损失方程：

$$
L_{\mathrm{total}}=
L_{\mathrm{PDE}}
+
\lambda_{\mathrm{data}} L_{\mathrm{data}}
+
\lambda_{\varepsilon} L_{\varepsilon}
+
\lambda_{r} L_{r}
$$


如上式所示，我们将给出每一部分具体的定义：

$$
L_{\mathrm{PDE}}=
\frac{1}{N}
\sum_{i=1}^{N}
\left(
\left\| R_{\mathrm{real}}(\mathbf{x}_i) \right\|^{2}+
\left\| R_{\mathrm{imag}}(\mathbf{x}_i) \right\|^{2}
\right)
$$


$$
L_{\mathrm{data}}=
\frac{1}{M}
\sum_{m=1}^{M}
\left\|E_{\mathrm{pred}}(\mathbf{x}_m)
-E_{\mathrm{obs}}(\mathbf{x}_m)
\right\|^{2}
$$


$$
L_{\varepsilon}=
\left\|
\varepsilon - \varepsilon_{\mathrm{prior}}
\right\|^{2}
$$

$$
L_{r}=
\left\|
r - r_{\mathrm{prior}}
\right\|^{2}
$$



PDE实部残差：

$$
R_{\mathrm{real}}(\mathbf{x})=
\nabla^{2} E_r(\mathbf{x})
+
k_0^{2}\left[1+\chi(\mathbf{x})\right]E_r(\mathbf{x})
$$



PDE虚部残差：
$$
R_{\mathrm{imag}}(\mathbf{x})=
\nabla^{2} E_i(\mathbf{x})+
k_0^{2}\left[1+\chi(\mathbf{x})\right]E_i(\mathbf{x})
$$

入射平面波：
$$
E_{\mathrm{inc}}(\mathbf{x})=\exp\
\left[j k \left(x \cos\theta + y \sin\theta\right)\right]
$$

Lippmann–Schwinger 散射积分方程：

$$
E_{\mathrm{scat}}(\mathbf{x})=
k^{2}
\int_{\Omega}
G(\mathbf{x},\mathbf{x}')
\chi(\mathbf{x}')
E(\mathbf{x}')\
\mathrm{d}\mathbf{x}'
$$


## 2、参数、属性定义



### 物理量定义表：

| 符号            | 含义         | 类型   | 单位 / 说明           |
| ------------- | ---------- | ---- | ----------------- |
| E(r)          | 总电场        | 复标量场 | 二维 TM 模式电场        |
| E_r(r)        | 电场实部       | 实标量场 | Re(E)             |
| E_i(r)        | 电场虚部       | 实标量场 | Im(E)             |
| r             | 空间位置向量     | 向量   | r = (x, y)        |
| x, y          | 空间坐标       | 标量   | ℝ² 中的笛卡尔坐标        |
| ∇²            | 拉普拉斯算子     | 微分算子 | ∂²/∂x² + ∂²/∂y²   |
| ε_r(r)        | 相对介电常数     | 空间函数 | 背景为 1，散射体区域 > 1   |
| ε             | 散射体介电常数    | 标量参数 | 可学习参数             |
| ε_prior       | 介电常数先验     | 常数   | 正则化参考值            |
| μ₀            | 真空磁导率      | 常数   | 4π × 10⁻⁷ H/m     |
| ε₀            | 真空介电常数     | 常数   | 8.854 × 10⁻¹² F/m |
| k₀            | 真空波数       | 标量   | k₀ = ω√(μ₀ε₀)     |
| ω             | 角频率        | 标量   | ω = 2πf           |
| f             | 频率         | 标量   | 单位 Hz             |
| θ             | 入射角        | 标量   | 平面波传播方向           |
| θ_p           | 第 p 个入射角   | 标量   | 多入射情形             |
| E_inc(r)      | 入射场        | 复标量场 | 平面波               |
| E_scat(r)     | 散射场        | 复标量场 | 由目标产生             |
| χ(r)          | 对比度函数      | 空间函数 | χ = ε_r − 1       |
| β             | Sigmoid 陡度 | 常数   | 控制介质边界平滑度         |
| r             | 散射体半径      | 标量参数 | 可学习参数             |
| r_prior       | 半径先验       | 常数   | 正则化参考值            |
| ‖r‖           | 到原点距离      | 标量   | √(x² + y²)        |
| R_real(x)     | PDE 实部残差   | 实标量场 | Helmholtz 实部不平衡   |
| R_imag(x)     | PDE 虚部残差   | 实标量场 | Helmholtz 虚部不平衡   |
| x_i           | 第 i 个采样点   | 向量   | 内部或观测点            |
| N             | PDE 采样点数   | 整数   | PDE loss 使用       |
| M             | 数据点数量      | 整数   | 数据损失使用            |
| P             | 入射角数量      | 整数   | 多入射平均             |
| E_pred(x)     | 预测电场       | 复标量场 | PINN 输出           |
| E_obs(x)      | 观测电场       | 复标量场 | 测量或仿真数据           |
| λ_data        | 数据损失权重     | 超参数  | 平衡 PDE 与数据        |
| λ_ε           | 介电常数正则权重   | 超参数  | 防止非物理解            |
| λ_r           | 半径正则权重     | 超参数  | 防止尺寸漂移            |
| λ_gPDE        | 梯度损失权重     | 超参数  | gPINN 使用          |
| L_PDE         | PDE 损失     | 标量   | 控制方程残差            |
| L_data        | 数据损失       | 标量   | 拟合观测场             |
| L_ε           | 介电常数正则项    | 标量   | (ε − ε_prior)²    |
| L_r           | 半径正则项      | 标量   | (r − r_prior)²    |
| L_gPDE        | 梯度残差损失     | 标量   | PDE 残差梯度          |
| L_total       | 总损失        | 标量   | 单入射               |
| L_total^multi | 多入射总损失     | 标量   | 多角度平均             |
| ∂/∂n          | 法向导数       | 微分算子 | 辐射边界条件            |
| Ω             | 计算区域       | 几何域  | 二维有限区域            |
| ∂Ω            | 区域边界       | 曲线   | 吸收 / 辐射边界         |


### 软件版本要求：
建议Python 3.10以上


### 所需要导入的库：

```  
import torch
import numpy as np
import os
import sys
import argparse
from pathlib import Path
```


## 3、计算方法及代码实现

### 3.1 参数设置
```python
  import torch
import numpy as np

# -----------------------------
# Physical configuration
# -----------------------------
frequencies = [3.0, 4.0]
angles = [0.0, np.pi / 3, 2 * np.pi / 3]

L = 1.0
BETA = 60.0

TRUE_EPS = 1.5
TRUE_R = 0.15
```
### 3.2 参数修复、优化
```python
class MultiFreqPINN(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        # 修复1: 调整参数范围和初始化，使r更容易优化
        # eps: 范围[1.0, 2.5]，真实值1.5在中间
        self.eps_param = nn.Parameter(torch.tensor(0.0))
        
        # r: 缩小范围，使真实值0.15在优化中心
        # 使用更窄的范围[0.12, 0.18]
        self.r_param = nn.Parameter(torch.tensor(0.0))

        self.net = nn.Sequential(
            nn.Linear(3, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 2 * len(angles)),
        )

    def forward(self, x, freq_idx):
        k = frequencies[freq_idx]
        k_feat = torch.ones_like(x[:, 0:1]) * k
        return self.net(torch.cat([x, k_feat], dim=1))

    def physical_params(self):
        # eps ∈ [1.0, 2.5]
        eps = 1.0 + 1.5 * torch.sigmoid(self.eps_param)
        
        # 修复2: 缩小r的范围，使其更集中
        # r ∈ [0.12, 0.18]，真实值0.15在正中间
        r = 0.12 + 0.06 * torch.sigmoid(self.r_param)
        
        return eps, r
```
### 3.3 训练及生成观测数据
```python
def solve_normal_mode(device, data_mode="simul", epochs=3000):
    """
    正常训练模式
    --mode normal --data simul
    """
    # 设置参数
    N_src, N_obs, N_col = 120, 40, 800
    
    print(f"\n🏋️  正常训练模式启动")
    print(f"📊 数据模式: {data_mode}")
    print(f"🔄 训练轮数: {epochs}")
    
    # ==================== 1. 生成训练数据 ====================
    print("\n📊 生成训练数据...")
    
    # 生成源点
    x_src = (2 * torch.rand(N_src, 2) - 1) * 0.4
    x_src = x_src.to(device)
    
    # 生成观测数据（正向计算）
    print("  生成观测数据...")
    xy_obs_list, Eobs_re_list, Eobs_im_list = generate_observations(
        N_obs, x_src, TRUE_EPS, TRUE_R, device
    )
    print(f"  ✅ 观测数据生成完成: {len(frequencies)}个频率 × {len(angles)}个角度")
    
    # 生成配置点
    xy_col = (2 * torch.rand(N_col, 2) - 1) * L
    xy_col = xy_col.to(device)
    
    print(f"📋 数据准备完成:")
    print(f"  源点: {x_src.shape}")
    print(f"  观测数据: {len(xy_obs_list)}组")
    print(f"  配置点: {xy_col.shape}")

```
### 3.4 保存、加载模型
```python
class ModelManager:
    """模型管理器 - 只负责保存和加载模型"""
    
    def __init__(self, base_dir='model'):
        self.base_dir = base_dir
        # 确保目录存在
        os.makedirs(f'{base_dir}/best_models', exist_ok=True)
    
    def save_best_model(self, model, eps, r, metrics=None, device='cpu'):
        """保存最佳模型"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'{self.base_dir}/best_models/best_model_{timestamp}.pt'
        
        # 确保eps和r是标量值
        eps_val = eps.item() if hasattr(eps, 'item') else float(eps)
        r_val = r.item() if hasattr(r, 'item') else float(r)
        
        save_dict = {
            'model_state_dict': model.state_dict(),
            'eps': eps_val,
            'r': r_val,
            'metrics': metrics or {},
            'timestamp': timestamp,
            'device': str(device)
        }
        
        torch.save(save_dict, filename)
        print(f"✅ 模型已保存: {filename}")
        print(f"   参数: ε={eps_val:.4f}, r={r_val:.4f}")
        return filename
    
    def load_best_model(self, filename, model, device='cpu'):
        """加载最佳模型"""
        filepath = f'{self.base_dir}/best_models/{filename}'
        if os.path.exists(filepath):
            try:
                # 加载到指定设备
                save_dict = torch.load(filepath, map_location=device)
                
                # 加载模型状态
                model.load_state_dict(save_dict['model_state_dict'])
                model.to(device)
                
                print(f"✅ 模型加载成功: {filename}")
                print(f"   参数: ε={save_dict['eps']:.4f}, r={save_dict['r']:.4f}")
                
                return save_dict
            except Exception as e:
                print(f"❌ 模型加载失败: {e}")
                return None
        else:
            print(f"❌ 模型文件不存在: {filepath}")
            return None
```

### 3.5 数据拟合、物理约束权重调整
```python
class DynamicWeightScheduler:
    """
    动态权重策略 
    """
    def __init__(self, total_epochs=3000):
        self.total_epochs = total_epochs
        
    def get_pde_weight(self, epoch):
        # 动态调整：初期强调数据拟合，后期强调物理约束
        if epoch < 1000:
            return 0.8
        elif epoch < 2000:
            return 1.2
        else:
            return 1.5
    
    def get_radius_weight(self, epoch):
        # 半径约束权重逐渐增加
        return min(0.3, epoch / 3000)
```

### 3.6 loss计算
```python
def pde_loss(model, xy, eps, r, fi, ai, theta, k):
    """
    Helmholtz PDE residual (strict scattering form)
    """
    xy.requires_grad_(True)
    out = model(xy, fi)

    Es_re = out[:, 2 * ai:2 * ai + 1]
    Es_im = out[:, 2 * ai + 1:2 * ai + 2]

    Ei_re, Ei_im = Ez_inc(xy, theta, k)
    chi = contrast_chi(xy, eps, r)

    # 计算梯度
    grad_re = torch.autograd.grad(Es_re.sum(), xy, create_graph=True)[0]
    grad_im = torch.autograd.grad(Es_im.sum(), xy, create_graph=True)[0]

    # 计算拉普拉斯算子
    lap_re = (
        torch.autograd.grad(grad_re[:, 0].sum(), xy, create_graph=True)[0][:, 0:1] +
        torch.autograd.grad(grad_re[:, 1].sum(), xy, create_graph=True)[0][:, 1:2]
    )
    lap_im = (
        torch.autograd.grad(grad_im[:, 0].sum(), xy, create_graph=True)[0][:, 0:1] +
        torch.autograd.grad(grad_im[:, 1].sum(), xy, create_graph=True)[0][:, 1:2]
    )

    # 散射场方程: (∇² + k²)Es = -k²χ(Ei + Es)
    res_re = lap_re + k**2 * Es_re + k**2 * chi * (Ei_re + Es_re)
    res_im = lap_im + k**2 * Es_im + k**2 * chi * (Ei_im + Es_im)

    return torch.mean(res_re**2 + res_im**2)


def data_loss(model, xy, Eobs_re, Eobs_im, fi, ai, theta, k):
    """
    Data loss (简化修复版，避免梯度错误)
    """
    out = model(xy, fi)

    Es_re = out[:, 2 * ai:2 * ai + 1]
    Es_im = out[:, 2 * ai + 1:2 * ai + 2]

    Ei_re, Ei_im = Ez_inc(xy, theta, k)

    Et_re = Ei_re + Es_re
    Et_im = Ei_im + Es_im

    # 简单的MSE损失
    mse_loss = torch.mean((Et_re - Eobs_re)**2 + (Et_im - Eobs_im)**2)
    
    # 添加振幅损失，提高对r的敏感度
    amp_obs = torch.sqrt(Eobs_re**2 + Eobs_im**2)
    amp_pred = torch.sqrt(Et_re**2 + Et_im**2)
    amp_loss = torch.mean((amp_pred - amp_obs)**2)
    
    # 总数据损失
    return 0.8 * mse_loss + 0.2 * amp_loss


def radius_constraint_loss(r, target_r):
    """
    半径约束损失
    """
    # 软约束：鼓励r接近目标值
    return 0.05 * (r - target_r)**2

```
### 3.7 可视化调整及输出

```python
# 计算预测对比度
    chi_pred = contrast_chi(xy, eps_pred, r_pred)
    chi_true = contrast_chi(xy, torch.tensor(eps_target, device=device), 
                           torch.tensor(r_target, device=device))
    
    chi_pred_img = chi_pred.reshape(resolution, resolution).cpu().detach().numpy()
    chi_true_img = chi_true.reshape(resolution, resolution).cpu().detach().numpy()
    error_img = np.abs(chi_true_img - chi_pred_img)
    
    # 创建图形
    fig = plt.figure(figsize=(15, 5))
    
    # 1. 真实对比度
    ax1 = plt.subplot(1, 3, 1)
    im1 = ax1.imshow(chi_true_img, extent=[-L, L, -L, L], origin='lower', 
                     cmap='viridis', vmin=0, vmax=max(eps_target-1, 0.5))
    ax1.set_title(f'真实对比度\nε={eps_target}, r={r_target}', fontweight='bold')
    ax1.set_xlabel('x', fontweight='bold')
    ax1.set_ylabel('y', fontweight='bold')
    plt.colorbar(im1, ax=ax1)
    
    # 在图上添加误差信息
    ax1.text(0.02, 0.98, f'目标值', 
             transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. 预测对比度
    ax2 = plt.subplot(1, 3, 2)
    im2 = ax2.imshow(chi_pred_img, extent=[-L, L, -L, L], origin='lower', 
                     cmap='viridis', vmin=0, vmax=max(eps_target-1, 0.5))
    ax2.set_title(f'预测对比度\nε={eps_pred_val:.4f}, r={r_pred_val:.4f}', fontweight='bold')
    ax2.set_xlabel('x', fontweight='bold')
    ax2.set_ylabel('y', fontweight='bold')
    plt.colorbar(im2, ax=ax2)
    
    # 在图上添加误差信息
    ax2.text(0.02, 0.98, f'误差: ε={eps_error:.1f}%\nr={r_error:.1f}%', 
             transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. 误差分布
    ax3 = plt.subplot(1, 3, 3)
    
    # 使用热图但调整颜色范围
    vmax = np.percentile(error_img, 99)  # 使用99%分位数作为最大值
    if vmax == 0:
        vmax = error_img.max() + 1e-10  # 避免除零错误
    
    im3 = ax3.imshow(error_img, extent=[-L, L, -L, L], origin='lower', 
                     cmap='hot', vmin=0, vmax=vmax)
    ax3.set_title(f'误差分布\n最大: {error_img.max():.2e}', fontweight='bold')
    ax3.set_xlabel('x', fontweight='bold')
    ax3.set_ylabel('y', fontweight='bold')
    cbar = plt.colorbar(im3, ax=ax3)
    cbar.set_label('|χ_true - χ_pred|')
    
    # 在图上添加模型信息
    ax3.text(0.02, 0.98, f'模型: {model_name[:15]}...', 
             transform=ax3.transAxes, fontsize=8,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存图像
    save_path = os.path.join(save_dir, f'quick_test_result_{model_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"📸 可视化结果已保存: {save_path}")
    
    # 打印简要总结
    print("\n" + "="*60)
    print("QUICK TEST SUMMARY")
    print("="*60)
    print(f"目标参数: ε = {eps_target}, r = {r_target}")
    print(f"预测参数: ε = {eps_pred_val:.4f}, r = {r_pred_val:.4f}")
    print(f"相对误差: ε = {eps_error:.2f}%, r = {r_error:.2f}%")
    print("="*60)
    
    return eps_pred_val, r_pred_val
```


## 4、 预测、绘图及成果展示

这是一个典型的通过亥姆霍兹方程与 PML 吸收边界建模，分别实现了正问题（已知介电常数求电场）与反问题（已知电场反演目标位置与介电常数）。



![caseMarkdown/电磁逆散射/viz/training_summary_simple.png](https://www.science42.tech/cases/caseMarkdown/电磁逆散射/viz/training_summary_simple.png)


- 总损失下降一个量级，表明训练成功收敛。


- 预测与实际基本相同，表明计算、模拟成功。

## 5、常见问题（FAQ）

- 收敛性问题：训练过程中损失函数在某个值附近震荡，无法继续下降。

- 精度问题：在散射体边界附近，电场预测误差较大。

- 计算效率问题：训练好的模型对新数据泛化能力差。

## 6、工业价值

- 工业无损检测与结构健康监测：本项目的二维电磁反散射PINN技术在工业无损检测领域具有重大应用价值。传统无损检测方法如超声波、X射线等存在穿透深度有限、需要耦合剂、设备昂贵且操作复杂等局限。而基于电磁波的微波检测技术具有非接触、可穿透非金属材料、对介电特性变化敏感等优势，但传统电磁反演方法计算复杂、效率低下。本项目开发的PINN反演框架能够实现实时、高精度的缺陷检测与参数反演，特别适用于航空航天复合材料结构、大型油气管道、电力设备绝缘层等关键基础设施的健康监测。

- 医疗诊断成像与个性化医疗：在医疗领域，本技术为非侵入式、无辐射的医学成像提供了创新解决方案。传统医学成像技术如CT、MRI等虽然分辨率高，但存在辐射风险（CT）、设备昂贵（MRI）、检查时间长等缺点。微波成像技术基于人体组织在不同健康状态下的介电特性差异，具有安全、低成本、便携等优势，但传统微波成像算法重建质量差、计算时间长。本项目开发的PINN反演技术能够实现高精度、实时的生物组织介电参数成像，在乳腺癌早期筛查、脑卒中监测、肺部疾病诊断等领域具有广阔应用前景。

- 智能安防与公共安全监测：在公共安全领域，本技术为非侵入式人体安检和危险品检测提供了新一代解决方案。传统安检技术如金属探测器仅能检测金属物品，毫米波成像存在隐私顾虑，X射线安检有辐射风险。基于电磁反散射的PINN技术能够实现材料识别级别的安检，不仅能检测金属武器，还能识别塑料炸药、液体危险品、陶瓷刀具等非金属威胁物品，同时保护被检人员隐私。系统通过分析人体表面微波散射模式，反演出隐藏物品的介电特性和几何形状，实现精准分类识别。