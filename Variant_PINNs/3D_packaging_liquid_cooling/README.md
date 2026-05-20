# 3D-IC叠层微通道液冷计算

## 快速使用

您可跳转[使用示例](#使用示例)查看如何使用秋月白平台快速执行该类问题，以下是实现逻辑详解：

## 背景介绍
随着 3D 封装芯片集成度与功率密度的不断提升，传统风冷散热已难以满足其高热流密度下的温控需求，液冷微流道散热器凭借高效的对流换热能力成为主流散热方案之一。然而，3D 封装芯片的液冷共轭传热问题涉及复杂的三维固体导热、流体对流换热以及固液界面热流耦合，传统数值仿真方法（如 CFD）存在计算成本高、网格依赖性强、多物理场耦合求解效率低等问题。物理信息神经网络（PINN）作为一种数据驱动与物理约束相结合的求解框架，能够直接将控制方程、边界条件与界面耦合条件嵌入损失函数，无需复杂网格剖分即可求解三维温度场，为 3D 封装芯片液冷热管理提供了一种高效、无网格的求解思路。通过构建同时满足固体导热方程、流体对流扩散方程以及固液界面温度与热流连续条件的 PINN 模型，可实现对微流道散热器内部及芯片表面温度场的快速预测，为液冷结构优化与热可靠性评估提供理论支撑。


## 1. 3D封装嵌入式液冷芯片几何建模
本文建立的3D封装嵌入式液冷芯片结构如下图所示，图中的方形空洞表示流体流过的区域即微流道，每个微流道的尺寸为0.4mm×1.6mm，微流道共分上下两层，中间间隔的部分是用于放置芯片发热源，两层流道的外侧也均有一块区域用于方式芯片发热源，该模型是模拟实际的3D芯片封装工况下三层发热源和双层微流道冷却之间的流热耦合现象。

![caseMarkdown/case_7b43c3b1/viz/37fd2555f81c76b424c5825ac36fed65.png](https://www.science42.tech/cases/caseMarkdown/case_7b43c3b1/viz/37fd2555f81c76b424c5825ac36fed65.png)

---

## 2. 控制方程

### 2.1 固体域控制方程（热传导）
#### 稳态形式
$$
\nabla \cdot \left( k_s \nabla T_s \right) + q_{\text{vol}} = 0
$$

#### 瞬态形式
$$
\rho_s c_{p,s} \frac{\partial T_s}{\partial t} = \nabla \cdot \left( k_s \nabla T_s \right) + q_{\text{vol}}
$$

### 2.2 流体域控制方程（对流-扩散）
假设流体速度场 $\mathbf{u} = (u, v, w) \$已知（可通过仿真或解析解给定）。

#### 稳态形式
$$
\rho_f c_{p,f} \left( \mathbf{u} \cdot \nabla T_f \right) = \nabla \cdot \left( k_f \nabla T_f \right)
$$

#### 瞬态形式
$$
\rho_f c_{p,f} \left( \frac{\partial T_f}{\partial t} + \mathbf{u} \cdot \nabla T_f \right) = \nabla \cdot \left( k_f \nabla T_f \right)
$$

符号说明：
- $\ T_f \$：流体温度场
- $\ k_f \$：流体导热系数
- $\\rho_f \$：流体密度
- $\ c_{p,f} \$：流体定压比热
- $ \mathbf{u} \$：流体速度矢量场

---

### 2.3 固-流共轭界面条件
在微通道壁面处，温度与热流需连续：

$\
T_s = T_f \quad (\text{界面温度连续})
\$

$\
k_s \frac{\partial T_s}{\partial n} = k_f \frac{\partial T_f}{\partial n} \quad (\text{界面热流连续})
\$

符号说明：
- \( n \)：界面法向方向

---

## 3. 典型边界条件

### 3.1 热源边界（芯片接触面）
给定热流密度 $\ q''_{\text{in}} \$：

$$
 k_s \frac{\partial T_s}{\partial n} = q''_{\text{in}}
$$

### 3.2 流体入口边界
给定入口温度 $\ T_{\text{in}} \$：
$$
T_f = T_{\text{in}}
$$

### 3.3 流体出口边界
通常采用 Neumann 边界（充分发展流）：
$$
\frac{\partial T_f}{\partial n} = 0
$$

### 3.4 散热器外边界（与环境接触）
#### 对流换热边界
$$
-k_s \frac{\partial T_s}{\partial n} = h_{\text{env}} \left( T_s - T_{\text{env}} \right)
$$

#### 绝热边界
$$
\frac{\partial T_s}{\partial n} = 0
$$

符号说明：
- $\ h_{\text{env}} \$：环境对流换热系数
- $\ T_{\text{env}} \$：环境温度
- 
- 
- ## 4. PINN 损失函数形式



### 4.1 物理方程残差损失
#### 固体域 PDE 残差损失

$$
\mathcal{L}_{\text{solid}} = \frac{1}{N_s} \sum_{i=1}^{N_s} \left[\rho_s c_{p,s} \frac{\partial T_s}{\partial t}- \nabla \cdot \left( k_s \nabla T_s \right)- q_{\text{vol}}\right]^2 \bigg|_{(x_i,y_i,z_i,t_i)}
$$

#### 流体域 PDE 残差损失

$$
\mathcal{L}_{\text{fluid}} = \frac{1}{N_f} \sum_{i=1}^{N_f} \left[\rho_f c_{p,f} \left( \frac{\partial T_f}{\partial t} + \mathbf{u} \cdot \nabla T_f \right)- \nabla \cdot \left( k_f \nabla T_f \right)\right]^2 \bigg|_{(x_i,y_i,z_i,t_i)}
$$

#### 总物理损失

$$
\mathcal{L}_{\text{physics}} = \mathcal{L}_{\text{solid}} + \mathcal{L}_{\text{fluid}}
$$

### 4.2 共轭界面损失

$$
\mathcal{L}_{\text{interface}} = \mathcal{L}_{T,\text{int}} + \mathcal{L}_{q,\text{int}}
$$

#### 温度连续损失

$$
\mathcal{L}_{T,\text{int}} = \frac{1}{N_{\text{int}}} \sum_{i=1}^{N_{\text{int}}}
\left( T_s(x_i,y_i,z_i,t_i) - T_f(x_i,y_i,z_i,t_i) \right)^2
$$

#### 热流连续损失
$$
\mathcal{L}_{q,\text{int}} = \frac{1}{N_{\text{int}}} \sum_{i=1}^{N_{\text{int}}}\left( k_s \frac{\partial T_s}{\partial n}\bigg|_{(x_i,y_i,z_i,t_i)}- k_f \frac{\partial T_f}{\partial n}\bigg|_{(x_i,y_i,z_i,t_i)} \right)^2
$$



---

### 4.3 边界条件损失

$$
\mathcal{L}_{\text{bc}} = \mathcal{L}_{\text{bc,heat}} + \mathcal{L}_{\text{bc,in}} + \mathcal{L}_{\text{bc,out}} + \mathcal{L}_{\text{bc,env}}
$$

#### 热源边界（热流）损失
$$
\mathcal{L}_{\text{bc,heat}} = \frac{1}{N_{\text{heat}}} \sum_{i=1}^{N_{\text{heat}}}
\left( -k_s \frac{\partial T_s}{\partial n}\bigg|_{(x_i,y_i,z_i,t_i)} - q''_{\text{in}} \right)^2
$$

#### 流体入口（温度）损失
$$
\mathcal{L}_{\text{bc,in}} = \frac{1}{N_{\text{in}}} \sum_{i=1}^{N_{\text{in}}}
\left( T_f(x_i,y_i,z_i,t_i) - T_{\text{in}} \right)^2
$$

#### 流体出口（梯度）损失
$$
\mathcal{L}_{\text{bc,out}} = \frac{1}{N_{\text{out}}} \sum_{i=1}^{N_{\text{out}}}
\left( \frac{\partial T_f}{\partial n}\bigg|_{(x_i,y_i,z_i,t_i)} \right)^2
$$

#### 外边界对流损失
$$
\mathcal{L}_{\text{bc,env}} = \frac{1}{N_{\text{env}}} \sum_{i=1}^{N_{\text{env}}}
\left( -k_s \frac{\partial T_s}{\partial n}\bigg|_{(x_i,y_i,z_i,t_i)}- h_{\text{env}} \left( T_s(x_i,y_i,z_i,t_i) - T_{\text{env}} \right) \right)^2
$$