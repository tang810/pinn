# 嵌入式液冷芯片流固热耦合求解


## 快速使用

您可跳转[使用示例](#使用示例)查看如何使用秋月白平台快速执行该类问题，以下是实现逻辑详解：

## 背景介绍

针对嵌入式微流道芯片内部复杂、高梯度、非均匀的瞬态热场分布特征，传统数值模拟方法存在网格划分复杂、求解耗时、算力成本高昂等问题，而常规有限测点插值方法难以精准刻画芯片内部微细结构带来的局部高温与热梯度突变。为此，本案例采用物理信息神经网络（PINN）对嵌入式微流道芯片开展三维瞬态热场重建研究。PINN无需依赖大量高精度标注数据，依托热传导控制方程构建物理约束，通过拉丁超立方采样在三维空间与时域范围内均匀生成域内残差点、初始条件点与边界条件点，使神经网络在训练过程中同时拟合观测温度数据并满足传热物理规律。相较于传统数值仿真，PINN摆脱了结构化网格限制，能够以连续函数形式表征芯片全域温度场，精准捕捉微流道固液交界面、局部高热集中区域的温度突变特征。基于少量离散监测数据，该方法可实现嵌入式微流道芯片四维时空热场的高精度、高分辨率逆向重建，清晰复现不同时刻下芯片内部热量积聚、扩散与演化过程，为微流控芯片散热优化、热失效分析以及结构嵌入式换热设计提供高效、可靠的热场表征技术手段。


## 1.嵌入式液冷芯片几何建模
本文建立的嵌入式液冷芯片结构如下图所示，芯片整体尺寸为376.8μm×376.8μm×200μm，其中在芯片的上半部分有三条微流道供冷却水流过。该流道采用平行微通道阵列设计，微流道在散热器内部呈纵向延伸，贯穿整个散热器块体，流体在通道内流动时，通过对流换热带走基底的热量。单个流道横截面数据为72μm×72μm，左右两个流道与芯片的外壁面间距为8.4μm。流道中通有冷却水，控制入口流速为0.1m/s。芯片热源设置为二维形式，其几何位置位于距芯片底面50μm处。

![caseMarkdown/case_915caa5b/viz/f028c0b031fa5e4f90985c425230d0b8.png](https://www.science42.tech/cases/caseMarkdown/case_915caa5b/viz/f028c0b031fa5e4f90985c425230d0b8.png)



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




## 4. PINN 损失函数形式



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