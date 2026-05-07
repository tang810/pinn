# Equation Card: Maxwell Equations (Time-Domain, Linear Media)

## 0. Metadata (REQUIRED)
- id: maxwell_time_domain_linear_media
- title: Maxwell Equations (Time-Domain, Linear Media)
- aliases: Maxwell方程组, 时域麦克斯韦方程, 电磁场方程
- type: PDE system (hyperbolic + constraints)
- canonical_fields: E, H (or E only via curl-curl)
- keywords: electromagnetics, wave propagation, curl, divergence constraints, PEC/PMC, impedance, radiation, PML
- typical_dimensions: 3D (2D via TE/TM reduction)
- related_components:
  - components/boundary_conditions.md
  - components/initial_conditions.md
  - components/source_terms.md

## 1. Background (REQUIRED)
Maxwell 方程组描述电磁场在介质中的时空演化，是电磁波传播、散射、谐振腔、天线、微波器件、光子学等问题的基础模型。

在数值/学习求解中常见两类表述：
1) **一阶旋度系统（E-H）**：保持物理结构清晰，但需要同时满足旋度方程与散度约束。  
2) **二阶 curl-curl（E-only 或 H-only）**：便于形成单场 PDE，但更依赖合适的边界条件与介质参数一致性。

本卡片以**线性、各向同性介质**为默认（可扩展到各向异性/色散/非线性）。

## 2. Variables & Parameters (REQUIRED)

### 2.1 Primary Fields
- 电场：$$\mathbf{E}(\mathbf{x},t)\in \mathbb{R}^3$$  
- 磁场：$$\mathbf{H}(\mathbf{x},t)\in \mathbb{R}^3$$

### 2.2 Auxiliary Fields (constitutive)
- 电位移：$$\mathbf{D}(\mathbf{x},t)$$
- 磁感应：$$\mathbf{B}(\mathbf{x},t)$$

### 2.3 Sources / Charges
- 体电流密度：$$\mathbf{J}(\mathbf{x},t)$$（可含外加源 $$\mathbf{J}_s$$）
- 体电荷密度：$$\rho(\mathbf{x},t)$$

### 2.4 Material Parameters (linear isotropic default)
- 介电常数：$$\varepsilon(\mathbf{x})$$
- 磁导率：$$\mu(\mathbf{x})$$
- 电导率：$$\sigma(\mathbf{x})$$

常用本构关系（线性各向同性）：
$$
\mathbf{D}=\varepsilon \mathbf{E},\qquad
\mathbf{B}=\mu \mathbf{H},\qquad
\mathbf{J}=\sigma \mathbf{E} + \mathbf{J}_s.
$$

### 2.5 Derived Quantities (optional)
- 波阻抗：$$\eta=\sqrt{\mu/\varepsilon}$$
- 相速度：$$c=1/\sqrt{\mu\varepsilon}$$

## 3. Governing Equation (REQUIRED)

### 3.1 Time-Domain Maxwell System (canonical)
在区域 $$\Omega\subset\mathbb{R}^d,\ d=3$$ 上：
$$
\nabla\times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t},
\qquad
\nabla\times \mathbf{H} = \mathbf{J} + \frac{\partial \mathbf{D}}{\partial t},
$$
并伴随散度约束：
$$
\nabla\cdot \mathbf{D} = \rho,
\qquad
\nabla\cdot \mathbf{B} = 0.
$$
配合电荷守恒（连续性方程）：
$$
\frac{\partial \rho}{\partial t} + \nabla\cdot \mathbf{J} = 0.
$$

### 3.2 Common Reduced Form: Curl-Curl (E-only, frequency or time)
在频域（$$e^{j\omega t}$$ 约定）下常用：
$$
\nabla\times \left(\mu^{-1}\nabla\times \mathbf{E}\right) - \omega^2 \varepsilon \mathbf{E}
= j\omega \mathbf{J}_s \quad (\text{lossless})
$$
含电导（等效复介电）可写为：
$$
\nabla\times \left(\mu^{-1}\nabla\times \mathbf{E}\right) - \omega^2 \varepsilon_c \mathbf{E}
= j\omega \mathbf{J}_s,
\qquad
\varepsilon_c=\varepsilon - j\frac{\sigma}{\omega}.
$$

> 选择一阶系统还是 curl-curl，取决于：目标（传播/散射/本征）、边界类型、观测量、以及实现（FDTD/FEM/PINN）。

## 4. Initial and Boundary Conditions (REQUIRED)

### 4.1 Initial Conditions (time-domain required)
需要给定：
$$
\mathbf{E}(\mathbf{x},0)=\mathbf{E}_0(\mathbf{x}),\qquad
\mathbf{H}(\mathbf{x},0)=\mathbf{H}_0(\mathbf{x}).
$$
并建议满足（或弱满足）散度一致性：
$$
\nabla\cdot(\varepsilon \mathbf{E}_0)=\rho(\mathbf{x},0),\qquad
\nabla\cdot(\mu \mathbf{H}_0)=0.
$$

### 4.2 Boundary Conditions (choose per physics)
在边界 $$\partial\Omega$$ 上常用：

**(A) PEC (Perfect Electric Conductor)**
$$
\mathbf{n}\times \mathbf{E} = \mathbf{0}\quad \text{on } \partial\Omega.
$$

**(B) PMC (Perfect Magnetic Conductor)**
$$
\mathbf{n}\times \mathbf{H} = \mathbf{0}\quad \text{on } \partial\Omega.
$$

**(C) Impedance / Leontovich (approx absorbing)**
常见形式（方向与符号按约定调整）：
$$
\mathbf{n}\times \mathbf{H} = \frac{1}{\eta}\,\mathbf{n}\times(\mathbf{n}\times \mathbf{E}).
$$

**(D) Radiation / Silver–Müller (open-domain outgoing)**
在远场截断边界：
$$
\mathbf{n}\times \mathbf{H} \approx \frac{1}{\eta}\,\mathbf{E}_t,
\qquad \mathbf{E}_t=\mathbf{E}- (\mathbf{E}\cdot\mathbf{n})\mathbf{n}.
$$

**(E) PML (Perfectly Matched Layer)**
以吸收层形式实现（通常作为域扩展与坐标拉伸，不写成简单 BC）。

> 说明：不同 BC 对应不同物理场景（腔体/导体/开域散射）。必须明确“整个边界”如何覆盖。

## 5. Domain & Typical Setups (REQUIRED)

### 5.1 Typical Domains
- 规则域：盒形腔体、圆柱/球域（benchmark 友好）
- 复杂几何：导体结构、天线、波导、散射体（工程常见）
- 开域问题：散射/辐射，常用 PML 或 radiation BC 截断

### 5.2 Typical Problem Types
- **Propagation (time-domain)**：脉冲/调制源 + absorbing/PML
- **Scattering (frequency-domain)**：入射场 + 散射体 + PML
- **Eigenmodes**：腔体/波导本征频率（通常无源）

### 5.3 Dimensional Reduction (2D TE/TM)
若几何与介质在 $$z$$ 方向不变，可做 TE/TM 约化（例如只保留 $$E_z$$ 或 $$H_z$$ 标量场），从而得到类 Helmholtz/curl-curl 的 2D 形式。

## 6. Inverse Problems / Identification (REQUIRED if meaningful; otherwise write "N/A")
常见反问题（有意义）：
- **介质参数反演**：识别 $$\varepsilon(\mathbf{x}), \mu(\mathbf{x}), \sigma(\mathbf{x})$$（成像/无损检测）
- **源项识别**：识别 $$\mathbf{J}_s(\mathbf{x},t)$$ 或入射波参数
- **几何/边界识别**：散射体形状、边界阻抗参数
- **模式识别**：腔体/波导本征频率或等效材料参数

观测常见为：
- 时域：$$\mathbf{E}(\mathbf{x}_m,t)$$ 或 $$\mathbf{H}(\mathbf{x}_m,t)$$ 传感器序列
- 频域：S-parameters、近场/远场采样、散射截面等

## 7. Constraint Decomposition (for agent compilation) (REQUIRED)

### 7.1 PDE Residuals (time-domain, E-H form)
定义残差（用于 PINN / 约束优化）：
- Faraday（旋度-电）：
$$
\mathbf{r}_1 = \nabla\times \mathbf{E} + \frac{\partial (\mu \mathbf{H})}{\partial t}
$$
- Ampère–Maxwell（旋度-磁）：
$$
\mathbf{r}_2 = \nabla\times \mathbf{H} - \sigma \mathbf{E} - \mathbf{J}_s - \frac{\partial (\varepsilon \mathbf{E})}{\partial t}
$$
- Gauss（电）：
$$
r_3 = \nabla\cdot (\varepsilon \mathbf{E}) - \rho
$$
- Gauss（磁）：
$$
r_4 = \nabla\cdot (\mu \mathbf{H})
$$
- Charge continuity（可选强化）：
$$
r_5 = \frac{\partial \rho}{\partial t} + \nabla\cdot(\sigma\mathbf{E}+\mathbf{J}_s)
$$

### 7.2 Boundary Residuals (examples)
- PEC：$$\mathbf{r}_{bc}^{PEC}=\mathbf{n}\times\mathbf{E}$$
- PMC：$$\mathbf{r}_{bc}^{PMC}=\mathbf{n}\times\mathbf{H}$$
- Impedance：$$\mathbf{r}_{bc}^{imp}=\mathbf{n}\times\mathbf{H}-\frac{1}{\eta}\mathbf{n}\times(\mathbf{n}\times\mathbf{E})$$

### 7.3 Initial Residuals
- $$\mathbf{r}_{ic}^E=\mathbf{E}(\mathbf{x},0)-\mathbf{E}_0(\mathbf{x})$$
- $$\mathbf{r}_{ic}^H=\mathbf{H}(\mathbf{x},0)-\mathbf{H}_0(\mathbf{x})$$

> 实务建议：散度约束可作为 soft penalty 或通过结构化表示/投影方法处理；开域问题优先 PML 或近似吸收 BC。

## 8. Minimal Problem Spec Checklist (REQUIRED)
求解前必须明确（最小集合）：
1) 选择表述：时域一阶（E-H）还是频域 curl-curl  
2) 空间维度：2D TE/TM 还是 3D 全矢量  
3) 介质模型：$$\varepsilon,\mu,\sigma$$ 是否空间变、是否各向异性/色散  
4) 域 $$\Omega$$ 与边界划分（PEC/PMC/impedance/radiation/PML 等）  
5) 源项：$$\mathbf{J}_s$$ / 入射场 / 边界激励 的形式与支撑区域  
6) 时域初值（若时域）：$$\mathbf{E}_0,\mathbf{H}_0$$（及可选散度一致性）  
7) 观测量与目标：场值、功率、S 参数、散射截面、本征频率等  
8) 反问题（若有）：待识别参数 + 先验范围/正则项/可观测性假设

## 9. Minimal Example (toy specification, NO CODE) (REQUIRED)
**Toy 1：矩形腔体本征/瞬态响应（PEC）**
- 域：$$\Omega=[0,L_x]\times[0,L_y]\times[0,L_z]$$  
- 介质：常数 $$\varepsilon,\mu$$，无耗散 $$\sigma=0$$  
- 边界：PEC，$$\mathbf{n}\times\mathbf{E}=0$$ on $$\partial\Omega$$  
- 源：可选脉冲电流 $$\mathbf{J}_s(\mathbf{x},t)=\mathbf{e}_z\,\exp(-|\mathbf{x}-\mathbf{x}_0|^2/a^2)\,g(t)$$  
- 初值：$$\mathbf{E}_0=\mathbf{0},\ \mathbf{H}_0=\mathbf{0}$$  
- 目标：得到 $$\mathbf{E},\mathbf{H}$$ 的时域响应；或在频域求腔体模式

**Toy 2：开域散射（PML）**
- 域：散射体外部开域，用 PML 扩展截断  
- 入射：平面波 $$\mathbf{E}^{inc}$$  
- 目标：散射场 $$\mathbf{E}^{sc}$$ 与远场指标（RCS）

## 10. References (REQUIRED)
- J. D. Jackson, *Classical Electrodynamics*, 3rd ed.
- A. Taflove, S. C. Hagness, *Computational Electrodynamics: The Finite-Difference Time-Domain Method*.
- P. Monk, *Finite Element Methods for Maxwell’s Equations*.
- W. C. Chew, *Waves and Fields in Inhomogeneous Media*.

## Chinese Notes (Optional)
- 若你希望知识库更“可拼装”，建议在 components 中补充：PEC/PMC/Impedance/Radiation/PML 的标准条目（Canonical Form + Notes），这样 EquationRetriever 只需抽“最小片段”即可。
- 若使用 PINN：常见挑战包括高频振荡、边界反射、散度约束漂移、以及开域截断误差；可用分域、频域训练、特征网络/傅里叶特征、或物理一致的约束权重调度缓解。
