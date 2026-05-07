# 🎯 PINNs4Science：科学与工程的 PINNs 解决方案合集
**PINNs4Science** 是一个基于物理信息神经网络（PINNs）及其多种变体的开源案例库，涵盖多领域物理问题的求解示例。


> **Pi**: Physics‑Informed  
> **NNs**: Neural Networks  
> **4Science**: 面向科学与工程仿真的通用平台 
---
## 🗺️ 项目导览
该项目整合了**固体力学**、**热力学**、**流体力学**、**电磁学**、**地质学**、**多种 PINNs 技术**以及**基础数学/物理方程**七大方向，帮助科研人员与工程师快速上手 Physics‑Informed AI，并应用于复杂物理场景。


<details open>
<summary markdown="span">
  <h3>固体力学</h3><br/>
  在固体力学领域，三维弹性体的位移与应力预测对结构安全评估、材料疲劳分析和工程设计优化至关重要。传统有限元方法在复杂边界条件、非线性材料和高维参数空间下常常计算量巨大且难以实时反馈。PINNs4Science 通过融合 gPINNs、SPINN 等先进架构，以及物理驱动的损失约束和子域并行技术，实现了无需网格且高效准确的物理场预测，为智能结构健康监测和新材料设计提供了强大工具。<br/><br/>
</summary>

| 示例 | 场景说明 |
| --- | --- |
| 三维弹性力学应力与位移预测([简化结构](./Solid_Mechanics/3D_ElasticForce_Deformation)/[卫星结构](./Solid_Mechanics/3D_ElasticForce_Deformation_卫星结构))| 准确预测固体结构在受力情况下的内部应力与变形，对于确保工程结构（如桥梁、航空航天部件、机械零件等）的安全可靠运行和优化设计至关重要。结构的响应行为通常由弹性力学偏微分方程描述。当结构几何、载荷或边界条件复杂时，求解这些方程可能非常具有挑战性。<br/>本项目致力于应用**物理信息神经网络 (PINN)** 技术，求解一个定义在三维域内的线性弹性力学问题。目标是预测该域内任意点的位移场 (u, v, w) 以及相应的应力张量 (σ) 分布。PINN 的核心思想是将控制物理定律（在此为 **Navier-Cauchy 弹性力学方程**）和边界条件直接嵌入到神经网络的损失函数中进行训练。 |
|[3D 线性弹性力学：翼型结构位移约束的物理信息神经网络（PINN）求解](./Solid_Mechanics/翼型结构位移约束的物理信息神经网络求解)| 本项目专注于利用**物理信息神经网络（PINN）**高效解决**翼型结构**在复杂载荷下的三维线性弹性力学问题。特别是，通过施加精确的**位移边界约束**，实现对翼型变形的精确预测。传统数值方法在处理复杂几何和边界条件时面临网格划分的挑战，而PINN通过将控制方程和位移边界条件直接编码到神经网络的损失函数中，提供了一种无网格的解决方案。项目目标是精确预测三维翼型在给定载荷下的位移场 $(u_x, u_y, u_z)$。|
| [3D 线性弹性力学：翼型结构力约束的物理信息神经网络求解](./Solid_Mechanics/翼型结构力约束的物理信息神经网络求解) | 在三维立方体模型顶部施加分布式法向载荷 $T(x,y)=\cos(\tfrac{\pi x}{2})\cos(\tfrac{\pi y}{2})$，并对其余三个面施加零位移约束，通过 PINNs 物理驱动方法联合求解平衡方程、几何方程与本构关系，预测全场位移 $u,v,w$ 与应力分量 $\sigma_{ij}$，用于结构安全与变形评估。 |
|[二维弹性力学基函数预计算与在线求解](./Solid_Mechanics/二维弹性力学基函数预计算与在线求解)|本项目旨在解决二维弹性力学问题中的参数化挑战，通过创新的数化物理信息神经网络（GPT-PINN）方法实现高效求解。针对传统PINN每次参数变化需重新训练的巨大计算开销，GPT-PINN将求解分解为离线预计算基函数和在线优化元网络两个阶段，以快速逼近新参数下的解。项目解决具有解析解的二维线性弹性力学问题，预测域内任意点的位移场 (u_x, u_y) 和应力张量分量 $\sigma_{xx}$, $\sigma_{yy}$, $\sigma_{xy}$ |

</details>


<details open>
<summary markdown="span"><h3>热力学</h3></summary>

| 示例 | 场景说明 |
| --- | --- |
|[卫星温度场稳态空间建模 ](./Thermodynamics/STSS_without_time)|预测卫星各个位置的温度对于维持正常工作至关重要。<br/>当卫星位于地球阴影面（无太阳辐照）时，向外辐射热量导致温度急剧下降，可能损耗电子元件寿命。<br/>本案例使用 **STSS**（基于经典 PINNs 的自适应加权物理损失）预测卫星稳态温度空间分布。|
|[卫星温度场时空建模 ](./Thermodynamics/Pinnsformer)|在无太阳辐照条件下预测温度时空演化，同样需应对阴影面快速降温风险。<br/>本案例采用 **Pinnsformer**（引入自注意力机制的 Transformer-PINNs），增强长时序和空间耦合建模能力。|
|[有外界热源工况下的卫星温度场模拟](./Thermodynamics/STSS_withsun)|当卫星从阴影面进入太阳辐射区，向阳面迅速升温且内部热源叠加，若无有效散热，仪器易过热受损。<br/>本案例用 **STSS** 处理瞬态辐照边界和内部热源，预测温度场演化。|
|[有外界热源工况下的卫星温度场时空建模](./Thermodynamics/Pinnsformer_withsun)|结合太阳辐照与内部热源的复杂时空热耦合，对温度场进行高精度仿真。<br/>本案例基于 **Pinnsformer** 的自注意力编码器，捕捉多尺度时空依赖，提升预测精度。 |


</details>


<details open>
<summary markdown="span" ><h3>流体力学</h3> </summary>

| 示例 | 场景说明 |
| --- | --- |
|[ PIKAN 用于求解方腔流问题 ](./Fluid_Mechanics/cavity_flow)|在二维方形腔体中，顶部壁以恒定水平速度驱动流体循环，其余三壁为无滑移固壁。PIKAN（Physics-Informed Kolmogorov–Arnold Network）将物理约束引入 Kolmogorov–Arnold 网络（KAN）结构中，具备优秀的表达能力，能精准逼近高雷诺数下边界层和涡结构。该模型快速求解稳态速度场和压力场，并支持流线、切片及误差分布等可视化功能。|
|[PINN对流体绕圆柱的流动模拟 ](./Fluid_Mechanics/cylinder_flow)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|在二维通道中放置圆柱，入口施加恒定来流速度，壁面无滑移、出口为零压边界。基于 PINNs 的物理信息残差网络直接求解 Navier–Stokes 方程，将速度和压力残差引入损失，实现涡脱落与涡街结构捕捉，并预测速度场、压力场及涡量分布，适用于流动分离和气动特性分析。|
|[强迫对流问题模拟 ](./Fluid_Mechanics/convection/hard_convection)|在外部机械驱动（如泵、风扇）下，强迫流体流动实现热量传递，基于 PINNs 联立求解 Navier–Stokes 和能量守恒方程，对流场与温度场耦合建模，适用于冷却系统、换热器等工程设计。|
|[混合对流问题模拟 ](./Fluid_Mechanics/convection/mixed_convection)|在传热过程中同时存在自然对流（浮力驱动）和强迫对流（机械驱动）的耦合机制，基于 PINNs 联立求解 Navier–Stokes、能量守恒及浮力源项方程，精准模拟通风系统、电子冷却与环境控制中的温度场与流场分布。|
|[KdV方程内孤立波问题中的应用：算子学习与参数估计 ](./Fluid_Mechanics/KdV_Korteweg-de-Vries)|面向两层流体内部孤立波的 KdV 方程，利用 PINNs 端到端框架：(1) 算子学习构建参数→波形映射，实现未知参数下的前向演化预测，1000 点训练时误差低至 10⁻⁴；(2) 结合稀疏观测和物理正则化反演非线性系数，有效克服网格离散与高维参数迭代成本，实现高精度参数估计。 |
|[RANS 后向台阶流仿真](./Fluid_Mechanics/rans_backstep)|使用雷诺平均 Navier–Stokes 方程 (RANS) 结合 Eddy 粘性模型 (EVM)，对后向分离腔体流场进行高雷诺数湍流模拟。|


</details>

<details open>
<summary markdown="span" ><h3>电磁学</h3> </summary>

| 示例 | 场景说明 |
| --- | --- |
|[PINN 电磁反演 ](./Electromagnetism/PINNrever)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|传统找矿依赖电磁勘探获取地下信息，但面临成本高、效率低等问题，深部隐伏矿体的电磁信号微弱，导致定位困难，人工分析电磁数据也易遗漏矿化信息。PINN 电磁反演技术以大地电磁的 Maxwell 方程组为核心数学物理模型，将电磁物理方程深度融入神经网络，通过常系数和变系数反演，解析地下介质电磁特性，捕捉微弱电磁信号变化，预测矿体位置。尽管目前基于假设数据研究，但已展现良好效果，未来应用于实际，将为电磁勘探找矿带来新突破。​​|
| [混合磁矩反演器](./Electromagnetism/HybridMagMomEstimator) | 传统磁矩估计方法常依赖简化假设与离线拟合，难以兼顾宏观椭球体效应和微观磁偶极子分布，且在低信噪比环境下精度受限。混合磁矩反演器基于 Maxwell 方程组，将椭球体模型与磁偶极子阵列模型联合建模，利用 PINN 网络端到端反演磁场观测数据，输入空间坐标、入射角度及磁场分量，输出 16 个磁偶极子与椭球体共 48 维磁矩参数，通过 Xavier 初始化和自定义磁场残差损失实现高效训练与鲁棒反演，为舰船隐身评估和目标识别提供高精度磁矩估计工具。|
| [船舶磁场建模与仿真教程](./Electromagnetism/船舶磁场建模) | 本项目基于 Maxwell 方程组，联合构建磁偶极子阵列与椭球体模型的正问题仿真框架，输入传感器坐标 (x,y,z)、船长 $L$、船宽 $W$ 和航向角 $\theta$，以及 15 个磁偶极子 + 1 个椭球体的磁矩向量，端到端计算传感器系下磁场分量 (Hx, Hy, Hz)。采用 PINN 结构结合物理约束与数据驱动损失训练，具备高精度预测能力，可用于舰船磁场异常检测与目标识别。 |
| [电磁散射模拟推演](./Electromagnetism/电磁散射模拟推演) | 本项目针对二维电磁波在未知目标区域中的传播与反射问题，构建基于 PINN 的逆散射求解器。通过亥姆霍兹方程与 PML 吸收边界建模，分别实现了正问题（已知介电常数求电场）与反问题（已知电场反演目标位置与介电常数）。项目包含两种模型：`pytorchES.py` 用于求解简单区域内电场分布，`pytorchEIS.py` 结合边界条件与可训练介电常数，识别散射体几何与参数。结果显示，该方法可有效捕捉弱电磁扰动信号，为雷达成像与深部目标识别提供可行路径。 |


</details>



<details open>
<summary markdown="span">
  <h3>地质学</h3><br/>
  在地质研究与资源勘探领域，复杂的地形信息解读和稀疏的矿产数据处理一直是行业难题。传统方法在面对冰川动态监测、深部矿体定位、元素分布分析等工作时，往往效率低下、精准度不足。随着技术发展，各类创新工具应运而生，它们借助前沿算法与技术，为地质工作者提供了更高效、更精准的分析手段，有力推动着地质行业向智能化、科学化方向迈进。<br/><br/>
  
</summary>

| 示例 | 场景说明 |
| --- | --- |
|[ 地形可视化工具 TopoZeko ](./Geology/Tpozeko)|地质研究与资源勘探常需可视化地形信息。在冰川变化监测、矿产定位、环境演变评估等工作中，亟需直观展示地形数据。TopoZeko 基于 PyTorch 和 Matplotlib，通过高程矩阵计算地表厚度，以多维度可视化方式及自定义参数，助力地质工作者高效分析地形|
|[ GeoChatBot 异常图生成器 ](./Geology/geochatbot)|地质勘探中，稀疏的矿产元素数据难以直观呈现分布特征。GeoChatBot 异常图生成器采用 IDW 和克里金插值方法，生成详细异常图展示元素分布，还具备定位和测距功能，为矿产勘探与地质研究提供便利工具。|

</details>

<details open>
<summary markdown="span" ><h3>多变种 PINNs</h3> </summary>

| 技术名称 | 简介 |适用场景|
| -------- | -------------- |----------------|
| [基础 PINN](./Variant_PINNs/multi_pinns/PINN) | Physics‑Informed Neural Networks (PINNs) 利用神经网络近似 PDE 解，将方程强形式残差与边界条件残差一并作为损失，训练过程中强制网络满足物理模型。|适合作为各种物理场（热传导、流体力学、弹性力学等）问题的基线方案与教学示例，为改进型 PINNs 提供对照。|
| [gPINN](./Variant_PINNs/multi_pinns/gPINN) | Generalized Physics‑Informed Neural Networks (gPINNs) 扩展传统 PINNs，加入自适应残差加权与多样性损失形式，增强网络对不同物理场约束的控制和多目标平衡能力。|适用于多物理场耦合、高非线性问题或边界/源项复杂的仿真，如燃烧-流动耦合、电热-结构耦合等。 |
| [VPINN](./Variant_PINNs/multi_pinns/vPINN) |Variational PINNs (vPINNs) 基于 Petrov–Galerkin 变分公式，将试探空间设为神经网络、测试空间采用 Legendre 多项式，用高斯数值积分替代传统 PINNs 的海量惩罚点，可解析处理浅层网络、并可推广至深度网络。|适用于需变分求解的 PDE，如结构固有频率分析、弹性力学和流体-热耦合系统，对积分精度要求高且强形式残差难以直接估计的问题。 |
| [XPINN](./Variant_PINNs/multi_pinns/XPINN) | Extended Physics‑Informed Neural Networks (XPINNs) 将空间–时间域划分为多个子域，在每个子域内训练独立的 PINN 模型，并通过界面一致性条件耦合子域解，有效提升非线性 PDE 的求解精度与可扩展性。|适用于大规模或复杂几何、多物理耦合问题的并行求解，如气候模拟、海洋环流及分段结构力学分析。 |
| [PPINN](./Variant_PINNs/multi_pinns/PPINN) |Parareal Physics‑Informed Neural Network (PPINN) 将长时间 PDE 积分问题分解为多个短时子问题，先用粗粒度解算器串行预测，再并行训练细粒度 PINNs 进行修正，几轮迭代即可收敛；显著降低训练开销并支持并行加速。|适用于长时间演化的时域 PDE，如大气/气候模拟、化学反应动力学和工程系统的周期性加载分析。|
| [SPINN](./Variant_PINNs/multi_pinns/SPINN) |可分离 PINN (SPINN) 架构将多维坐标拆分给独立子网络，结合前向自动微分与低秩张量近似，有效降低多维 PDE 求解的计算成本。|适用于高维热传导、量子系统模拟和 3D 结构力学等对计算效率要求极高的物理场建模。|
| [Dropout BNN](./Variant_PINNs/bnn/Dropout) |结合贝叶斯方法与 PINNs，通过在训练和推理阶段启用 Monte Carlo Dropout，对 PDE 残差进行多次随机采样，输出解的均值与不确定性估计。|适用于对预测置信区间和模型稳健性要求高的物理问题，如地下水流模拟、气候预测与材料疲劳分析。 |
| [HMC BNN](./Variant_PINNs/bnn/HMC) | 结合贝叶斯 PINNs 与 Hamiltonian Monte Carlo，通过构造哈密顿动力学对网络参数后验分布进行精确采样，捕捉多峰不确定性。|适用于需要高精度多模态不确定性评估的复杂 PDE 问题，如湍流模拟、地球物理反演及量子场动力学。 |
| [VI BNN](./Variant_PINNs/bnn/VI) | 基于变分推断的贝叶斯 PINNs，利用重参数化技巧将模型参数的后验分布（通常为高斯族）嵌入网络，通过最大化 ELBO 进行端到端训练；提供采样与对数概率计算。|适用于需要高效计算不确定性但资源受限的物理场问题，如大规模热传导、不确定性量化与工程设计优化。 |
| [PIVAE](./Variant_PINNs/pivae) | 物理信息引导的变分自编码器，生成式物理场建模与重构，适用于数据增强与无监督物理场表示学习。 |

</details>

<details open>
<summary markdown="span" ><h3>数学公式与物理方程</h3> </summary>

#### [公式链接](./Math_Physics_Formulations)
| 方程名称            | 方程形式                                                                                                                                              | 应用场景                         |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| Navier–Cauchy 方程  | $$(\lambda + \mu)\nabla(\nabla\cdot\mathbf{u}) +\mu\nabla^2\mathbf{u} +\mathbf{f} =0$$                                                | 弹性固体力学、结构变形与应力分析 |
| Navier–Stokes 方程  | $$\mathbf{u}_t + (\mathbf{u}\cdot\nabla)\mathbf{u} = -\nabla p + \nu\nabla^2\mathbf{u},\quad \nabla\cdot\mathbf{u}=0$$                         | 流体力学、湍流模拟、空气/海洋动力学 |
| Korteweg–de Vries 方程 (KdV) | $$u_t + \alpha u u_x + \beta u_{xxx} = 0$$                                                                                                     | 孤立波与浅水波、大气/内波模拟     |
| 1D 泊松/Helmholtz 方程     | $$\displaystyle \frac{d^2u}{dx^2} = f(x)$$                                                                                                       | 电势分布、热传导一维示例         |
| 2D 泊松/Helmholtz 方程     | $$\displaystyle \frac{\partial^2u}{\partial x^2} + \frac{\partial^2u}{\partial y^2} = f(x,y)$$                                                       | 平面热传导、电场/重力场模拟       |
|热传导方程   |$$\frac{\partial u}{\partial t} = \alpha \nabla^2 u$$| 稳态与非稳态的热传导、扩散问题|
| 波动方程                 | $$\displaystyle \frac{\partial^2u}{\partial t^2} = c^2\Bigl(\frac{\partial^2u}{\partial x^2} + \frac{\partial^2u}{\partial y^2}\Bigr)$$                | 声学/地震波传播、弦/膜振动        |
| Vlasov 方程  (一维)             | $$\frac{\partial f}{\partial t}+\lambda_1 \cdot v \cdot \frac{\partial f}{\partial x}+\lambda_2 \cdot E \cdot \frac{\partial f}{\partial v}=0$$ | 无碰撞等离子体物理、粒子分布演化     |
| Maxwell 方程             | $$\nabla \times (\nabla \times \mathbf{E}) - \mu \epsilon \omega^2 \mathbf{E} = 0$$                                                                | 电磁场模拟、电磁波传播、光学设计   |
| 亥姆霍兹方程             | $$\nabla^2 u + k^2 u = 0$$                                                                                                                        | 声学、光学、电磁散射问题         |
</details>
