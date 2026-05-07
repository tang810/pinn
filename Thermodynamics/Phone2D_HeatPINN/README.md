# Smartphone 2D Thermal PINN
## 手机二维散热 PINN 案例

## 1. 项目概述
本案例展示了一个基于 Physics-Informed Neural Networks（PINNs）的二维手机散热模拟示例，用于预测手机在工作过程中，由芯片（SoC）发热引起的温度场随时间的演化。

该案例不依赖真实测量数据，而是直接将热传导物理方程、初始条件和边界条件嵌入神经网络训练过程，实现一个无网格、物理一致的散热预测模型，适合作为手机热设计的早期评估与展示型案例。

## 2. 物理问题背景
手机等电子设备中，SoC 在持续工作时会产生显著热量，热量通过机身内部材料传导，并通过外表面对流散出。

本案例将手机简化为一个二维矩形区域：
- 中心区域表示 SoC 热源
- 其余区域表示机身结构
- 外边界通过对流方式与环境换热

## 3. 控制方程与变量定义
### 3.1 控制方程
二维瞬态热传导方程：

rho * cp * dT/dt = k * (d2T/dx2 + d2T/dy2) + q(x,y,t)

### 3.2 初始与边界条件
- 初始条件：T(x,y,0) = T_inf  
- 边界条件：对流换热边界

## 4. PINN 建模思路
PINN 将温度场 T(x,y,t) 表示为神经网络输出，通过自动微分构建 PDE、初始条件和边界条件残差，并在整个时空域内进行联合优化。

## 5. 项目结构
```
Phone2D_HeatPINN/
├── data/
├── model/
├── results/
├── src/
├── Smartphone_Thermal_PINN_main.py
└── readme.md
```

## 6. 运行方式
### 6.1 训练模式
python Smartphone_Thermal_PINN_main.py --mode train --data simul

### 6.2 快速推理模式
python Smartphone_Thermal_PINN_main.py --mode quick --data simul

## 7. 结果展示
- 多时间点二维温度场云图
- SoC 中心温度随时间变化曲线

## 8. 模型简化与限制
- 二维简化模型
- 忽略材料层间界面热阻
- 材料参数视为常数
- 均匀矩形 SoC 热源

## 9. 工业价值与扩展
- 适用于散热设计早期评估
- 可扩展至 3D、多热源、实验数据融合
