# Smartphone Drop Impact PINN
## 手机跌落冲击应力 PINN 案例

## 1. 项目概述
本案例展示了一个基于 Physics-Informed Neural Networks（PINNs）的手机跌落冲击应力场模拟示例，用于预测短时冲击载荷下的应力传播与集中行为。

## 2. 物理问题背景
手机跌落会在机身内部激发应力波，应力集中区域往往对应潜在的结构失效风险。本案例用于分析冲击后应力随时间的演化。

## 3. 控制方程与变量定义
### 3.1 控制方程
二维标量应力波方程：

d2S/dt2 = c^2 * (d2S/dx2 + d2S/dy2) + q(x,y,t)

## 4. PINN 建模思路
PINN 学习等效应力场 S(x,y,t)，通过自动微分约束 PDE、初始条件与边界条件。

## 5. 项目结构
```
Smartphone_DropImpact_PINN/
├── data/
├── model/
├── results/
├── src/
├── Smartphone_DropImpact_PINN_main.py
└── readme.md
```

## 6. 运行方式
### 6.1 训练模式
python Smartphone_DropImpact_PINN_main.py --mode train

### 6.2 快速推理模式
python Smartphone_DropImpact_PINN_main.py --mode quick

## 7. 结果展示
- 最大等效应力-时间曲线
- 关键时刻应力云图
- 多时间点应力场快照

## 8. 模型简化与限制
- 二维简化模型
- 标量应力近似
- 均匀材料参数
- 等效冲击源项

## 9. 工业价值与扩展
- 跌落风险的快速定性评估
- 可扩展至弹性动力学和多材料模型
