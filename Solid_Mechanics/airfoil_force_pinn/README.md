<!--
text: "翼型结构力约束 PINN 求解"
area: "面向三维线弹性位移场重构的物理信息神经网络案例。"
tags: ["Solid Mechanics", "PINN", "Linear Elasticity", "Airfoil"]
search: ["翼型结构", "弹性力学", "PINN"]
-->

# airfoil_force_pinn：翼型结构力约束的物理信息神经网络求解

## 项目概述
本案例基于物理信息神经网络（PINN）求解三维线弹性问题，以翼型结构区域内的位移场 `u,v,w` 为预测目标。

模型将坐标 `(x, y, z)` 输入神经网络，通过自动微分构造弹性控制方程残差，并与边界条件损失联合优化，实现无网格求解。

## 物理模型
采用三维各向同性线弹性静力平衡方程（含体力项）：

- 位移场：`u(x,y,z), v(x,y,z), w(x,y,z)`
- 拉梅参数由弹性模量 `E` 与泊松比 `nu` 计算
- 方程残差在 `src/pinn_solver.py` 的 `neural_net_equations` 中定义

训练目标为：

- `L_eq`：控制方程残差损失
- `L_bc`：边界位移约束损失
- `L_total = lambda_eq * L_eq + lambda_bc * L_bc`

## 快速开始

### 1. 环境依赖
- Python >= 3.8
- PyTorch
- NumPy
- SciPy
- Matplotlib

### 2. 快速验证（推荐）
```bash
python main.py --mode quick --data simul
```

### 3. 完整训练
```bash
python main.py --mode train --data simul
```

## 运行参数
主入口：`main.py`

- `--mode`：`quick` / `train`
- `--data`：`simul` / `real`
- `--data_path`：数据路径，默认 `data/data.mat`
- `--model_dir`：模型输出目录，默认 `model`
- `--result_dir`：结果目录，默认 `results`
- `--train_epochs`：训练轮数（train 模式）
- `--quick_epochs`：快速模式短训轮数（无权重时启用）
- `--lr`：学习率

## 输出说明
- 模型权重：`model/airfoil_force_pinn_best.pt`
- 快速模式图：`results/result_quick.png`
- 训练模式图：`results/result_train.png`
- 评估误差与中间结果：`results/*.mat`

## 工程结构
```text
airfoil_force_pinn/
├── src/                  # 核心源码（网络、求解器、runner）
├── data/                 # 输入数据（默认 data.mat）
├── model/                # 模型参数输出
├── results/              # 图像与评估结果输出
├── main.py               # 主入口（train/quick）
├── README.md             # 案例文档
└── file.json             # 结构描述文件
```

## 代码模块说明
- `src/config.py`：参数定义
- `src/runner.py`：训练/快速模式调度
- `src/net.py`：全连接网络结构
- `src/pinn_solver.py`：PINN 方程、损失、训练与评估
- `src/SM_data.py`：数据加载
- `src/run_legacy.py`：历史单文件脚本备份

## 备注
- 若 `model/airfoil_force_pinn_best.pt` 不存在，`quick` 模式会先进行短轮次训练，再输出快速结果图。
- 当前 `--data simul` 的默认路径为 `data/data.mat`，可通过 `--data_path` 覆盖。
