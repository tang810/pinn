<!-- 
text: "PINN 六自由度战斗机气动系数识别",
area: "本案例通过物理信息神经网络（PINN）学习“类 F-16”飞机在六自由度（6DOF）刚体动力学下的气动系数映射关系。网络以攻角/侧滑角、无量纲角速度、舵面偏转与速度等状态为输入，输出六个气动系数，并通过将预测系数代入 6DOF 动力学后形成的物理残差进行约束，从而在少量（含噪）轨迹数据上实现可解释的气动建模。",
tags: ["PINN", "6DOF", "气动系数", "系统辨识", "飞行力学"],
search: ["F-16", "aerodynamic coefficients", "physics-informed neural network", "6DOF rigid body", "system identification"]
-->

# PINN 六自由度战斗机气动系数识别（pinn_6dof_f16）

## 1. 问题背景与物理方程

在飞行力学建模中，气动系数（例如 $C_X,C_Y,C_Z,C_l,C_m,C_n$）决定了气动力/力矩的大小与方向，是六自由度（6DOF）刚体动力学仿真与控制设计的关键。本案例构造了一个“类 F-16”的光滑非线性气动真值模型用于生成训练数据，并使用 PINN 在含噪轨迹数据上学习气动系数映射。

### 1.1 6DOF 刚体动力学（概念形式）
本案例的核心约束来自 6DOF 动力学（机体系平动 + 转动 + 欧拉角运动学 + 位置运动学）。概念上可写为：
- 平动：$\dot{\mathbf{v}} = f_v(\mathbf{x},\mathbf{u},\mathbf{C})$
- 转动：$\dot{\boldsymbol\omega} = f_\omega(\mathbf{x},\mathbf{u},\mathbf{C})$
- 姿态：$\dot{\boldsymbol\eta} = f_\eta(\boldsymbol\omega,\boldsymbol\eta)$
- 位置：$\dot{\mathbf{p}} = f_p(\mathbf{v},\boldsymbol\eta)$

其中 $\mathbf{C} = (C_X,C_Y,C_Z,C_l,C_m,C_n)$ 由神经网络预测；$\mathbf{u}$ 为舵面偏转（升降舵/副翼/方向舵）输入；$\mathbf{x}$ 为系统状态（速度分量、角速度、姿态角、位置等）。

> 说明：代码中采用 RK4 单步积分作为离散时间推进，并将“观测到的状态导数”与“由网络气动系数产生的动力学导数”之间的差作为物理残差。

---

## 2. 建模方法说明（PINN）

### 2.1 网络输入/输出
- **输入特征（示例）**：$\alpha,\beta,\hat p,\hat q,\hat r,\delta_e,\delta_a,\delta_r, V$
- **输出**：$C_X,C_Y,C_Z,C_l,C_m,C_n$

其中 $\hat p,\hat q,\hat r$ 为按速度与气动参考尺寸缩放后的无量纲角速度。

### 2.2 损失函数（数据项 + 物理项）
训练时综合两类约束：
1. **数据一致性**：网络输出系数与真值系数（或由真值模型生成的标签）之间的误差；
2. **物理残差**：将网络系数代入 6DOF 动力学后得到的导数与数据轨迹导数之间的误差。

---

## 3. 主函数入口与运行方式（--quick / --train）

主入口脚本：`pinn_6dof_f16_main.py`

### 3.1 默认行为（零参数运行）
为便于本地调试，**直接运行不带参数**时默认采用：
- `--quick` + `--simul`
- 使用原始脚本默认参数（例如 `t_end=20.0, dt=0.02, noise=0.02, epochs=5000, lr=1e-3, seed=0`）
- 若 `model/best_model.pt` 不存在：会按默认 epoch 训练并保存；若存在：加载模型并评估/出图

直接运行：
```bash
python pinn_6dof_f16_main.py
```

### 3.2 显式训练（推荐用于正式训练）
```bash
python pinn_6dof_f16_main.py --train --simul --epochs 5000 --lr 1e-3
```

### 3.3 从文件加载数据
```bash
python pinn_6dof_f16_main.py --train --load --data_path data/flight_dataset.npz
```

### 3.4 仅评估/可视化（加载权重）
```bash
python pinn_6dof_f16_main.py --quick --load --data_path data/flight_dataset.npz --ckpt model/best_model.pt
```

### 3.5 常用参数说明
- `--train / --quick`：训练模式 / 快速模式（评估与可视化；无权重时可按默认训练）
- `--simul / --load`：仿真生成数据 / 从文件加载数据
- `--data_path`：`--load` 时的数据路径（`.npz`）
- `--ckpt`：权重路径（`.pt`）
- `--epochs`：训练轮数（默认与原始脚本一致）
- `--lr`：学习率（默认与原始脚本一致）
- `--noise`：仿真数据噪声幅度（默认与原始脚本一致）
- `--t_end, --dt`：仿真时长与时间步长（默认与原始脚本一致）
- `--seed`：随机种子（默认与原始脚本一致）
- `--no_resume`：关闭断点续训（默认开启 resume）
- `--verbose_every`：训练日志打印间隔（默认与原始脚本一致）

---

## 4. 输入输出数据说明

### 4.1 输入（data/）
- `data/flight_dataset.npz`（示例）：包含时间序列、状态、控制量、真值气动系数等（由 `--simul` 生成或自行提供）

### 4.2 输出（model/ 与 results/）
- `model/best_model.pt`：训练得到的最优网络权重
- `results/loss_hist.npy`：训练损失曲线数据
- `results/loss_curve.png`：训练损失可视化
- `results/coeff_pred.npy`：网络预测气动系数
- `results/coeff_true.npy`：真值气动系数（用于对比）
- `results/coeff_compare.png`：系数对比可视化图

---

## 5. 可视化与结果展示

运行结束后，`results/` 目录下会自动生成：
- 损失曲线图（loss curve）
- 气动系数真值 vs 预测的对比图（coeff compare）

可视化逻辑集中在 `src/plotting.py` 中，便于二次扩展（例如添加 RMSE 统计图、分段误差、频域分析等）。

---

## 代码文件说明

### 主程序
- `pinn_6dof_f16_main.py`：案例主入口（命令行参数解析 + 运行模式编排）。负责选择数据来源（`--simul/--load`）、选择运行模式（默认 `quick+simul`，也可显式 `--train/--quick`）、调用训练/评估/可视化流程，并将输出统一写入 `model/` 与 `results/`。

### src/ 模块职责划分
- `src/config.py`：全局配置与默认超参数（设备选择、dtype、默认 `t_end/dt/noise/seed/epochs/lr` 等）。
- `src/aero_true.py`：真值（仿真）气动系数模型，用于生成带噪观测数据与对照评估。
- `src/controls.py`：控制输入定义（多正弦舵面输入等），用于仿真飞行动力学轨迹。
- `src/dynamics.py`：6DOF 动力学方程与数值积分（RK4），将气动系数与状态/控制量耦合得到状态演化。
- `src/data_utils.py`：数据生成/保存/加载（`.npz`），以及将状态/控制量整理成网络输入特征（含归一化与无量纲化逻辑，若启用）。
- `src/model.py`：PINN 中的气动网络 `AeroNet`（输入特征 → 输出 6 个气动系数）。
- `src/losses.py`：PINN 物理残差损失的构造（将网络输出代入 6DOF 动力学形成残差，并与数据项组合）。
- `src/train.py`：训练流程（优化器/学习率、loss 记录、checkpoint 保存与 resume 逻辑）。
- `src/eval.py`：评估流程（加载模型后输出预测系数，与真值系数对比并计算 RMSE 等指标）。
- `src/plotting.py`：可视化（loss 曲线、气动系数真值 vs 预测对比图等），所有图输出到 `results/`。
- `src/io_utils.py`：模型与结果的 I/O 工具（保存/加载 `.pt`，路径与命名约定集中管理）。

> 设计原则：**主脚本只做“编排”，功能逻辑全部沉到 `src/`**，以满足《案例内容规范：函数组织与主程序逻辑》的要求。


## 6. 参考资料（如有）
- Stevens, B. L., & Lewis, F. L. *Aircraft Control and Simulation*（6DOF 形式与缩放常见来源）
- Physics-Informed Neural Networks（PINN）相关论文与综述

---

## 项目结构说明

│   ├── pinn_6dof_f16                                   # 🧩 pinn_6dof_f16：在pinn_6dof_f16项目中，作为根目录，用于组织该案例的全部数据、模型、源码与可视化结果内容。
│   │   ├── data                                        # 🗂️ pinn_6dof_f16_data：在pinn_6dof_f16项目中，作为数据准备环节，用于存储仿真生成或外部提供的训练/验证数据文件。
│   │   ├── model                                       # 🗂️ pinn_6dof_f16_model：在pinn_6dof_f16项目中，作为模型存储环节，用于保存训练完成的神经网络权重文件（.pt）。
│   │   ├── results                                     # 🗂️ pinn_6dof_f16_results：在pinn_6dof_f16项目中，作为结果输出与可视化环节，用于保存预测数据、损失曲线与对比图等产物。
│   │   ├── src                                         # 🧩 pinn_6dof_f16_src：在pinn_6dof_f16项目中，作为功能模块集合，用于承载数据、模型、损失、训练、评估与可视化等可复用代码。
│   │   │   ├── config.py                               # 📄 pinn_6dof_f16_config.py：在pinn_6dof_f16项目中，作为配置管理环节，用于统一设备选择与全局常量设置。
│   │   │   ├── aero_true.py                            # 📄 pinn_6dof_f16_aero_true.py：在pinn_6dof_f16项目中，作为真值气动模型环节，用于生成用于训练的数据标签气动系数。
│   │   │   ├── controls.py                             # 📄 pinn_6dof_f16_controls.py：在pinn_6dof_f16项目中，作为控制输入构造环节，用于生成多正弦舵面偏转信号以激发系统动态。
│   │   │   ├── dynamics.py                             # 📄 pinn_6dof_f16_dynamics.py：在pinn_6dof_f16项目中，作为动力学计算环节，用于实现6DOF方程与RK4积分推进。
│   │   │   ├── data_utils.py                           # 📄 pinn_6dof_f16_data_utils.py：在pinn_6dof_f16项目中，作为数据处理环节，用于仿真生成/加载数据并构造网络训练输入特征。
│   │   │   ├── model.py                                # 📄 pinn_6dof_f16_model.py：在pinn_6dof_f16项目中，作为网络构建环节，用于定义气动系数预测网络结构与前向传播逻辑。
│   │   │   ├── losses.py                               # 📄 pinn_6dof_f16_losses.py：在pinn_6dof_f16项目中，作为损失函数环节，用于实现数据项与物理残差项的PINN训练目标。
│   │   │   ├── train.py                                # 📄 pinn_6dof_f16_train.py：在pinn_6dof_f16项目中，作为模型训练环节，用于配置优化器与训练主循环并支持断点续训。
│   │   │   ├── eval.py                                 # 📄 pinn_6dof_f16_eval.py：在pinn_6dof_f16项目中，作为评估对比环节，用于计算预测与真值的误差指标并生成对比数据。
│   │   │   ├── plotting.py                             # 📄 pinn_6dof_f16_plotting.py：在pinn_6dof_f16项目中，作为可视化展示环节，用于绘制损失曲线与气动系数对比图。
│   │   │   ├── io_utils.py                             # 📄 pinn_6dof_f16_io_utils.py：在pinn_6dof_f16项目中，作为模型IO环节，用于保存/加载权重文件与相关元数据。
│   │   ├── pinn_6dof_f16_main.py                       # 🧠 pinn_6dof_f16_pinn_6dof_f16_main.py：在pinn_6dof_f16项目中，作为主控脚本环节，用于组织训练、加载、预测与保存模型结果。
│   │   ├── README.md                                   # 📄 pinn_6dof_f16_README.md：在pinn_6dof_f16项目中，作为说明文档环节，用于介绍问题背景、建模方法、数据说明与运行方式。
