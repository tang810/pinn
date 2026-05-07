# HybridMagMomEstimator

本项目用于混合磁矩估计（Hybrid Magnetic Moment Estimation），已按统一规范整理目录与入口。

## 目录结构

- `src/`：核心逻辑（配置、数据加载、模型、物理约束、训练、评估）
- `data/`：输入数据与历史推理数据
- `model/`：模型参数输出目录
- `results/`：评估图像与指标输出目录
- `main.py`：统一入口
- `file.json`：机器可读结构描述

## 运行方式

快速模式（规范必过命令）：

```bash
python main.py --mode quick --data simul
```

完整训练：

```bash
python main.py --mode train --data real
```

参数说明：
- `--mode`：`quick` / `train`
- `--data`：`simul` / `real`
- `--data real` 时使用 `--mat_path` 指定真实 `.mat` 数据（不指定则使用默认文件）

## 输出

- `results/img.png`：磁矩对比图
- `results/metrics.txt`：误差指标
- `model/final_model.pt`：训练后模型（train 模式生成）