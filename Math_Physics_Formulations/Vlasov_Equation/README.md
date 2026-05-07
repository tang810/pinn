# Vlasov_Equation

本项目基于 PINN 求解 Vlasov 方程，已按统一工程规范整理。

## 标准目录

- `src/`：核心源码（模型、数据处理、训练/推理、绘图）
- `data/`：输入数据目录（`E.npy`、`f.npy` 或模拟数据）
- `model/`：模型参数目录（`.pth`）
- `results/`：结果输出目录
- `main.py`：统一入口
- `README.md`：项目说明
- `file.json`：结构化描述

## 运行

规范快速命令：

```bash
python main.py --mode quick --data simul
```

完整训练：

```bash
python main.py --mode train --data simul
```

真实数据模式：

```bash
python main.py --mode quick --data real --data_path data
```

## 参数

- `--mode`: `quick` / `train`
- `--data`: `simul` / `real`
- `--data_path`: real 模式数据目录

## 输出

- `results/quick_velocity_distribution.png`
- `results/quick_metrics.txt`
- `model/PINN_final.pth`