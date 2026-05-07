# 二维弹性力学基函数预计算与在线求解

本项目已按统一规范整理为标准工程结构，支持快速验证与完整训练。

## 标准目录

- `src/`：核心源码（数据、模型、训练/推理、绘图）
- `data/`：输入数据目录（含 real 模式示例数据）
- `model/`：模型参数目录
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
python main.py --mode quick --data real --data_path data/real_data.npz
```

## 参数

- `--mode`: `quick` / `train`
- `--data`: `simul` / `real`
- `--data_path`: real 模式数据文件路径

## 输出

- `results/quick_result.png`
- `results/quick_metrics.txt`
- `model/elastic_basis_model.pt`