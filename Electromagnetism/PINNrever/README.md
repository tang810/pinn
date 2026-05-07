# PINNrever

PINNrever 是二维参数反演示例项目，已按统一规范整理为标准工程结构。

## 目录

- `src/`：核心逻辑（模型、损失、训练、推理、配置）
- `data/`：输入数据（`real_points.npy` 等）
- `model/`：模型参数输出（`.pt`）
- `results/`：结果输出（图像、指标）
- `main.py`：统一入口
- `file.json`：结构化描述文件

## 运行

规范快速命令：

```bash
python main.py --mode quick --data simul
```

完整训练：

```bash
python main.py --mode train --data simul
```

真实数据推理（可选）：

```bash
python main.py --mode quick --data real --data_path data/real_points.npy
```

## 参数

- `--mode`: `quick` / `train`
- `--data`: `simul` / `real`
- `--task`: `scalar` / `recover`