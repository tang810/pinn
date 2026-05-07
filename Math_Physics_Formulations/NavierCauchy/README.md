# NavierCauchy

本项目使用 PINN 求解二维 Navier-Cauchy 方程，已按统一规范整理目录与入口。

## 标准目录

- `src/`：核心源码（网络、物理、损失、数据与可视化）
- `data/`：输入数据目录（PDE/边界采样点）
- `model/`：模型参数目录
- `results/`：结果输出目录
- `main.py`：统一主入口
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

真实数据目录模式（可选）：

```bash
python main.py --mode quick --data real --data_path data
```

## 参数

- `--mode`: `quick` / `train`
- `--data`: `simul` / `real`
- `--data_path`: 数据目录路径

## 输出

- `results/quick_loss.png`
- `results/loss_curve_final.png`
- `model/pinn_final.pt`