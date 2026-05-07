# cavity_flow_PIKAN

本项目用于方腔流（Lid-driven Cavity）PIKAN 求解，已按统一工程规范整理。

## 标准目录

- `src/`：核心代码（数据加载、网络、求解器、工具）
- `data/`：输入数据（`.mat`）
- `model/`：模型参数（`.pth`）
- `results/`：结果输出（图像、mat、日志）
- `main.py`：统一入口
- `README.md`：说明文档
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

真实数据模式（可选）：

```bash
python main.py --mode quick --data real --data_path data/cavity_Re2000_256.mat
```

## 参数

- `--mode`: `quick` / `train`
- `--data`: `simul` / `real`
- `--data_path`: `real` 模式指定评估数据文件
- `--model_type`: `standard` / `ev_enhanced`

## 输出

- `results/result_plot.png`：预测结果图
- `results/result_uvdata/`：阶段性场数据
- `model/*.pth`：模型参数