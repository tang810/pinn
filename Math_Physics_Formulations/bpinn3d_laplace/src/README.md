# B-PINN 3D 非线性拉普拉斯逆问题（功能化拆分版）

将原始单文件脚本按功能拆分为多个独立模块：

```
bpinn3d_split/
├── config.py        # 设备/精度、超参数与数据规模
├── truth.py         # 真解、拉普拉斯算子、右端 f 与采样点 rand_points
├── data.py          # 生成训练集与验证网格
├── model.py         # Net3D 与构建网络
├── loss.py          # B-PINN 对数似然（数据项 + PDE 项）
├── plotting.py      # 后验切片可视化（z=mid）
├── run_hmc.py       # 主入口（采样、预测、统计与作图）
└── README.md
```

> 依赖：`torch`, `hamiltorch`, `matplotlib`，以及仓库内的 `util`（需可导入）。

## 使用方式

1. 确保你的 `util` 模块与本项目在同一层目录或已加入 `PYTHONPATH`：

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/your/repo
```

2. 运行主脚本：

```bash
cd bpinn3d_split
python run_hmc.py
```

运行后会：
- 在控制台打印验证对数似然均值、参数 α 的后验均值/标准差/置信区间；
- 在 `result/` 目录下保存 4 张 z=中间切片的后验均值/标准差图（`u_mean_zmid.png` 等）。

## 与原脚本的一致性

- 强制 `CPU + float64`，保持二阶导稳定；
- HMC 超参数与网络结构保持一致，可在 `config.py` 中统一修改；
- `loss.model_loss` 的签名与行为与原逻辑一致，兼容 `util.sample_model_bpinns/predict_model_bpinns`。

## 常见问题

- **导入报错：找不到 `util`**  
  请确认 `util.py` 或 `util` 包在 Python 搜索路径内。可用 `sys.path` 打印检查。

- **接受率偏低/偏高**  
  调整 `config.py` 中的 `step_size` 与 `L`，或缩放 `like_std_*`／`prior_std`。

- **显存/内存压力**  
  适当降低 `N_tr_f` 与 `N_val_1d`，或减小网络宽度。
