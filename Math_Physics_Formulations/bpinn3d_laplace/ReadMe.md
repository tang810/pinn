
# ReadMe

本仓库提供一个基于 **Hamiltonian Monte Carlo (HMC)** 的 3D B-PINN 训练入口 **`main.py`**。  
支持两种数据来源：
- **synthetic**：在线合成数据进行训练与验证；
- **file**：从 `data/` 目录读取训练数据。

此外，程序会将：
- 训练/评估结果保存到与 `main.py` **同级**目录下的 `result/`（或自定义名称）；
- 模型后验样本与（可行时的）后验均值参数保存到与 `main.py` **同级**目录下的 `model/`。

---

## 目录结构

假设项目根目录如下（`main.py` 与 `src/` 同级）：

```
project_root/
├─ main.py
├─ src/
│  ├─ config.py
│  ├─ data.py
│  ├─ loss.py
│  ├─ model.py
│  ├─ plotting.py
│  ├─ truth.py
│  └─ hamiltorch/
│     ├─ __init__.py
│     ├─ samplers.py
│     └─ util.py        # 本地 hamiltorch 工具（已通过相对路径强制加载）
├─ data/                # file 模式使用（可选）
│  ├─ train.npz         # 或四个 CSV：x_u.csv, y_u.csv, x_f.csv, y_f.csv
├─ result/              # 运行后自动生成（或自定义目录名）
└─ model/               # 运行后自动生成（保存 .pth）
```

> **注意**  
> - 程序会自动将 `src/` 加入 `sys.path` 并强制加载 `src/hamiltorch/util.py`，因此无需设置 `PYTHONPATH`。  
> - 所有输出路径（`result/` 与 `model/`）均相对于 `main.py` 所在目录，保证从任意工作目录调用都能正确落盘。

---

## 运行环境

- Python ≥ 3.9
- 依赖库：
  - `numpy`
  - `torch`（默认 **CPU + float64**，见 `src/config.py`）
  - `matplotlib`（仅用于 `synthetic` 模式下绘图）
- **无需安装**外部 `hamiltorch`：项目内置 `src/hamiltorch/`。

> 若需切换到 GPU 或 float32，请在 `src/config.py` 中修改相关设置。

---

## 查看命令行参数

```bash
python main.py -h
```

会显示主要参数（部分默认值来自 `src/config.py`）：

- `--mode {synthetic,file}`：数据来源模式（默认 `synthetic`）
- `--data-dir PATH`：当 `--mode file` 时数据目录（默认 `data`）
- `--outdir NAME`：结果输出目录名（**相对 main.py**；默认 `result`）
- HMC 超参数：
  - `--num-samples INT`
  - `--burn INT`
  - `--L INT`（Leapfrog 步数）
  - `--step-size FLOAT`
- 数据规模与网格：
  - `--N-tr-u INT`、`--N-tr-f INT`、`--N-val-1d INT`
  - `--lb FLOAT`、`--ub FLOAT`
- 其他：
  - `--no-plots`：`synthetic` 模式下不绘图（仅保存数值结果）

---

## 快速冒烟测试（推荐先跑）

一次超快的功能验证（样本与网格都很小）：

```bash
python main.py   --mode synthetic   --num-samples 10 --burn 5 --L 10 --step-size 0.003   --N-tr-u 16 --N-tr-f 64 --N-val-1d 8   --outdir result_debug --no-plots
```

若成功，你将看到终端打印验证集 log probability 与 α 的统计，并在：
- `model/posterior_samples.pth`（后验样本与元信息）
- （可行时）`model/posterior_mean_state_dict.pth`
- `result_debug/`（无图；若去掉 `--no-plots` 则会生成 4 张切片图）

---

## 标准运行：合成数据

```bash
python main.py   --mode synthetic   --num-samples 400 --burn 200 --L 80 --step-size 0.003   --N-tr-u 64 --N-tr-f 512 --N-val-1d 20   --lb -0.7 --ub 0.7   --outdir result
```

输出：
- `result/`：保存 4 张切片图（`u_mean_zmid.png` 等）
- `model/`：
  - `posterior_samples.pth`（**总会**保存；包含 `samples_flat/shapes/group_lens/n_params_single/layer_sizes` 等）
  - `posterior_mean_state_dict.pth`（**仅当**本次运行使用“手写 functional”映射且维度匹配时）

---

## 使用外部数据训练（file 模式）

在 `data/` 下准备数据，支持两种格式（任选一种）：

**方式 A：NPZ**
- 文件：`data/train.npz`
- 必含键：`x_u, y_u, x_f, y_f`
- 形状要求：
  - `x_*`：`N×3`
  - `y_*`：`N×1`（或 `N` 将自动 reshape）

**方式 B：四个 CSV**
- 文件：`data/x_u.csv, data/y_u.csv, data/x_f.csv, data/y_f.csv`
- 逗号分隔，形状同上

运行命令：
```bash
python main.py --mode file --data-dir data --outdir result_file
```

输出：
- `result_file/`：保存 `u_mean.npy, u_std.npy, f_mean.npy, f_std.npy`
- `model/`：保存 `.pth` 文件（同上）

> `file` 模式默认验证集 = 训练集。如需单独验证集，可拓展 `main.py` 增加 `--val-dir` 并在代码内读取 `val.npz` 或 CSV。

---

## 结果与模型文件说明

- **`result/`（或自定义 `--outdir`）**
  - `synthetic`：保存 4 张 `z=mid` 切片图；
  - `file`：保存 `u/f` 的后验均值与标准差 `*.npy`。
- **`model/`**
  - `posterior_samples.pth`  
    - `samples_flat`：采样得到的扁平参数向量（大小 `S × P`）  
    - `shapes/group_lens/n_params_single`：用于将扁平向量还原为各层权重  
    - `layer_sizes/kinds`：网络结构与 functional 来源（`util` 或 `manual`）
  - `posterior_mean_state_dict.pth`（可选）  
    - 当 functional 来源为 **`manual`** 且参数模板与 `Net3D(l1..l4)` 对齐时保存。  
    - 可直接用 `torch.load(...); net.load_state_dict(...)` 进行复现或下游推断。

---

## 常见问题（FAQ）

1. **`FileNotFoundError: src/hamiltorch/util.py 不存在`**  
   确认 `src/hamiltorch/util.py` 在仓库中，且从项目根目录运行 `python main.py`。

2. **Integrators 相关错误**  
   如果本地 `hamiltorch` 不支持 `Integrator.IMPLICIT`，程序会自动回退到默认 integrator，无需手动修改。

3. **数据形状不匹配**  
   `x_*` 必须是 `N×3`，`y_*` 必须是 `N×1`（或 `N` 会自动 reshape）。请检查 `data/` 下的文件。

4. **CPU / dtype 太慢**  
   代码默认 **CPU + float64** 以提升二阶导稳定性。如需提速，可在 `src/config.py` 改为 `float32` 或减少 `--num-samples / --L / --N-*-*`。

---

## 复现与下游推断（利用 .pth）

利用 `model/posterior_samples.pth` 可以**无须重训**地复现后验预测流程：  
- 读取 `samples_flat` 和还原元信息（`shapes/group_lens/n_params_single`），  
- 通过 `main.py` 中的 `_unflatten_vector` 将某一条样本向量还原成网络参数，  
- 再用 `model_loss` 或网络前向函数进行评估/预测。  
如果存在 `posterior_mean_state_dict.pth`，也可直接 `load_state_dict` 进行推断。

---

## 许可证

请根据你项目的实际情况填写许可证信息（例如 MIT、Apache-2.0 等）。
