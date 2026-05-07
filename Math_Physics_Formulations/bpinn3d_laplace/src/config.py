# -*- coding: utf-8 -*-
"""
配置与超参数（强制 CPU + float64；B-PINN/HMC 超参数；数据规模等）
"""
import torch
import hamiltorch

# -----------------------------
# Device & dtype（强制 CPU + float64，更稳的二阶导）
# -----------------------------
# if torch.backends.mps.is_available():
#     print("[Info] MPS 可用，但将被忽略（FORCING CPU）。")

device = torch.device('cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE  = torch.float64
torch.set_default_dtype(DTYPE)
print(f"Device: {device}, dtype: {torch.get_default_dtype()}  (FORCING CPU)")

# 传给 util/hamiltorch 的设备字符串（多数版本兼容 'cpu' 最好）
DEVICE_STR_FOR_UTIL = device.type  # 'cpu' 或 'cuda'

# -----------------------------
# 超参数（先验/似然 & HMC & 网络）
# -----------------------------
hamiltorch.set_random_seed(123)

# 先验/似然
prior_std  = 1.0
like_std_u = 0.10  # u 数据噪声
like_std_f = 0.10  # f 数据噪声

# 精度（precision）
tau_priors = 1.0 / (prior_std ** 2)
# 这里仍传 2 维以兼容 model_loss 的索引写法
tau_likes  = torch.tensor([1.0/(like_std_u**2), 1.0/(like_std_f**2)], dtype=DTYPE)

# HMC（可按接受率调 step_size/L）
step_size   = 0.003
L           = 80
burn        = 200
num_samples = 400

# 网络（MLP 近似 u(x,y,z)）
layer_sizes = [3, 32, 32, 32, 1]  # in, h1, h2, h3, out
activation  = torch.tanh

# 与 util API 对齐的标记
pde     = True
pinns   = False
epochs  = 2000  # 若 util 内有 MAP 预优化阶段会用到

# 区间与数据规模
lb, ub    = -0.7, 0.7
N_tr_u    = 64      # 带噪 u 观测点数
N_tr_f    = 512     # 带噪 f 观测点数（或把 PDE 残差当似然）
N_val_1d  = 20      # 验证网格分辨率（每轴）→ 总点数 N_val_1d^3

# 真值（仅用于合成数据/对照）
alpha_true      = 1.0
exact_single    = [alpha_true]
n_params_single = 1
