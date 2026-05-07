import os
import torch

# 有些环境（比如 Windows + MKL）会有 OpenMP 重复加载的 warning，这里简单忽略
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

def get_device() -> torch.device:
    """自动选择 CPU / GPU。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 质量和惯性矩（近似取自论文的 F-16 参数表）
MASS = 9298.0   # kg
IX = 12874.0
IY = 75673.0
IZ = 85552.0
IXZ = 1331.0

# 几何参数
S_REF = 27.87    # m^2   机翼面积
B_REF = 9.14     # m     翼展
C_REF = 3.45     # m     平均气动弦长

# 环境参数
RHO = 1.225      # kg/m^3 空气密度（海平面）
G = 9.81         # m/s^2  重力加速度
T_CONST = 20000  # N 恒定推力（简化处理）
