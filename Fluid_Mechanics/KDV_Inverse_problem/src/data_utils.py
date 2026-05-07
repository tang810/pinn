import numpy as np
import os

def load_user_data(file_path):
    """加载用户提供的数据文件 (.csv 或 .npy)"""
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return None
    try:
        if file_path.endswith(".csv"):
            data = np.loadtxt(file_path, delimiter=",")
        elif file_path.endswith(".npy"):
            data = np.load(file_path)
        else:
            print("❌ 不支持的文件格式，仅支持 .csv 或 .npy")
            return None
        return data
    except Exception as e:
        print(f"❌ 加载文件失败: {e}")
        return None

def validate_user_data(user_data):
    """验证用户数据格式是否符合训练要求"""
    if user_data is None:
        return False
    if isinstance(user_data, np.ndarray):
        if user_data.ndim == 2 and user_data.shape[1] >= 3:
            return True
    return False

def generate_virtual_data(n_samples=100):
    """生成虚拟训练数据"""
    x = np.random.uniform(-20, 20, n_samples)
    t = np.random.uniform(0, 4, n_samples)
    
    a = 1.0
    c = 1.0
    lam = 2.0
    eta = a * (1 / np.cosh((x - c * t) / lam))**2

    data = np.stack([x, t, eta], axis=1)
    return data
