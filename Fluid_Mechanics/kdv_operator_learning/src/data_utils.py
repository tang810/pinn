# data_utils.py
import numpy as np
import torch
import json
import os
import pandas as pd

def load_user_data(file_path):
    """
    读取用户数据文件 (.csv 或 .npz)，返回标准字典
    返回字典格式: {'x': ..., 'rho': ..., 'h1': ..., 'a': ..., 'ref': ...}
    """
    data = {}
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".csv":
            df = pd.read_csv(file_path)
            data['x'] = df['x'].values if 'x' in df.columns else None
            data['rho'] = df['rho'].values if 'rho' in df.columns else None
            data['h1'] = df['h1'].values if 'h1' in df.columns else None
            data['a'] = df['a'].values if 'a' in df.columns else None
            data['ref'] = df['ref'].values if 'ref' in df.columns else None
        elif ext == ".npz":
            npzfile = np.load(file_path)
            data['x'] = npzfile['x'] if 'x' in npzfile else None
            data['rho'] = npzfile['rho'] if 'rho' in npzfile else None
            data['h1'] = npzfile['h1'] if 'h1' in npzfile else None
            data['a'] = npzfile['a'] if 'a' in npzfile else None
            data['ref'] = npzfile['ref'] if 'ref' in npzfile else None
        else:
            print(f"❌ 不支持的文件类型: {ext}")
            return None
    except Exception as e:
        print(f"❌ 加载用户数据失败: {e}")
        return None

    return data

def generate_virtual_data(n_samples=100, n_x=300, device='cpu'):
    """
    生成虚拟训练数据
    """
    min_x, max_x = -40, 40
    x = np.linspace(min_x, max_x, n_x).reshape(1, -1)
    x = np.repeat(x, n_samples, axis=0)

    rho = np.random.uniform(0.9, 0.99, size=(n_samples, 1))
    h1 = np.random.uniform(4, 10, size=(n_samples, 1))
    a = np.random.uniform(-0.4, -0.05, size=(n_samples, 1))

    # 简单参考解，可以根据你的 ref_sol 改
    h2 = 1
    g = 9.81
    h = h2 / h1
    c0 = np.sqrt((g * h2 * (1-rho)) / (h + rho))
    c1 = -1.5 * c0 * ((rho - h**2) / (rho * h2 + h*h2))
    c2 = (1/6) * c0 * ((rho * h2**2 * h1 + h1**2 * h2) / (rho * h1 + h2))
    c = c0 + a*c1 / 3
    lam = np.sqrt(12*c2 / (a*c1))
    ref = a / (np.cosh(x / lam)**2)

    data = {
        'x': x.astype(np.float32),
        'rho': rho.astype(np.float32),
        'h1': h1.astype(np.float32),
        'a': a.astype(np.float32),
        'ref': ref.astype(np.float32)
    }
    return data

def save_data_info(data, file_path, source='simulation'):
    info = {
        'source': source,
        'num_samples': data['x'].shape[0] if 'x' in data else 0,
        'num_points': data['x'].shape[1] if 'x' in data else 0,
    }
    with open(file_path, 'w') as f:
        json.dump(info, f, indent=2)
    # print(f"📄 数据摘要已保存: {file_path}")

def extract_data_parameters(data, n_samples=100, n_x=300):
    if 'x' in data:
        n_samples = data['x'].shape[0]
        n_x = data['x'].shape[1]
    return n_samples, n_x

def validate_data(data):
    required_keys = ['x', 'rho', 'h1', 'a', 'ref']
    for k in required_keys:
        if k not in data or data[k] is None:
            print(f"❌ 数据缺少字段: {k}")
            return False
    return True
