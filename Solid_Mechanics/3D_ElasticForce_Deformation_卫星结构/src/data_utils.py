import os
import torch
import pandas as pd
import numpy as np
import json
from torch.utils.data import Dataset

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_user_data(data_file):
    """
    加载用户上传的数据文件
    支持格式: CSV, TXT, NPZ
    
    Args:
        data_file (str): 数据文件路径
    
    Returns:
        dict: 包含用户数据的字典，失败时返回None
    """
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return None
    
    try:
        file_ext = os.path.splitext(data_file)[1].lower()
        
        if file_ext == '.csv':
            return _load_csv_data(data_file)
        elif file_ext == '.txt':
            return _load_txt_data(data_file)
        elif file_ext == '.npz':
            return _load_npz_data(data_file)
        else:
            print(f"❌ 不支持的文件格式: {file_ext}. 支持的格式: .csv, .txt, .npz")
            return None
            
    except Exception as e:
        print(f"❌ 加载数据文件时出错: {str(e)}")
        return None

def _load_csv_data(data_file):
    """加载CSV格式数据"""
    df = pd.read_csv(data_file)
    data = {
        'coordinates': df[['x', 'y', 'z']].values if all(col in df.columns for col in ['x', 'y', 'z']) else None,
        'displacements': df[['u', 'v', 'w']].values if all(col in df.columns for col in ['u', 'v', 'w']) else None,
        'forces': df[['fx', 'fy', 'fz']].values if all(col in df.columns for col in ['fx', 'fy', 'fz']) else None,
        'boundary_conditions': df[['bc_type', 'bc_value']].values if all(col in df.columns for col in ['bc_type', 'bc_value']) else None,
        'material_properties': {
            'E': df['E'].iloc[0] if 'E' in df.columns else None,
            'mu': df['mu'].iloc[0] if 'mu' in df.columns else None
        }
    }
    
    # 尝试识别内部点和边界点索引
    if 'point_type' in df.columns:
        interior_mask = df['point_type'] == 'interior'
        boundary_mask = df['point_type'] == 'boundary'
        data['interior_indices'] = np.where(interior_mask)[0] if interior_mask.any() else None
        data['boundary_indices'] = np.where(boundary_mask)[0] if boundary_mask.any() else None
    
    _print_data_summary(data, data_file)
    return data

def _load_txt_data(data_file):
    """加载TXT格式数据"""
    data_array = np.loadtxt(data_file)
    if data_array.shape[1] < 6:
        print("❌ TXT文件格式不正确，需要至少6列数据 (x, y, z, u, v, w)")
        return None
    
    data = {
        'coordinates': data_array[:, :3],
        'displacements': data_array[:, 3:6] if data_array.shape[1] >= 6 else None,
        'forces': data_array[:, 6:9] if data_array.shape[1] >= 9 else None,
        'boundary_conditions': None,
        'material_properties': {'E': None, 'mu': None},
        'interior_indices': None,
        'boundary_indices': None
    }
    
    _print_data_summary(data, data_file)
    return data

def _load_npz_data(data_file):
    """加载NPZ格式数据"""
    npz_data = np.load(data_file)
    data = {
        'coordinates': npz_data['coordinates'] if 'coordinates' in npz_data else None,
        'displacements': npz_data['displacements'] if 'displacements' in npz_data else None,
        'forces': npz_data['forces'] if 'forces' in npz_data else None,
        'boundary_conditions': npz_data['boundary_conditions'] if 'boundary_conditions' in npz_data else None,
        'interior_indices': npz_data['interior_indices'] if 'interior_indices' in npz_data else None,
        'boundary_indices': npz_data['boundary_indices'] if 'boundary_indices' in npz_data else None,
        'material_properties': {
            'E': float(npz_data['E']) if 'E' in npz_data else None,
            'mu': float(npz_data['mu']) if 'mu' in npz_data else None
        }
    }
    
    _print_data_summary(data, data_file)
    return data

def _print_data_summary(data, data_file):
    """打印数据摘要信息"""
    print(f"✅ 成功加载用户数据文件: {data_file}")

def generate_virtual_data(n_interior, n_boundary, E, mu, domain_size=(1.0, 1.0, 1.0), seed=42):
    """
    生成虚拟数据用于测试
    
    Args:
        n_interior (int): 内部点数量
        n_boundary (int): 边界点数量
        E (float): 弹性模量
        mu (float): 泊松比
        domain_size (tuple): 域大小 (Lx, Ly, Lz)
        seed (int): 随机种子，保证结果可重现
    
    Returns:
        dict: 包含虚拟数据的字典
    """
    # 设置随机种子
    np.random.seed(seed)
    
    Lx, Ly, Lz = domain_size
    
    # 生成内部点
    interior_coords = np.random.rand(n_interior, 3) * np.array([Lx, Ly, Lz])
    
    # 生成边界点
    boundary_coords = _generate_boundary_points(n_boundary, Lx, Ly, Lz)
    
    # 合并所有坐标
    all_coords = np.vstack([interior_coords, boundary_coords])
    
    # 生成虚拟位移场和力场
    displacements = _generate_displacement_field(all_coords)
    forces = _generate_force_field(all_coords)
    
    # 生成边界条件
    boundary_conditions = _generate_boundary_conditions(len(boundary_coords))
    
    data = {
        'coordinates': all_coords,
        'displacements': displacements,
        'forces': forces,
        'boundary_conditions': boundary_conditions,
        'interior_indices': np.arange(n_interior),
        'boundary_indices': np.arange(n_interior, len(all_coords)),
        'material_properties': {'E': E, 'mu': mu}
    }
    
    print(f"✅ 虚拟数据生成完成:")
    print(f"   - 总点数: {len(all_coords)}")
    print(f"   - 内部点: {n_interior}")
    print(f"   - 边界点: {len(boundary_coords)}")
    print(f"   - 域大小: {Lx} × {Ly} × {Lz}")
    print("%Line Break%")
    
    return data

def _generate_boundary_points(n_boundary, Lx, Ly, Lz):
    """生成边界点坐标"""
    boundary_coords = []
    n_per_face = max(1, n_boundary // 6)
    
    # 6个面的边界点
    faces = [
        ([0, Lx], [0, Ly], [0]),      # z=0 面
        ([0, Lx], [0, Ly], [Lz]),     # z=Lz 面
        ([0, Lx], [0], [0, Lz]),      # y=0 面
        ([0, Lx], [Ly], [0, Lz]),     # y=Ly 面
        ([0], [0, Ly], [0, Lz]),      # x=0 面
        ([Lx], [0, Ly], [0, Lz])      # x=Lx 面
    ]
    
    for x_range, y_range, z_range in faces:
        for _ in range(n_per_face):
            x = np.random.choice(x_range) if len(x_range) == 1 else np.random.uniform(x_range[0], x_range[1])
            y = np.random.choice(y_range) if len(y_range) == 1 else np.random.uniform(y_range[0], y_range[1])
            z = np.random.choice(z_range) if len(z_range) == 1 else np.random.uniform(z_range[0], z_range[1])
            boundary_coords.append([x, y, z])
    
    return np.array(boundary_coords)

def _generate_displacement_field(coords):
    """生成虚拟位移场（简单的线性分布）"""
    u = 0.001 * coords[:, 0]  # x方向位移
    v = 0.0005 * coords[:, 1]  # y方向位移
    w = 0.0002 * coords[:, 2]  # z方向位移
    return np.column_stack([u, v, w])

def _generate_force_field(coords):
    """生成虚拟力场"""
    return np.random.normal(0, 0.1, (len(coords), 3))

def _generate_boundary_conditions(n_boundary):
    """生成边界条件（简化）"""
    boundary_conditions = np.zeros((n_boundary, 2))
    boundary_conditions[:, 0] = 1  # 位移边界条件类型
    boundary_conditions[:, 1] = 0  # 边界值为0
    return boundary_conditions

def save_data_info(data_info, output_dir):
    """
    保存数据信息到JSON文件
    
    Args:
        data_info (dict): 数据信息字典
        output_dir (str): 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    info_file = os.path.join(output_dir, 'data_info.json')
    
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(data_info, f, indent=2, ensure_ascii=False)
    
    print(f"💾 数据信息已保存到: {info_file}")

def extract_data_parameters(user_data, default_E, default_mu):
    """
    从用户数据中提取参数，如果不存在则使用默认值
    
    Args:
        user_data (dict): 用户数据字典
        default_E (float): 默认弹性模量
        default_mu (float): 默认泊松比
    
    Returns:
        tuple: (E, mu, n_interior, n_boundary)
    """
    # 提取材料参数
    E = user_data['material_properties']['E'] if user_data['material_properties']['E'] is not None else default_E
    mu = user_data['material_properties']['mu'] if user_data['material_properties']['mu'] is not None else default_mu
    
    # 提取采样点数量
    if user_data.get('interior_indices') is not None:
        n_interior = len(user_data['interior_indices'])
    else:
        n_interior = len(user_data['coordinates']) // 2 if user_data['coordinates'] is not None else 5000
    
    if user_data.get('boundary_indices') is not None:
        n_boundary = len(user_data['boundary_indices'])
    else:
        n_boundary = len(user_data['coordinates']) - n_interior if user_data['coordinates'] is not None else 600
    
    return E, mu, n_interior, n_boundary

def validate_data(data):
    """
    验证数据的完整性和有效性
    
    Args:
        data (dict): 数据字典
    
    Returns:
        bool: 数据是否有效
    """
    if data is None:
        print("❌ 数据验证失败: 数据为空")
        return False
    
    # 检查必需的字段
    if data.get('coordinates') is None:
        print("❌ 数据验证失败: 缺少坐标信息")
        return False
    
    coords = data['coordinates']
    if len(coords) == 0:
        print("❌ 数据验证失败: 坐标数据为空")
        return False
    
    if coords.shape[1] != 3:
        print("❌ 数据验证失败: 坐标数据必须是3维 (x, y, z)")
        return False
    
    # 检查数据范围是否合理
    if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
        print("❌ 数据验证失败: 坐标数据包含无效值 (NaN 或 Inf)")
        return False
    
    # 检查其他字段的维度一致性
    n_points = len(coords)
    
    if data.get('displacements') is not None:
        if len(data['displacements']) != n_points:
            print("❌ 数据验证失败: 位移数据长度与坐标不匹配")
            return False
        if np.any(np.isnan(data['displacements'])) or np.any(np.isinf(data['displacements'])):
            print("❌ 数据验证失败: 位移数据包含无效值")
            return False
    
    if data.get('forces') is not None:
        if len(data['forces']) != n_points:
            print("❌ 数据验证失败: 力数据长度与坐标不匹配")
            return False
        if np.any(np.isnan(data['forces'])) or np.any(np.isinf(data['forces'])):
            print("❌ 数据验证失败: 力数据包含无效值")
            return False
    
    # 验证材料参数
    E = data['material_properties'].get('E')
    mu = data['material_properties'].get('mu')
    
    if E is not None and (E <= 0 or np.isnan(E) or np.isinf(E)):
        print("❌ 数据验证失败: 弹性模量必须为正数")
        return False
    
    if mu is not None and (mu <= -1 or mu >= 0.5 or np.isnan(mu) or np.isinf(mu)):
        print("❌ 数据验证失败: 泊松比必须在 (-1, 0.5) 范围内")
        return False
    
    print("✅ 数据验证通过")
    return True

class CustomDataset(Dataset):
    def __init__(self, batch_size, **kwargs):
        """
        弹性力学问题的数据集 - 支持字典参数输入
        
        参数:
            batch_size: 批量大小
            **kwargs: 包含训练数据的字典
        """
        # 从kwargs中提取数据
        if 'x' in kwargs:
            # 直接传递的参数
            x = kwargs['x']
            x1u = kwargs['x1u']
            x1b = kwargs['x1b']
            x2u = kwargs['x2u']
            x2b = kwargs['x2b']
            x3u = kwargs['x3u']
            x3b = kwargs['x3b']
        else:
            # 如果没有直接参数，尝试从其他格式提取
            raise ValueError("需要提供 x, x1u, x1b, x2u, x2b, x3u, x3b 参数")
        
        # 自动检测设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 转换为tensor并移动到指定设备
        self.x = torch.tensor(x, dtype=torch.float32).to(device)
        self.x1u = torch.tensor(x1u, dtype=torch.float32).to(device)
        self.x1b = torch.tensor(x1b, dtype=torch.float32).to(device)
        self.x2u = torch.tensor(x2u, dtype=torch.float32).to(device)
        self.x2b = torch.tensor(x2b, dtype=torch.float32).to(device)
        self.x3u = torch.tensor(x3u, dtype=torch.float32).to(device)
        self.x3b = torch.tensor(x3b, dtype=torch.float32).to(device)
        
        self.batch_size = batch_size
        self.device = device

        # 计算目标值
        self.s3u1 = torch.cos(self.x3u[:, 0] / 2 * np.pi) * torch.cos(self.x3u[:, 1] / 2 * np.pi)
        self.s3u1 = self.s3u1.to(device)
        
        self.N = x.shape[0]
        
        print(f"✓ CustomDataset 初始化成功:")
        print(f"  内部点数量: {self.N}")
        print(f"  批量大小: {self.batch_size}")
        print(f"  设备: {self.device}")
        
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        indices = torch.randperm(self.N, device=self.device)[:self.batch_size]
        x_batch = self.x[indices]

        return x_batch, self.x1u, self.x1b, self.x2u, self.x2b, self.x3u, self.x3b, self.s3u1