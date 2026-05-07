import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io
from .config import device,E,mu,load_data_path,test_data_path,model_path

def material(u_x, u_y, u_z, u_xx, u_xy, u_xz, u_yy, u_yz,u_zz,
            v_x, v_y, v_z, v_xx, v_xy, v_xz, v_yy, v_yz,v_zz,
            w_x, w_y, w_z, w_xx, w_xy, w_xz, w_yy, w_yz,w_zz, E=E, mu=mu):
    """
    计算材料特性和应力
    
    参数:
        u_x, u_y, ... : 位移的一阶和二阶导数
        E: 杨氏模量
        mu: 泊松比
    
    返回:
        e1, e2, e3, e12, e23, e13: 应变
        s1, s2, s3, s12, s23, s13: 应力
        Gex, Gey, Gez: 平衡方程残差
    """

    # 计算拉梅常数
    la = E * mu / (1 + mu) / (1 - 2 * mu)
    G = E / (1 + mu) / 2

    # 计算应变
    e1 = u_x
    e2 = v_y
    e3 = w_z
    e12 = 0.5 * (u_y + v_x)
    e23 = 0.5 * (w_y + v_z)
    e13 = 0.5 * (u_z + w_x)

    # 计算应力
    s1 = (2 * G + la) * e1 + la * e2 + la * e3
    s2 = (2 * G + la) * e2 + la * e1 + la * e3
    s3 = (2 * G + la) * e3 + la * e1 + la * e2
    s12 = 2 * G * e12
    s23 = 2 * G * e23
    s13 = 2 * G * e13

    # 计算平衡方程残差
    Gex = (G + la) * (u_xx + v_xy + w_xz) + G * (u_xx + u_yy + u_zz)
    Gey = (G + la) * (v_yy + u_xy + w_yz) + G * (v_xx + v_yy + v_zz)
    Gez = (G + la) * (w_zz + u_xz + v_yz) + G * (w_xx + w_yy + w_zz)

    return e1, e2, e3, e12, e23, e13, s1, s2, s3, s12, s23, s13, Gex, Gey, Gez

class CustomDataset(Dataset):
    # 设置设备
    print(f"使用设备: {device}")
    
    def __init__(self, x, x1u, x1b, x2u, x2b, x3u, x3b,batch_size):
        """
        弹性力学问题的数据集
        
        参数:
            x: 内部点坐标
            x1u, x1b, x2u, x2b, x3u, x3b: 边界点坐标
        """
        self.x = torch.tensor(x, dtype=torch.float32).to(device)
        self.x1u = torch.tensor(x1u, dtype=torch.float32).to(device)
        self.x1b = torch.tensor(x1b, dtype=torch.float32).to(device)
        self.x2u = torch.tensor(x2u, dtype=torch.float32).to(device)
        self.x2b = torch.tensor(x2b, dtype=torch.float32).to(device)
        self.x3u = torch.tensor(x3u, dtype=torch.float32).to(device)
        self.x3b = torch.tensor(x3b, dtype=torch.float32).to(device)
        self.batch_size = batch_size

        # 计算目标值
        self.s3u1 = torch.cos(self.x3u[:, 0] / 2 * np.pi) * torch.cos(self.x3u[:, 1] / 2 * np.pi)
        self.s3u1 = self.s3u1.to(device)
        
        self.N = x.shape[0]
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        indices = torch.randperm(self.N, device=self.x.device)[:self.batch_size]
        x_batch = self.x[indices]
        y_batch = self.s3u1

        return x_batch, self.x1u, self.x1b, self.x2u, self.x2b, self.x3u, self.x3b,y_batch  

# 加载训练数据
def load_data(file_path=load_data_path):
    print("加载训练数据...")
    # 加载数据
    C = scipy.io.loadmat(file_path)
    x = C['xy']
    x1u = C['x1u']
    x1b = C['x1b']
    x2u = C['x2u']
    x2b = C['x2b']
    x3u = C['x3u']
    x3b = C['x3b']
    ns = C['n'][0, 0]
    batch_size = ns
    # 创建数据集
    dataset = CustomDataset(x, x1u, x1b, x2u, x2b, x3u, x3b,batch_size)
    return dataset


# 加载测试数据
def load_test_data(file_path=test_data_path):
    print("加载测试数据...")
    # 加载测试数据
    test_data = scipy.io.loadmat(file_path)
    test_x = np.array(test_data['X'])
    print(f"测试数据大小: {test_x.shape}")
    return test_x

#保存模型
def save_model(model, filepath=model_path):
    """
    模型参数和训练日志
    """
    log_dict = {
        'loss_total':    model.loss_log,
        'loss_pde':      model.loss_res_log,
        'loss_x_bnd':    model.loss_x_log,
        'loss_y_bnd':    model.loss_y_log,
        'loss_z_bnd':    model.loss_z_log
    }

    checkpoint = {
        'model_state': model.state_dict(),
        'logs':        log_dict
    }

    torch.save(checkpoint, filepath)
    print(f"✅ 模型和日志已保存到 {filepath}")
    
#加载模型    
def load_model(model, filepath=model_path, device=device):
    """
    从 checkpoint 中加载模型参数及训练日志（按 logs 字段组织）。
    """
    ckpt = torch.load(filepath, map_location=device)
    
    # 恢复模型参数
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    
    # 恢复日志
    logs = ckpt.get('logs', {})
    model.loss_log      = logs.get('loss_total', [])
    model.loss_res_log  = logs.get('loss_pde', [])
    model.loss_x_log    = logs.get('loss_x_bnd', [])
    model.loss_y_log    = logs.get('loss_y_bnd', [])
    model.loss_z_log    = logs.get('loss_z_bnd', [])
    
    print(f"✅ 已从 {filepath} 加载模型参数和训练日志")
    return model
