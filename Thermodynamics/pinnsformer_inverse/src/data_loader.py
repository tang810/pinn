import numpy as np
import torch
import pandas as pd
from .utils import LHSample

def get_evaluation_data(args):
    """读取观测数据，返回坐标和温度真值（评估用）"""
    data = pd.read_csv(args.data_path, delimiter=' ', header=None).to_numpy()
    x, y, z, t, T_star = (data[:, i:i+1] for i in range(5))
    
    device = torch.device(args.device)
    
    # 评估时不需要梯度
    x_star = torch.tensor(x, dtype=torch.float32, requires_grad=False).to(device)
    y_star = torch.tensor(y, dtype=torch.float32, requires_grad=False).to(device)
    z_star = torch.tensor(z, dtype=torch.float32, requires_grad=False).to(device)
    t_star = torch.tensor(t, dtype=torch.float32, requires_grad=False).to(device)
    
    T_star_values = T_star.flatten()  # numpy array
    
    return x_star, y_star, z_star, t_star, T_star_values

def get_pinn_domain_points(args):
    """
    使用 LHS 生成训练点（残差、初值、边值）
    返回的张量 requires_grad=False，训练时会自动启用
    """
    device = torch.device(args.device)
    
    num_res_points = 100000
    num_ic_points = 3000
    num_bc_points_per_face = 2000

    domain_bounds = [[0, 1], [0, 1], [0, 1], [0, 1]]  # [x, y, z, t]
    
    # 残差点
    res_points_np = LHSample(4, domain_bounds, num_res_points)
    res_tensor = torch.tensor(res_points_np, dtype=torch.float32, requires_grad=False).to(device)

    # 初始条件点 (t=0)
    ic_bounds_spatial = [[0, 1], [0, 1], [0, 1]]
    ic_points_spatial = LHSample(3, ic_bounds_spatial, num_ic_points)
    ic_points_time = np.zeros((num_ic_points, 1))
    ic_points_np = np.hstack([ic_points_spatial, ic_points_time])
    ic_tensor = torch.tensor(ic_points_np, dtype=torch.float32, requires_grad=False).to(device)

    # 边界点 (六个面)
    bc_tensors = {}
    face_bounds = [[0, 1], [0, 1], [0, 1]]  # (y,z,t) 或 (x,z,t) 等

    # x=0, x=1
    points_x0_yzt = LHSample(3, face_bounds, num_bc_points_per_face)
    points_x1_yzt = LHSample(3, face_bounds, num_bc_points_per_face)
    bc_points_x0 = np.hstack([np.zeros((num_bc_points_per_face, 1)), points_x0_yzt])
    bc_points_x1 = np.hstack([np.ones((num_bc_points_per_face, 1)), points_x1_yzt])
    bc_tensors['x0'] = torch.tensor(bc_points_x0, dtype=torch.float32, requires_grad=False).to(device)
    bc_tensors['x1'] = torch.tensor(bc_points_x1, dtype=torch.float32, requires_grad=False).to(device)

    # y=0, y=1
    points_y0_xzt = LHSample(3, face_bounds, num_bc_points_per_face)
    points_y1_xzt = LHSample(3, face_bounds, num_bc_points_per_face)
    bc_points_y0 = np.hstack([points_y0_xzt[:, :1], np.zeros((num_bc_points_per_face, 1)), points_y0_xzt[:, 1:]])
    bc_points_y1 = np.hstack([points_y1_xzt[:, :1], np.ones((num_bc_points_per_face, 1)), points_y1_xzt[:, 1:]])
    bc_tensors['y0'] = torch.tensor(bc_points_y0, dtype=torch.float32, requires_grad=False).to(device)
    bc_tensors['y1'] = torch.tensor(bc_points_y1, dtype=torch.float32, requires_grad=False).to(device)

    # z=0, z=1
    points_z0_xyt = LHSample(3, face_bounds, num_bc_points_per_face)
    points_z1_xyt = LHSample(3, face_bounds, num_bc_points_per_face)
    bc_points_z0 = np.hstack([points_z0_xyt[:, :2], np.zeros((num_bc_points_per_face, 1)), points_z0_xyt[:, 2:]])
    bc_points_z1 = np.hstack([points_z1_xyt[:, :2], np.ones((num_bc_points_per_face, 1)), points_z1_xyt[:, 2:]])
    bc_tensors['z0'] = torch.tensor(bc_points_z0, dtype=torch.float32, requires_grad=False).to(device)
    bc_tensors['z1'] = torch.tensor(bc_points_z1, dtype=torch.float32, requires_grad=False).to(device)

    return {'res': res_tensor, 'ic': ic_tensor, 'bc': bc_tensors}