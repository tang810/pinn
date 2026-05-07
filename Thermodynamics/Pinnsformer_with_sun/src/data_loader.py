import numpy as np
import torch
import pandas as pd
import copy

from .utils import LHSample

def get_clones(module, N):
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def get_n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def make_time_sequence(src_tensor, num_step, step):
    src_expanded = src_tensor.unsqueeze(1).repeat(1, num_step, 1)
    time_offset = torch.arange(num_step, device=src_tensor.device, dtype=torch.float32) * step
    src_expanded[..., 3] += time_offset
    return src_expanded

def get_separated_points(args):
    device = torch.device(args.device)

    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)
    z = np.linspace(0, 1, 10)
    t = np.linspace(0, 1, 6)
    
    x_mesh, y_mesh, z_mesh, t_mesh = np.meshgrid(x, y, z, t, indexing='ij')
    all_points = np.stack([x_mesh, y_mesh, z_mesh, t_mesh], axis=-1).reshape(-1, 4)

    ic_mask = np.isclose(all_points[:, 3], 0)
    
    bc_x0_mask = np.isclose(all_points[:, 0], 0) & ~ic_mask
    bc_x1_mask = np.isclose(all_points[:, 0], 1) & ~ic_mask
    bc_y0_mask = np.isclose(all_points[:, 1], 0) & ~ic_mask
    bc_y1_mask = np.isclose(all_points[:, 1], 1) & ~ic_mask
    bc_z0_mask = np.isclose(all_points[:, 2], 0) & ~ic_mask
    bc_z1_mask = np.isclose(all_points[:, 2], 1) & ~ic_mask
    
    bc_combined_mask = (bc_x0_mask | bc_x1_mask | bc_y0_mask | 
                        bc_y1_mask | bc_z0_mask | bc_z1_mask)
    res_mask = ~ic_mask & ~bc_combined_mask

    res_tensor = torch.tensor(all_points[res_mask], dtype=torch.float32, requires_grad=True).to(device)
    ic_tensor = torch.tensor(all_points[ic_mask], dtype=torch.float32, requires_grad=True).to(device)
    
    bc_tensors = {
        'x0': torch.tensor(all_points[bc_x0_mask], dtype=torch.float32, requires_grad=True).to(device),
        'x1': torch.tensor(all_points[bc_x1_mask], dtype=torch.float32, requires_grad=True).to(device),
        'y0': torch.tensor(all_points[bc_y0_mask], dtype=torch.float32, requires_grad=True).to(device),
        'y1': torch.tensor(all_points[bc_y1_mask], dtype=torch.float32, requires_grad=True).to(device),
        'z0': torch.tensor(all_points[bc_z0_mask], dtype=torch.float32, requires_grad=True).to(device),
        'z1': torch.tensor(all_points[bc_z1_mask], dtype=torch.float32, requires_grad=True).to(device),
    }

    return {'res': res_tensor, 'ic': ic_tensor, 'bc': bc_tensors}

def get_pinn_domain_points(args):
    """
    Generates training points using Latin Hypercube Sampling.
    This function is refactored to match the output of get_separated_points.
    """
    device = torch.device(args.device)
    
    num_res_points = 4096
    num_ic_points = 1024
    num_bc_points_per_face = 400

    domain_bounds = [[0, 1], [0, 1], [0, 1], [0, 1]] # [x, y, z, t]
    
    # Residual/PDE points
    res_points_np = LHSample(4, domain_bounds, num_res_points)
    res_tensor = torch.tensor(res_points_np, dtype=torch.float32, requires_grad=True).to(device)

    # IC points
    ic_bounds_spatial = [[0, 1], [0, 1], [0, 1]]
    ic_points_spatial = LHSample(3, ic_bounds_spatial, num_ic_points)
    ic_points_time = np.zeros((num_ic_points, 1))
    ic_points_np = np.hstack([ic_points_spatial, ic_points_time])
    ic_tensor = torch.tensor(ic_points_np, dtype=torch.float32, requires_grad=True).to(device)

    # BC points
    bc_tensors = {}
    face_bounds = [[0, 1], [0, 1], [0, 1]]

    # x=0, x=1
    points_x0_yzt = LHSample(3, face_bounds, num_bc_points_per_face)
    points_x1_yzt = LHSample(3, face_bounds, num_bc_points_per_face)
    bc_points_x0 = np.hstack([np.zeros((num_bc_points_per_face, 1)), points_x0_yzt])
    bc_points_x1 = np.hstack([np.ones((num_bc_points_per_face, 1)), points_x1_yzt])
    bc_tensors['x0'] = torch.tensor(bc_points_x0[:, [0, 1, 2, 3]], dtype=torch.float32, requires_grad=True).to(device) # Reorder if needed, here it is x,y,z,t
    bc_tensors['x1'] = torch.tensor(bc_points_x1[:, [0, 1, 2, 3]], dtype=torch.float32, requires_grad=True).to(device)

    # y=0, y=1
    points_y0_xzt = LHSample(3, face_bounds, num_bc_points_per_face)
    points_y1_xzt = LHSample(3, face_bounds, num_bc_points_per_face)
    bc_points_y0 = np.hstack([points_y0_xzt[:,:1], np.zeros((num_bc_points_per_face, 1)), points_y0_xzt[:,1:]])
    bc_points_y1 = np.hstack([points_y1_xzt[:,:1], np.ones((num_bc_points_per_face, 1)), points_y1_xzt[:,1:]])
    bc_tensors['y0'] = torch.tensor(bc_points_y0[:, [0, 1, 2, 3]], dtype=torch.float32, requires_grad=True).to(device)
    bc_tensors['y1'] = torch.tensor(bc_points_y1[:, [0, 1, 2, 3]], dtype=torch.float32, requires_grad=True).to(device)

    # z=0, z=1
    points_z0_xyt = LHSample(3, face_bounds, num_bc_points_per_face)
    points_z1_xyt = LHSample(3, face_bounds, num_bc_points_per_face)
    bc_points_z0 = np.hstack([points_z0_xyt[:,:2], np.zeros((num_bc_points_per_face, 1)), points_z0_xyt[:,2:]])
    bc_points_z1 = np.hstack([points_z1_xyt[:,:2], np.ones((num_bc_points_per_face, 1)), points_z1_xyt[:,2:]])
    bc_tensors['z0'] = torch.tensor(bc_points_z0[:, [0, 1, 2, 3]], dtype=torch.float32, requires_grad=True).to(device)
    bc_tensors['z1'] = torch.tensor(bc_points_z1[:, [0, 1, 2, 3]], dtype=torch.float32, requires_grad=True).to(device)

    return {'res': res_tensor, 'ic': ic_tensor, 'bc': bc_tensors}

def get_evaluation_data(args):
    data = pd.read_csv(args.data_path, delimiter=' ', header=None).to_numpy()
    x, y, z, t, T_star = (data[:, i:i+1] for i in range(5))
    
    device = torch.device(args.device)
    
    x_star = torch.tensor(x, dtype=torch.float32, requires_grad=True).to(device)
    y_star = torch.tensor(y, dtype=torch.float32, requires_grad=True).to(device)
    z_star = torch.tensor(z, dtype=torch.float32, requires_grad=True).to(device)
    t_star = torch.tensor(t, dtype=torch.float32, requires_grad=True).to(device)
    
    T_star_values = T_star.flatten()
    
    return x_star, y_star, z_star, t_star, T_star_values
