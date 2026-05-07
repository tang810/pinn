
import torch
import torch.nn.functional as F

class Config:
    T_MAX = 2.5
    WIND_U = 0.5
    WIND_V = -0.5
    DIFFUSION_D = 0.02
    MELT_RATE = 8.0
    LAPSE_RATE = 1.2

def get_terrain_tensor(x, y):
    """
    Simulates Beijing Terrain for Snowfall.
    Returns height H in [0, 1].
    """
    # 1. Main NW Mountain Range
    h_mountains = 0.8 * torch.exp(-((x + 0.5)**2 + (y - 0.5)**2) / 0.5)
    # 2. Specific Peaks
    h_peak1 = 0.6 * torch.exp(-((x + 0.8)**2 + (y - 0.2)**2) / 0.1)
    # 3. North Mountains
    h_peak2 = 0.5 * torch.exp(-((x - 0.2)**2 + (y - 0.8)**2) / 0.15)
    # 4. Base slope
    h_slope = 0.3 * ( (1 - x) + (1 + y) ) / 4.0 
    
    raw_h = h_mountains + h_peak1 + h_peak2 + h_slope
    return torch.clamp(raw_h, 0, 1.2)

def get_temperature_tensor(x, y, t, H):
    """
    Temperature Field T(x,y,t, H).
    """
    base_temp_initial = 0.5 
    cooling = 1.5 * t       
    altitude_cooling = Config.LAPSE_RATE * H
    lat_factor = -0.2 * y
    
    T = base_temp_initial - cooling - altitude_cooling + lat_factor
    return T

def generate_data_simul(n_coloc=4000, n_ic=1000, n_bc=1000, device='cpu'):
    """Generates synthetic training batches."""
    # Colocation points
    xy = (torch.rand(n_coloc, 2, device=device) * 2.0 - 1.0)
    t = torch.rand(n_coloc, 1, device=device) * Config.T_MAX
    x_pde = torch.cat([xy, t], dim=1)
    
    # IC: t=0, S=0
    xy_ic = (torch.rand(n_ic, 2, device=device) * 2.0 - 1.0)
    t_ic = torch.zeros(n_ic, 1, device=device)
    x_ic = torch.cat([xy_ic, t_ic], dim=1)
    y_ic = torch.zeros(n_ic, 1, device=device)
    
    # BC: Zero boundary condition
    xy_bc = torch.rand(n_bc, 2, device=device) * 2.0 - 1.0
    mask = torch.randint(0, 2, (n_bc,), device=device) 
    side = torch.randint(0, 2, (n_bc,), device=device) * 2.0 - 1.0 
    
    xy_bc[mask==0, 0] = side[mask==0]
    xy_bc[mask==1, 1] = side[mask==1]
    
    t_bc = torch.rand(n_bc, 1, device=device) * Config.T_MAX
    x_bc = torch.cat([xy_bc, t_bc], dim=1)
    y_bc = torch.zeros(n_bc, 1, device=device)
    
    return x_pde, x_ic, y_ic, x_bc, y_bc

def load_data_from_file(path):
    import numpy as np
    if path.endswith('.npy'):
        return np.load(path)
    raise NotImplementedError("Only .npy implemented")
