
import torch
import numpy as np

# Physical Parameters (Shared Config)
class Config:
    WIND_U = 0.6
    WIND_V = 0.6
    T_MAX = 1.0
    DIFFUSION_D = 0.03

def get_terrain_tensor(x, y):
    """
    Simulates Terrain Height H(x,y). 
    Returns value in [0, 1].
    """
    # West Mountains
    h1 = 0.8 * torch.exp(-((x + 0.6)**2 + (y - 0.2)**2) / 0.3)
    # North Mountains
    h2 = 0.7 * torch.exp(-((x - 0.2)**2 + (y - 0.8)**2) / 0.4)
    # Base slope
    base = 0.2 * (1.0 - (x + y) / 2.0)
    
    h_total = h1 + h2 + base
    return h_total * 0.5

def sample_spatiotemporal(n_samples, device):
    xy = (torch.rand(n_samples, 2, device=device) * 2.0 - 1.0)
    t = torch.rand(n_samples, 1, device=device) * Config.T_MAX
    return torch.cat([xy, t], dim=1)

def sample_initial_condition(n_samples, device):
    xy = (torch.rand(n_samples, 2, device=device) * 2.0 - 1.0)
    t = torch.zeros(n_samples, 1, device=device)
    # Initial condition: No rain
    return torch.cat([xy, t], dim=1), torch.zeros(n_samples, 1, device=device)

def sample_boundary_condition(n_samples, device):
    xy_b = torch.rand(n_samples, 2, device=device) * 2.0 - 1.0
    mask_side = torch.randint(0, 2, (n_samples,), device=device)
    xy_b[mask_side==0, 0] = torch.sign(xy_b[mask_side==0, 0])
    xy_b[mask_side==1, 1] = torch.sign(xy_b[mask_side==1, 1])
    t = torch.rand(n_samples, 1, device=device) * Config.T_MAX
    # Dirichlet BC: No rain at boundaries
    return torch.cat([xy_b, t], dim=1), torch.zeros(n_samples, 1, device=device)

def generate_data_simul(n_coloc=2000, n_ic=500, n_bc=500, device='cpu'):
    """Generates synthetic physics-informed training batches."""
    x_pde = sample_spatiotemporal(n_coloc, device)
    x_ic, y_ic = sample_initial_condition(n_ic, device)
    x_bc, y_bc = sample_boundary_condition(n_bc, device)
    return x_pde, x_ic, y_ic, x_bc, y_bc

def load_data_from_file(path):
    """Placeholder for loading external data."""
    if path.endswith('.npy'):
        return np.load(path)
    raise NotImplementedError("Only .npy loading implemented.")
