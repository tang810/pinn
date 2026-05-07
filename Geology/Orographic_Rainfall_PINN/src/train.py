
import torch
import torch.nn.functional as F
from .data_loader import get_terrain_tensor, Config

def pde_residual_terrain(model, x_yt):
    """
    Computes PDE residual: R_t + u.Grad(R) - D.Lap(R) - S_storm - S_terrain = 0
    """
    x_yt.requires_grad_(True)
    R = model(x_yt)

    # 1. Gradients of R
    grads = torch.autograd.grad(R, x_yt, torch.ones_like(R), create_graph=True)[0]
    R_x, R_y, R_t = grads[:, 0:1], grads[:, 1:2], grads[:, 2:3]
    
    grads_x = torch.autograd.grad(R_x, x_yt, torch.ones_like(R_x), create_graph=True)[0]
    R_xx = grads_x[:, 0:1]
    
    grads_y = torch.autograd.grad(R_y, x_yt, torch.ones_like(R_y), create_graph=True)[0]
    R_yy = grads_y[:, 1:2]

    # 2. Terrain Terms
    x = x_yt[:, 0:1]
    y = x_yt[:, 1:2]
    t = x_yt[:, 2:3]
    
    H = get_terrain_tensor(x, y)
    grad_h = torch.autograd.grad(H, x_yt, torch.ones_like(H), create_graph=True)[0]
    H_x = grad_h[:, 0:1]
    H_y = grad_h[:, 1:2]

    # 3. Orographic Source (Lifting)
    orographic_factor = Config.WIND_U * H_x + Config.WIND_V * H_y
    S_terrain = 5.0 * F.relu(orographic_factor)

    # 4. Storm Source (Moving Gaussian)
    center_x = -0.8 + 0.8 * t 
    center_y = -0.8 + 0.8 * t
    S_storm = 8.0 * torch.exp(-((x - center_x)**2 + (y - center_y)**2) / 0.08)

    # 5. Residual
    lhs = R_t + (Config.WIND_U * R_x + Config.WIND_V * R_y) - Config.DIFFUSION_D * (R_xx + R_yy)
    rhs = S_storm + S_terrain
    
    return lhs - rhs

def train_model(model, generate_data_fn, iterations=2000, device='cpu', callback=None):
    """
    Main training loop.
    generate_data_fn: Function that returns (x_pde, x_ic, y_ic, x_bc, y_bc)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(f"Start Training (Iterations: {iterations})...")
    
    for it in range(1, iterations + 1):
        optimizer.zero_grad()
        
        # Get Data
        x_pde, x_ic, y_ic, x_bc, y_bc = generate_data_fn()
        
        # PDE Loss
        loss_pde = torch.mean(pde_residual_terrain(model, x_pde) ** 2)
        
        # IC/BC Loss
        loss_ic = torch.mean((model(x_ic) - y_ic) ** 2)
        loss_bc = torch.mean((model(x_bc) - y_bc) ** 2)
        
        # Total Loss
        loss = 2.0 * loss_pde + 5.0 * loss_ic + 5.0 * loss_bc
        
        loss.backward()
        optimizer.step()
        
        if it % 500 == 0 or it == 1:
            print(f"Iter {it} | Loss: {loss.item():.4f} | PDE: {loss_pde.item():.4f}")
            if callback: callback(it, loss.item())
            
    return model
