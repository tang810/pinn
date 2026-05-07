
import torch
import torch.nn.functional as F
import time
from .data_loader import get_terrain_tensor, get_temperature_tensor, Config

def pde_residual(model, x_yt):
    x_yt.requires_grad_(True)
    S = model(x_yt)
    
    # Gradients
    grads = torch.autograd.grad(S, x_yt, torch.ones_like(S), create_graph=True)[0]
    S_x, S_y, S_t = grads[:, 0:1], grads[:, 1:2], grads[:, 2:3]
    
    grads_x = torch.autograd.grad(S_x, x_yt, torch.ones_like(S_x), create_graph=True)[0]
    S_xx = grads_x[:, 0:1]
    
    grads_y = torch.autograd.grad(S_y, x_yt, torch.ones_like(S_y), create_graph=True)[0]
    S_yy = grads_y[:, 1:2]
    
    # Coordinates
    x = x_yt[:, 0:1]
    y = x_yt[:, 1:2]
    t = x_yt[:, 2:3]
    
    # --- Physical Terms ---
    H = get_terrain_tensor(x, y)
    grad_h = torch.autograd.grad(H, x_yt, torch.ones_like(H), create_graph=True)[0]
    H_x, H_y = grad_h[:, 0:1], grad_h[:, 1:2]
    
    # Scaled Wind U, V
    uplift = Config.WIND_U * H_x + Config.WIND_V * H_y
    Q_oro = 3.0 * F.relu(uplift) 
    
    # Storm Source
    cx = -0.5 + Config.WIND_U * t 
    cy =  0.8 + Config.WIND_V * t
    Q_storm = 5.0 * torch.exp(-((x - cx)**2 + (y - cy)**2) / 0.2)
    
    # Temperature & Melting
    T = get_temperature_tensor(x, y, t, H)
    Q_melt = Config.MELT_RATE * F.relu(T) * S
    
    lhs = S_t + (Config.WIND_U * S_x + Config.WIND_V * S_y) - Config.DIFFUSION_D * (S_xx + S_yy)
    rhs = Q_storm + Q_oro - Q_melt
    
    return lhs - rhs

def train_model(model, data_gen_fn, iterations=5000, device='cpu', callback=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    
    print(f"Starting Training for {iterations} iters...")
    start_time = time.time()
    
    for it in range(1, iterations + 1):
        optimizer.zero_grad()
        
        x_pde, x_ic, y_ic, x_bc, y_bc = data_gen_fn()
        
        res = pde_residual(model, x_pde)
        loss_pde = torch.mean(res**2)
        
        loss_ic = torch.mean((model(x_ic) - y_ic)**2)
        loss_bc = torch.mean((model(x_bc) - y_bc)**2)
        
        loss = 2.0 * loss_pde + 10.0 * loss_ic + 5.0 * loss_bc
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if it % 500 == 0:
            elapsed = time.time() - start_time
            print(f"Iter {it} | Total Loss: {loss.item():.5f} | PDE: {loss_pde.item():.5f} | IC: {loss_ic.item():.5f} | Time: {elapsed:.1f}s")
            
    return model
