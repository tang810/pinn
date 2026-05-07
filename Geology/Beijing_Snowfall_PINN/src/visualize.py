
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import geopandas as gpd
from shapely.geometry import Point
from .data_loader import get_terrain_tensor, get_temperature_tensor, Config

def plot_result(model, t_val, device, geojson_path, save_path=None):
    """Visualizes snowfall at time t_val."""
    # GeoJSON
    try:
        gdf = gpd.read_file(geojson_path)
        minx, miny, maxx, maxy = gdf.total_bounds
        map_boundary = gdf.unary_union
        has_map = True
    except:
        minx, miny, maxx, maxy = 115.4, 39.4, 117.5, 41.1
        has_map = False
        gdf = None

    N = 300
    x = np.linspace(minx, maxx, N)
    y = np.linspace(miny, maxy, N)
    X, Y = np.meshgrid(x, y)
    
    X_norm = 2.0 * (X - minx) / (maxx - minx) - 1.0
    Y_norm = 2.0 * (Y - miny) / (maxy - miny) - 1.0
    
    inp_x = X_norm.flatten()
    inp_y = Y_norm.flatten()
    inp_t = np.full_like(inp_x, t_val)
    inp_tensor = torch.tensor(np.stack([inp_x, inp_y, inp_t], axis=1), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        S_pred = model(inp_tensor).cpu().numpy().flatten()
        
        # Aux Physics
        tx = inp_tensor[:, 0:1]
        ty = inp_tensor[:, 1:2]
        H_pred = get_terrain_tensor(tx, ty).cpu().numpy().flatten()
        Temp_pred = get_temperature_tensor(tx, ty, t_val, torch.tensor(H_pred).to(device).unsqueeze(1)).cpu().numpy().flatten()

    S_grid = S_pred.reshape(N, N)
    H_grid = H_pred.reshape(N, N)
    T_grid = Temp_pred.reshape(N, N)
    
    if has_map:
        points = [Point(gx, gy) for gx, gy in zip(X.flatten(), Y.flatten())]
        mask_flat = np.array([map_boundary.contains(p) for p in points])
        mask = mask_flat.reshape(N, N)
        S_grid[~mask] = np.nan
        H_grid[~mask] = np.nan
        T_grid[~mask] = np.nan

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 1. Terrain
    ax.imshow(H_grid, extent=[minx, maxx, miny, maxy], origin='lower', 
              cmap='terrain', alpha=0.5, vmin=-0.1, vmax=1.5)
    
    # 2. Snow
    if np.nanmax(S_grid) > 0.01:
        levels = np.linspace(0.01, np.nanmax(S_grid), 40)
        ax.contourf(X, Y, S_grid, levels=levels, cmap='PuBu', alpha=0.9, extend='max')
        
    # 3. Freezing Line
    if np.nanmin(T_grid) < 0 < np.nanmax(T_grid):
        cs_temp = ax.contour(X, Y, T_grid, levels=[0], colors='red', linewidths=2.5, linestyles='--')
        ax.clabel(cs_temp, inline=True, fmt='0°C', fontsize=10)
        
    if gdf is not None:
        gdf.boundary.plot(ax=ax, color='black', linewidth=1.5, alpha=0.6)
        
    ax.set_title(f"Beijing Snowfall Simulation (PINN)\nTime = {t_val:.2f}", fontsize=15)
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    plt.close()

def make_gif(model, device, geojson_path, save_path="snowfall_simulation.gif"):
    print("Generating snowfall animation...")
    try:
        gdf = gpd.read_file(geojson_path)
        minx, miny, maxx, maxy = gdf.total_bounds
        map_boundary = gdf.unary_union
        has_map = True
    except:
        minx, miny, maxx, maxy = 115.4, 39.4, 117.5, 41.1
        has_map = False
        gdf = None

    N_GRID = 150
    x_grid = np.linspace(minx, maxx, N_GRID)
    y_grid = np.linspace(miny, maxy, N_GRID)
    X, Y = np.meshgrid(x_grid, y_grid)
    X_norm = 2.0 * (X - minx) / (maxx - minx) - 1.0
    Y_norm = 2.0 * (Y - miny) / (maxy - miny) - 1.0
    
    # Pre-calc static
    t_x = torch.tensor(X_norm, dtype=torch.float32, device=device)
    t_y = torch.tensor(Y_norm, dtype=torch.float32, device=device)
    H_grid = get_terrain_tensor(t_x, t_y).cpu().numpy()
    
    mask = None
    if has_map:
        points = [Point(gx, gy) for gx, gy in zip(X.flatten(), Y.flatten())]
        mask_flat = np.array([map_boundary.contains(p) for p in points])
        mask = mask_flat.reshape(N_GRID, N_GRID)
        H_grid[~mask] = np.nan

    fig, ax = plt.subplots(figsize=(10, 8))
    
    def update(frame):
        frames_total = 40
        t_val = (frame / frames_total) * Config.T_MAX 
        
        xy_flat = np.vstack([X_norm.flatten(), Y_norm.flatten()]).T
        t_flat = np.full((xy_flat.shape[0], 1), t_val)
        x_in = torch.tensor(np.hstack([xy_flat, t_flat]), dtype=torch.float32, device=device)
        
        with torch.no_grad():
            pred_s = model(x_in).cpu().numpy().reshape(N_GRID, N_GRID)
            H_flat = get_terrain_tensor(x_in[:,0:1], x_in[:,1:2])
            pred_t = get_temperature_tensor(x_in[:,0:1], x_in[:,1:2], x_in[:,2:3], H_flat).cpu().numpy().reshape(N_GRID, N_GRID)
            
        if has_map:
            pred_s[~mask] = np.nan
            pred_t[~mask] = np.nan
        
        ax.clear()
        ax.imshow(H_grid, extent=[minx, maxx, miny, maxy], origin='lower', cmap='terrain', alpha=0.5, vmin=-0.1, vmax=1.5)
        if has_map: gdf.boundary.plot(ax=ax, color='black', linewidth=1)
            
        if np.nanmax(pred_s) > 0.01:
            levels = np.linspace(0.01, max(np.nanmax(pred_s), 1.0), 30)
            ax.contourf(X, Y, pred_s, levels=levels, cmap='PuBu', alpha=0.8, extend='max')
            
        if np.nanmin(pred_t) < 0 < np.nanmax(pred_t):
            ax.contour(X, Y, pred_t, levels=[0], colors='red', linewidths=2, linestyles='--')
            
        ax.set_title(f"Beijing Snowfall PINN Simulation\nTime: {t_val:.2f} / {Config.T_MAX}", fontsize=14)
        return ax.collections

    ani = animation.FuncAnimation(fig, update, frames=41, interval=150)
    ani.save(save_path, writer='pillow', fps=10)
    print(f"GIF saved to {save_path}")
    plt.close()
