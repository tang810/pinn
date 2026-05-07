
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.animation as animation
from .data_loader import get_terrain_tensor, Config

def plot_result(model, t_query, device, geojson_path, save_path=None):
    """
    Plots rainfall field, terrain, and wind vectors at a specific time t.
    """
    # 1. Map Data
    try:
        gdf = gpd.read_file(geojson_path)
        minx, miny, maxx, maxy = gdf.total_bounds
        map_boundary = gdf.unary_union
        has_map = True
    except Exception as e:
        print(f"GeoJSON not found ({e}), utilizing default box.")
        minx, miny, maxx, maxy = 115.4, 39.4, 117.5, 41.1
        has_map = False

    # 2. Grid Generation
    N_GRID = 250
    x_norm = np.linspace(-1, 1, N_GRID)
    y_norm = np.linspace(-1, 1, N_GRID)
    X_norm, Y_norm = np.meshgrid(x_norm, y_norm)
    
    # Input Tensor
    xy_flat = np.vstack([X_norm.flatten(), Y_norm.flatten()]).T
    t_flat = np.full((xy_flat.shape[0], 1), t_query)
    x_input = np.hstack([xy_flat, t_flat])
    x_tensor = torch.tensor(x_input, dtype=torch.float32, device=device)
    
    # 3. Model Prediction
    with torch.no_grad():
        pred_rain = model(x_tensor)
        pred_rain = torch.relu(pred_rain).cpu().numpy().reshape(N_GRID, N_GRID)
        
        x_t = torch.tensor(X_norm, dtype=torch.float32, device=device)
        y_t = torch.tensor(Y_norm, dtype=torch.float32, device=device)
        terrain = get_terrain_tensor(x_t, y_t).cpu().numpy()

    # 4. Mapping & Masking
    Lon = ((X_norm + 1) / 2) * (maxx - minx) + minx
    Lat = ((Y_norm + 1) / 2) * (maxy - miny) + miny
    
    if has_map:
        points = [Point(lon, lat) for lon, lat in zip(Lon.flatten(), Lat.flatten())]
        mask_flat = np.array([map_boundary.contains(p) for p in points])
        mask = mask_flat.reshape(N_GRID, N_GRID)
        
        pred_rain_masked = np.where(mask, pred_rain, np.nan)
        terrain_masked = np.where(mask, terrain, np.nan)
    else:
        mask = np.ones((N_GRID, N_GRID), dtype=bool)
        pred_rain_masked = pred_rain
        terrain_masked = terrain

    # 5. Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Layer 1: Terrain
    ax.imshow(terrain_masked, extent=[minx, maxx, miny, maxy], 
              origin='lower', cmap='gist_earth', alpha=0.5, vmin=-0.2)
    
    # Layer 2: Rainfall
    if np.nanmax(pred_rain_masked) > 0.01:
        rain_levels = np.linspace(0.01, np.nanmax(pred_rain_masked), 50)
        rain_contour = ax.contourf(Lon, Lat, pred_rain_masked, levels=rain_levels, 
                                  cmap='jet', alpha=0.6, extend='max')
        cbar = plt.colorbar(rain_contour, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Rainfall Intensity (mm/h)', rotation=270, labelpad=15)
    
    # Layer 3: Boundary
    if has_map:
        gdf.boundary.plot(ax=ax, color='black', linewidth=1.2)
        
    # Layer 4: Wind
    step = 20
    q_lon = Lon[::step, ::step]
    q_lat = Lat[::step, ::step]
    q_u = np.full_like(q_lon, Config.WIND_U)
    q_v = np.full_like(q_lon, Config.WIND_V)
    
    if has_map:
        mask_sub = mask[::step, ::step]
        q_lon = q_lon[mask_sub]
        q_lat = q_lat[mask_sub]
        q_u = q_u[mask_sub]
        q_v = q_v[mask_sub]

    ax.quiver(q_lon, q_lat, q_u, q_v, color='white', scale=10, width=0.003, alpha=0.8, label='Wind Vector')
    
    ax.set_title(f"Terrain-Aware PINN Rainfall Prediction\nTime: {t_query:.2f}s", fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    # plt.show()
    plt.close()

def make_gif(model, device, geojson_path, save_path="rainfall_simulation.gif"):
    """Generates a GIF of the simulation."""
    print("Generating animation...")
    
    # (Simplified Setup similar to plot_result, kept shared for brevity)
    try:
        gdf = gpd.read_file(geojson_path)
        minx, miny, maxx, maxy = gdf.total_bounds
        map_boundary = gdf.unary_union
        has_map = True
    except:
        minx, miny, maxx, maxy = 115.4, 39.4, 117.5, 41.1
        has_map = False

    N_GRID = 150
    x_norm = np.linspace(-1, 1, N_GRID)
    y_norm = np.linspace(-1, 1, N_GRID)
    X_norm, Y_norm = np.meshgrid(x_norm, y_norm)
    Lon = ((X_norm + 1) / 2) * (maxx - minx) + minx
    Lat = ((Y_norm + 1) / 2) * (maxy - miny) + miny
    
    mask = None
    if has_map:
        points = [Point(lon, lat) for lon, lat in zip(Lon.flatten(), Lat.flatten())]
        mask = np.array([map_boundary.contains(p) for p in points]).reshape(N_GRID, N_GRID)
    
    x_t = torch.tensor(X_norm, dtype=torch.float32, device=device)
    y_t = torch.tensor(Y_norm, dtype=torch.float32, device=device)
    terrain = get_terrain_tensor(x_t, y_t).cpu().numpy()
    if has_map: terrain = np.where(mask, terrain, np.nan)

    fig, ax = plt.subplots(figsize=(8, 6))
    
    def update(frame):
        t_val = frame / 20.0 
        xy_flat = np.vstack([X_norm.flatten(), Y_norm.flatten()]).T
        t_flat = np.full((xy_flat.shape[0], 1), t_val)
        x_in = torch.tensor(np.hstack([xy_flat, t_flat]), dtype=torch.float32, device=device)
        
        with torch.no_grad():
            pred = model(x_in)
            pred = torch.relu(pred).cpu().numpy().reshape(N_GRID, N_GRID)
        
        if has_map: pred = np.where(mask, pred, np.nan)
        
        ax.clear()
        ax.imshow(terrain, extent=[minx, maxx, miny, maxy], origin='lower', cmap='gist_earth', alpha=0.5, vmin=-0.2)
        if has_map: gdf.boundary.plot(ax=ax, color='black', linewidth=1)
        
        if np.nanmax(pred) > 0.01:
            levels = np.linspace(0.01, max(np.nanmax(pred), 1.0), 30)
            ax.contourf(Lon, Lat, pred, levels=levels, cmap='jet', alpha=0.7, extend='max')
        
        ax.set_title(f"Time: {t_val:.2f}s")
        return ax.collections

    ani = animation.FuncAnimation(fig, update, frames=21, interval=200)
    ani.save(save_path, writer='pillow', fps=5)
    print(f"GIF saved to {save_path}")
    plt.close()
