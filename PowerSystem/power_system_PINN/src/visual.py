import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
import numpy as np
import os

def plot_losses(losses, save_path="losses.png"):
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.plot([loss['Total Loss'] for loss in losses])
    plt.title('Total Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.subplot(2, 3, 2)
    plt.plot([loss['Rotor Motion Equation'] for loss in losses])
    plt.title('Rotor Motion Equation Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.subplot(2, 3, 3)
    plt.plot([loss['Power Balance'] for loss in losses])
    plt.title('Power Balance Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.subplot(2, 3, 4)
    plt.plot([loss['Power Transfer Equation'] for loss in losses])
    plt.title('Power Transfer Equation Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.subplot(2, 3, 5)
    plt.plot([loss['Boundary Conditions'] for loss in losses])
    plt.title('Boundary Conditions Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_frequency_heatmap(delta_f, t_grid, x_grid, num_nodes, t_max, save_path="frequency_heatmap.png"):
    delta_f_reshaped = delta_f.reshape(t_grid.shape)
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    im = ax.imshow(delta_f_reshaped.T, aspect='auto',
                   extent=[0, t_max, 0, num_nodes - 1],
                   origin='lower', cmap='coolwarm')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax, label='Frequency Increment Δf (Hz)')
    ax.set_title('Spatio-temporal Distribution of Power System Frequency Increment')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Node Number')
    plt.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_node_frequencies(delta_f, t_grid, x_grid, save_path="node_frequencies.png"):
    delta_f_reshaped = delta_f.reshape(t_grid.shape)
    num_nodes = delta_f_reshaped.shape[1]
    plt.figure(figsize=(12, 8))
    step = max(1, num_nodes // 10)  
    selected_nodes = range(0, num_nodes, step)
    for i in selected_nodes:
        plt.plot(t_grid[:, 0], delta_f_reshaped[:, i], label=f'Node {i}')
    plt.title('Frequency Increment vs Time for Each Node')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency Increment Δf (Hz)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_power_comparison(P_e, P_m, t_grid, x_grid, node_idx, save_path=None):
    P_e_reshaped = P_e.reshape(t_grid.shape)
    P_m_reshaped = P_m.reshape(t_grid.shape)
    plt.figure(figsize=(10, 6))
    plt.plot(t_grid[:, 0], P_e_reshaped[:, node_idx], label='Electromagnetic Power ΔPe')
    plt.plot(t_grid[:, 0], P_m_reshaped[:, node_idx], label='Mechanical Power ΔPm', linestyle='--')
    plt.title(f'Power Comparison for Node {node_idx}')
    plt.xlabel('Time (s)')
    plt.ylabel('Power Increment (pu)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    target = save_path or f'power_comparison_node_{node_idx}.png'
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    plt.savefig(target, dpi=300)
    plt.close()

def create_frequency_animation(delta_f, t_grid, x_grid, num_nodes, save_path="frequency_animation.gif"):
    delta_f_reshaped = delta_f.reshape(t_grid.shape)
    num_time_steps = delta_f_reshaped.shape[0]
    vmin = np.min(delta_f_reshaped)
    vmax = np.max(delta_f_reshaped)
    norm = Normalize(vmin=vmin, vmax=vmax)
    fig, ax = plt.subplots(figsize=(10, 6))
    def update(frame):
        ax.clear()
        im = ax.bar(range(num_nodes), delta_f_reshaped[frame, :],
                    color=plt.cm.coolwarm(norm(delta_f_reshaped[frame, :])))
        ax.set_ylim(vmin * 1.1, vmax * 1.1)
        ax.set_title(f'Power System Frequency Increment Distribution (t = {t_grid[frame, 0]:.2f}s)')
        ax.set_xlabel('Node Number')
        ax.set_ylabel('Frequency Increment Δf (Hz)')
        ax.grid(True, alpha=0.3, axis='y')
        return im
    ani = FuncAnimation(fig, update, frames=range(0, num_time_steps, 2), interval=50)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    ani.save(save_path, writer='pillow', fps=15)
    plt.close()

def create_wave_propagation_animation(delta_f, t_grid, x_grid, num_nodes, save_path="wave_propagation.gif"):
    delta_f_reshaped = delta_f.reshape(t_grid.shape)
    num_time_steps = delta_f_reshaped.shape[0]
    vmin = np.min(delta_f_reshaped)
    vmax = np.max(delta_f_reshaped)
    fig, ax = plt.subplots(figsize=(10, 6))
    def update(frame):
        ax.clear()
        ax.plot(range(num_nodes), delta_f_reshaped[frame, :], 'o-', color='b')
        ax.set_ylim(vmin * 1.1, vmax * 1.1)
        ax.set_title(f'Frequency Increment Propagation Along Line (t = {t_grid[frame, 0]:.2f}s)')
        ax.set_xlabel('Node Number')
        ax.set_ylabel('Frequency Increment Δf (Hz)')
        ax.grid(True, alpha=0.3)
        return ax.lines
    ani = FuncAnimation(fig, update, frames=range(0, num_time_steps, 2), interval=50)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    ani.save(save_path, writer='pillow', fps=15)
    plt.close()
