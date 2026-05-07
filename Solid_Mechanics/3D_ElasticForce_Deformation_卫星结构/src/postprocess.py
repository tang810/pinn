import torch
import numpy as np
import matplotlib.pyplot as plt
from src.pinn_model import MultiFNN, material

# 设置设备
device = torch.device("cpu")

def plot_results_3d(vc, x, title):
    """
    3D结果可视化函数 - 修改为不显示弹窗版本
    """
    # 使用当前活跃的figure，如果没有则创建新的
    if plt.get_fignums():
        # 如果已经有figure，使用当前的
        fig = plt.gcf()
        fig.clear()  # 清除当前内容
    else:
        # 创建新的figure
        fig = plt.figure(figsize=(12, 7))
    
    ax = fig.add_subplot(projection='3d')

    # 绘制所有点，不进行采样
    plot_every = 1
    x_plot, vc_plot = x[::plot_every], vc[::plot_every]

    vmax, vmin = np.percentile(vc_plot, 99), np.percentile(vc_plot, 1)

    # 极大增加点密度的设置
    im = ax.scatter(x_plot[:, 0], x_plot[:, 1], x_plot[:, 2],
                    c=vc_plot, cmap='jet', vmin=vmin, vmax=vmax, 
                    s=10,           # 进一步增加点大小到50
                    alpha=0.8,      # 添加透明度，让重叠区域更明显
                    edgecolors='none',  # 去除点的边缘线，让点更紧密
                    rasterized=True)    # 栅格化渲染，提高性能

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.set_box_aspect((np.ptp(x_plot[:, 0]), np.ptp(x_plot[:, 1]), np.ptp(x_plot[:, 2])))
    
    # 优化颜色条
    fig.colorbar(im, ax=ax, format='%.3e', fraction=0.02, shrink=0.8)
    
    # 移除网格线，让点更突出
    ax.grid(False)
    
    # 设置更好的视角
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    # ✅ 返回figure和axes，让调用者控制保存和显示
    return fig, ax

def predict(self, x_coords, batch_size=10000):
    self.eval(); u, v, w = [np.zeros(x_coords.shape[0]) for _ in range(3)]
    with torch.no_grad():
        for i in range(0, x_coords.shape[0], batch_size):
            end = min(i + batch_size, x_coords.shape[0])
            xb = torch.tensor(x_coords[i:end], dtype=torch.float32).to(device)
            u[i:end], v[i:end], w[i:end] = self.u_model(xb[:,0],xb[:,1],xb[:,2]).cpu().numpy(), self.v_model(xb[:,0],xb[:,1],xb[:,2]).cpu().numpy(), self.w_model(xb[:,0],xb[:,1],xb[:,2]).cpu().numpy()
    self.train(); return u, v, w
MultiFNN.predict = predict

def predict_stress(self, x_coords, batch_size=10000):
    self.eval() 
    stresses = [np.empty(x_coords.shape[0], dtype=np.float32) for _ in range(6)]
    # ✨ 关键修复：移除 with torch.no_grad()
    for i in range((x_coords.shape[0] + batch_size - 1) // batch_size):
        start, end = i * batch_size, min((i + 1) * batch_size, x_coords.shape[0])
        xb = torch.tensor(x_coords[start:end], dtype=torch.float32).to(device)
        derivs = self.compute_derivatives(xb[:,0], xb[:,1], xb[:,2])
        s = material(*derivs)
        for j in range(6): 
            stresses[j][start:end] = s[j+6].detach().cpu().numpy()
    self.train()
    return tuple(stresses)
MultiFNN.predict_stress = predict_stress