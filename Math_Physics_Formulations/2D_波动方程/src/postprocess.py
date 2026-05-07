import os
import numpy as np
import matplotlib.pyplot as plt
# plt.style.use("dark_background")
# 全局设置透明背景
plt.rcParams["figure.facecolor"] = (0, 0, 0, 0)   # 整个画布透明
plt.rcParams["axes.facecolor"]   = (0, 0, 0, 0)   # 坐标区透明
plt.rcParams["savefig.transparent"] = True        # 保存图片透明
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def plot_2d_snapshots(u_list, t, title_prefix="Wave Snapshot", save_path=None):
    time_indices = [0, len(t) // 2, -1]
    fig, axes = plt.subplots(1, len(time_indices), figsize=(15, 5))

    for idx, ax in zip(time_indices, axes):
        ax.imshow(u_list[idx], cmap='viridis', origin='lower')
        ax.set_title(f"{title_prefix} at t = {t[idx]:.3f}s")
        ax.axis('off')

    if save_path:
        ensure_dir(save_path)
        plt.savefig(save_path, dpi=300)
        #print(f"✅ 2D 快照已保存")
    plt.close(fig)

def plot_3d_snapshots(u_list, x, y, t, time_indices=None, save_path=None):
    if time_indices is None:
        time_indices = [0, len(t) // 2, -1]

    X, Y = np.meshgrid(x, y)
    fig = plt.figure(figsize=(5 * len(time_indices), 4))

    for i, idx in enumerate(time_indices):
        ax = fig.add_subplot(1, len(time_indices), i + 1, projection='3d')
        ax.plot_surface(X, Y, u_list[idx], cmap='viridis', linewidth=0)
        ax.set_title(f't = {t[idx]:.3f}s')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('u(x,y)')

    if save_path:
        ensure_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        #print(f"✅ 3D 快照已保存")
    plt.close(fig)

def create_3d_animation(u_list, x, y, t, save_path='wave_animation.gif'):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, y)
    surf = [ax.plot_surface(X, Y, u_list[0], cmap='viridis', linewidth=0)]
    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

    def update(frame):
        surf[0].remove()
        surf[0] = ax.plot_surface(X, Y, u_list[frame], cmap='viridis', linewidth=0)
        time_text.set_text(f'Time: {t[frame]:.3f}s')
        return surf[0], time_text

    anim = animation.FuncAnimation(fig, update, frames=len(t), interval=50, blit=False)

    if save_path:
        ensure_dir(save_path)
        try:
            if save_path.endswith('.gif'):
                anim.save(save_path, writer=animation.PillowWriter(fps=20))
            elif save_path.endswith('.mp4'):
                anim.save(save_path, writer=animation.FFMpegWriter(fps=20))
            #print(f"✅ 3D 动画已保存")
        except Exception as e:
            print(f"❗ 动画保存失败: {e}")
    plt.close(fig)

def save_numpy_result(u_list, save_path):
    ensure_dir(save_path)
    np.save(save_path, u_list)
    #print(f"✅ 结果数组已保存")

def save_interactive_plot(u_list, x, y, t, save_html="interactive_wave.html"):
    try:
        import plotly.graph_objects as go
        X, Y = np.meshgrid(x, y)
        frames = [
            go.Frame(data=[go.Surface(z=u_list[i], x=X, y=Y)], name=f"t={t[i]:.3f}")
            for i in range(len(t))
        ]

        fig = go.Figure(data=[go.Surface(z=u_list[0], x=X, y=Y)], frames=frames)
        fig.update_layout(
            title="Interactive Wave Solution",
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="u(x,y)"),
            updatemenus=[{
                "buttons": [
                    {"args": [None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}],
                     "label": "Play", "method": "animate"},
                    {"args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                     "label": "Pause", "method": "animate"}
                ],
                "type": "buttons"
            }]
        )

        if save_html:
            ensure_dir(save_html)
            fig.write_html(save_html)
            #print(f"✅ 交互式 3D 图已保存为 {save_html}")
    except ImportError:
        print("❗ Plotly 未安装，无法生成交互式图。")
