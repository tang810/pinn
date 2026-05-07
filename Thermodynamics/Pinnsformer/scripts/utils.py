import numpy as np
import torch.nn as nn
import copy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def get_data(x_range, y_range, z_range, t_range, x_num, y_num, z_num, t_num):
    x = np.linspace(x_range[0], x_range[1], x_num)
    y = np.linspace(y_range[0], y_range[1], y_num)
    z = np.linspace(z_range[0], z_range[1], z_num)
    t = np.linspace(t_range[0], t_range[1], t_num)

    x_mesh, y_mesh, z_mesh, t_mesh = np.meshgrid(x, y, z, t)
    data = np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(y_mesh, -1), np.expand_dims(z_mesh, -1),
                           np.expand_dims(t_mesh, -1)), axis=-1)

    # # 使用Matplotlib绘制网格
    # plt.figure(figsize=(1, 1))  # 设置图形大小
    # plt.pcolormesh(x_mesh, t_mesh, x_num, y_num, cmap='viridis')  # 假设我们有一个与网格大小相同的随机数据数组来可视化
    # plt.colorbar()  # 显示颜色条
    # plt.xlabel('X axis')
    # plt.ylabel('Y axis')
    # plt.title('Grid Visualization')
    # plt.grid(True)  # 显示网格线（可选，因为pcolormesh已经隐含了网格）
    # plt.show()

    x_bottom = data[:, 0, :, :].reshape(-1, 4)
    x_top = data[:, -1, :, :].reshape(-1, 4)
    y_bottom = data[0, :, :, :].reshape(-1, 4)
    y_top = data[-1, :, :, :].reshape(-1, 4)
    z_bottom = data[:, :, 0, :].reshape(-1, 4)
    z_top = data[:, :, -1, :].reshape(-1, 4)
    t_bottom = data[:, :, :, 0].reshape(-1, 4)
    t_top = data[:, :, :, -1].reshape(-1, 4)
    res = data.reshape(-1, 4)


    return res, x_bottom, x_top, y_bottom, y_top, z_bottom, z_top, t_bottom, t_top


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def make_time_sequence(src, num_step=5, step=1e-4):
    dim = num_step
    src = np.repeat(np.expand_dims(src, axis=1), dim, axis=1)  # (N, L, 2)
    for i in range(num_step):
        src[:, i, -1] += step * i
    return src


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_data_3d(x_range, y_range, t_range, x_num, y_num, t_num):
    step_x = (x_range[1] - x_range[0]) / float(x_num - 1)
    step_y = (y_range[1] - y_range[0]) / float(y_num - 1)
    step_t = (t_range[1] - t_range[0]) / float(t_num - 1)

    x_mesh, y_mesh, t_mesh = np.mgrid[x_range[0]:x_range[1] + step_x:step_x, y_range[0]:y_range[1] + step_y:step_y,
                             t_range[0]:t_range[1] + step_t:step_t]

    data = np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(y_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1)
    res = data.reshape(-1, 3)

    x_mesh, y_mesh, t_mesh = np.mgrid[x_range[0]:x_range[0] + step_x:step_x, y_range[0]:y_range[1] + step_y:step_y,
                             t_range[0]:t_range[1] + step_t:step_t]
    b_left = np.squeeze(
        np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(y_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1))[
             1:-1].reshape(-1, 3)

    x_mesh, y_mesh, t_mesh = np.mgrid[x_range[1]:x_range[1] + step_x:step_x, y_range[0]:y_range[1] + step_y:step_y,
                             t_range[0]:t_range[1] + step_t:step_t]
    b_right = np.squeeze(
        np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(y_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1))[
              1:-1].reshape(-1, 3)

    x_mesh, y_mesh, t_mesh = np.mgrid[x_range[0]:x_range[1] + step_x:step_x, y_range[0]:y_range[0] + step_y:step_y,
                             t_range[0]:t_range[1] + step_t:step_t]
    b_lower = np.squeeze(
        np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(y_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1))[
              1:-1].reshape(-1, 3)

    x_mesh, y_mesh, t_mesh = np.mgrid[x_range[0]:x_range[1] + step_x:step_x, y_range[1]:y_range[1] + step_y:step_y,
                             t_range[0]:t_range[1] + step_t:step_t]
    b_upper = np.squeeze(
        np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(y_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1))[
              1:-1].reshape(-1, 3)

    return res, b_left, b_right, b_upper, b_lower

def loading_evaluate_data(filename):
    data = pd.read_csv(filename, delimiter=' ', header=None)
    x = data.iloc[:, 0].to_numpy()
    y = data.iloc[:, 1].to_numpy()
    z = data.iloc[:, 2].to_numpy()
    t = data.iloc[:, 3].to_numpy()
    T = data.iloc[:, 4].to_numpy()

    # X_data = torch.tensor(X_data, dtype=torch.float32)
    # U_data = torch.tensor(U_data, dtype=torch.float32)
    # U_data = (U_data - U_data.min()) / (U_data.max() - U_data.min())

    # data = scipy.io.loadmat(filename)
    # x = data['X_ref']
    # y = data['Y_ref']
    # u = data['U_ref']
    # v = data['V_ref']
    x_star = x.reshape(-1, 1)
    y_star = y.reshape(-1, 1)
    z_star = z.reshape(-1, 1)
    t_star = t.reshape(-1, 1)
    T_star = T.reshape(-1, 1)
    # u_star = u.reshape(-1,1)
    # v_star = v.reshape(-1,1)
    return x_star, y_star, z_star, t_star, T_star

def plot_temperature_animation(x, y, z, t, T_true, x_star, y_star, z_star, T_pred, save_path='./viz/animation.gif'):
    """
    可视化预测与真实温度的3D动画对比图
    """
    t = t.flatten()
    times = np.unique(t)
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    scat1 = scat2 = None

    def update(frame):
        current_time = times[frame]
        mask = np.isclose(t, current_time, atol=1e-6)
        nonlocal scat1, scat2

        T_true_frame = T_true[mask].reshape(-1)
        T_pred_frame = T_pred[mask].reshape(-1)

        if scat1 is None:
            scat1 = ax1.scatter(x[mask], y[mask], z[mask], c=T_true_frame, cmap='viridis', s=60,
                                vmin=T_true.min(), vmax=T_true.max())
            fig.colorbar(scat1, ax=ax1, shrink=0.4, aspect=10, label='Temperature (true)')
        else:
            scat1._offsets3d = (x[mask], y[mask], z[mask])
            scat1.set_array(T_true_frame)

        if scat2 is None:
            scat2 = ax2.scatter(x_star[mask], y_star[mask], z_star[mask], c=T_pred_frame, cmap='viridis', s=60,
                                vmin=T_pred.min(), vmax=T_pred.max())
            fig.colorbar(scat2, ax=ax2, shrink=0.4, aspect=10, label='Temperature (pred)', location='right')
        else:
            scat2._offsets3d = (x_star[mask], y_star[mask], z_star[mask])
            scat2.set_array(T_pred_frame)


    anim = FuncAnimation(fig, update, frames=len(times), interval=100, blit=False)
    anim.save(save_path, writer=PillowWriter(fps=30))
    plt.close()


def plot_loss_curve(loss_track, save_path='./viz/loss_curve.png'):
    """
    绘制 loss 曲线图：PDE、BC、IC、Data loss 和 Total loss。
    参数:
        loss_track: list，每项为 [loss_pde, loss_bc, loss_ic, loss_data]
    """
    loss_track = np.array(loss_track)
    steps = np.arange(1, len(loss_track) + 1)

    loss_pde = loss_track[:, 0]
    loss_bc = loss_track[:, 1]
    loss_ic = loss_track[:, 2]
    loss_data = loss_track[:, 3]
    loss_total = loss_pde + loss_bc + loss_ic + 10 * loss_data

    plt.figure(figsize=(10, 6))
    plt.plot(steps, loss_pde, label='PDE Loss')
    plt.plot(steps, loss_bc, label='BC Loss')
    plt.plot(steps, loss_ic, label='IC Loss')
    plt.plot(steps, loss_data, label='Data Loss')
    plt.plot(steps, loss_total, label='Total Loss', linewidth=2, linestyle='--')

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_error_animation(x, y, z, t, T_true, T_pred, save_path='./viz/error_animation.gif'):
    x = np.array(x).reshape(-1)
    y = np.array(y).reshape(-1)
    z = np.array(z).reshape(-1)
    t = np.array(t).reshape(-1)
    T_true = np.array(T_true).reshape(-1)
    T_pred = np.array(T_pred).reshape(-1)

    times = np.unique(t)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    scat = None

    def update(frame):
        nonlocal scat
        t_val = times[frame]
        mask = np.isclose(t, t_val, atol=1e-6)
        error = (T_true[mask] - T_pred[mask]).reshape(-1)

        if scat is None:
            scat = ax.scatter(x[mask], y[mask], z[mask], c=error, cmap='coolwarm', s=60,
                              vmin=-np.max(np.abs(error)), vmax=np.max(np.abs(error)))
            fig.colorbar(scat, ax=ax, shrink=0.6, label='Error (T_true - T_pred)')
        else:
            scat._offsets3d = (x[mask], y[mask], z[mask])
            scat.set_array(error)

        ax.set_title(f"Error at t = {t_val:.2f}s")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    anim = FuncAnimation(fig, update, frames=len(times), interval=200, blit=False)
    anim.save(save_path, writer=PillowWriter(fps=5))
    plt.close()