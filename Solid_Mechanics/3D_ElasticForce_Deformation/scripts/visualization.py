import numpy as np
import matplotlib.pyplot as plt
from .config import loss_curve_path, plot_displacement_path, plot_stress_path,batch_size
# 物理模型评估
def evaluate_physics_metrics(model):
    """
    从训练时记录的日志中，直接汇总各物理残差的平均值。
    """
    avg_total = np.mean(model.loss_log)
    avg_pde   = np.mean(model.loss_res_log)
    avg_lx    = np.mean(model.loss_x_log)
    avg_ly    = np.mean(model.loss_y_log)
    avg_lz    = np.mean(model.loss_z_log)

    print(" 训练日志物理评估：")
    print(f"  平均总损失   : {avg_total:.3f}")
    print(f"  平均 PDE 残差: {avg_pde:.3f}")
    print(f"  平均边界残差 lx: {avg_lx:.3f}")
    print(f"  平均边界残差 ly: {avg_ly:.3f}")
    print(f"  平均边界残差 lz: {avg_lz:.3f}")


    return {
        'avg_total': avg_total,
        'avg_pde':   avg_pde,
        'avg_lx':    avg_lx,
        'avg_ly':    avg_ly,
        'avg_lz':    avg_lz,
    }

# 损失曲线
def loss_curve(model,filepath=loss_curve_path):
    """
    在同一张图中绘制总损失 (loss) 和各物理残差 (l1, lx, ly, lz) 的收敛曲线
    """

    plt.figure(figsize=(10, 6))
    # Plot all loss curves
    plt.plot(model.loss_log,      linewidth=2, label='Total Loss')
    plt.plot(model.loss_res_log,  linewidth=2, label='PDE Residual (l1)')
    plt.plot(model.loss_x_log,    linewidth=2, label='Boundary Residual lx')
    plt.plot(model.loss_y_log,    linewidth=2, label='Boundary Residual ly')
    plt.plot(model.loss_z_log,    linewidth=2, label='Boundary Residual lz')

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.title('Training Loss and Physics Residuals Over Iterations', fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(filepath, dpi=300)
    plt.show()

# 位移预测云图
def plot_displacement(test_x,model, batch_size=batch_size,save_path=plot_displacement_path):
    """
    对一组三维坐标批量预测并在 1×3 子图中绘制位移云图（u, v, w）
    """
    u, v, w = model.predict(test_x, batch_size=batch_size)
    titles = ['U Displacement', 'V Displacement', 'W Displacement']
    values = [u, v, w]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={'projection': '3d'})
    fig.suptitle('Displacement Prediction', fontsize=20, fontweight='bold', y=1.02)
    for ax, vc, title in zip(axes, values, titles):
        vmax, vmin = np.max(vc), np.min(vc)
        im = ax.scatter(test_x[:,1], test_x[:,0], test_x[:,2],
                        c=vc, cmap='jet', vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('y'); ax.set_ylabel('x'); ax.set_zlabel('z')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()
     # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

#应力预测云图
def plot_stress(test_x, model, batch_size=9261,save_path=plot_stress_path):
    """
    对一组三维坐标批量预测并在 2×3 子图中绘制应力云图
    """
    # predict_stress 返回 (e1,e2,e3,e12,e23,e13,s1,s2,s3,s12,s23,s13,gx,gy,gz)
    *_, s1, s2, s3, s12, s23, s13, _, _, _ = model.predict_stress(test_x, batch_size=batch_size)
    titles = ['S1 Stress', 'S2 Stress', 'S3 Stress', 'S12 Stress', 'S23 Stress', 'S13 Stress']
    values = [s1, s2, s3, s12, s23, s13]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), subplot_kw={'projection': '3d'})
    fig.suptitle('Stress Prediction', fontsize=20, fontweight='bold', y=1.02)
    for ax, vc, title in zip(axes.flat, values, titles):
        vmax, vmin = np.max(vc), np.min(vc)
        im = ax.scatter(test_x[:,1], test_x[:,0], test_x[:,2],
                        c=vc, cmap='jet', vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('y'); ax.set_ylabel('x'); ax.set_zlabel('z')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()
    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    