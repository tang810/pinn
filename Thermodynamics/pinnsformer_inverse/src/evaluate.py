import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy import stats
from . import data_loader
from .model import PINNsformer
import os

# =====================================================
# ============ 自定义颜色映射 =========================
# =====================================================
from matplotlib.colors import LinearSegmentedColormap

yellow_purple_cmap = LinearSegmentedColormap.from_list(
    'yellow_purple', ['yellow', 'purple'], N=256
)
blue_purple_cmap = LinearSegmentedColormap.from_list(
    'blue_purple', ['blue', 'purple'], N=256
)

def plot_k_comparison(true_k, inferred_k, save_path):
    """Plot a balanced comparison chart that highlights closeness without visual exaggeration."""
    delta = inferred_k - true_k
    rel_err = abs(delta) / (abs(true_k) + 1e-12) * 100.0
    agreement = max(0.0, 100.0 - rel_err)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

    # Left: absolute k values with full-scale baseline (avoids exaggeration).
    labels = ['True k', 'Inferred k']
    values = [true_k, inferred_k]
    bars = ax1.bar(labels, values, color=['#1f77b4', '#2ca02c'], width=0.55, edgecolor='black', linewidth=1.2)
    ax1.set_title('Absolute Value Comparison')
    ax1.set_ylabel('k')
    ax1.set_ylim(0.0, max(values) * 1.25)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    for bar, v in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{v:.6f}', ha='center', va='bottom')

    # Right: relative error against tolerance bands.
    ax2.axhspan(0, 5, color='#d9f2d9', alpha=0.8, label='Excellent (<=5%)')
    ax2.axhspan(5, 10, color='#fff4cc', alpha=0.8, label='Good (5%-10%)')
    ax2.axhspan(10, 20, color='#fde2e2', alpha=0.8, label='Fair (10%-20%)')
    ax2.bar(['Relative Error'], [rel_err], color='#ff7f0e', width=0.45, edgecolor='black', linewidth=1.2)
    ax2.set_title('Relative Error View')
    ax2.set_ylabel('Error (%)')
    ax2.set_ylim(0, max(20.0, rel_err * 1.8))
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    ax2.text(
        0.0,
        rel_err + max(0.4, rel_err * 0.06),
        f'{rel_err:.2f}%\nAgreement: {agreement:.2f}%',
        ha='center',
        va='bottom',
        fontweight='bold'
    )
    ax2.legend(loc='upper right', fontsize=8, framealpha=0.95)

    fig.suptitle('True vs Final Inferred k', fontsize=14, fontweight='bold')
    fig.text(0.5, 0.01, f'Abs error = {abs(delta):.6f} | Delta k = {delta:+.6f}', ha='center')
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.savefig(save_path, dpi=160)
    plt.close(fig)

# =====================================================
# ============ 分批推理函数（用于纯预测）===============
# =====================================================
def model_forward_in_batches(model, x, y, z, t, batch_size=4096, return_tensor=False):
    """
    分批前向推理，避免显存爆炸（不计算梯度）
    """
    n_points = x.shape[0]
    out_list = []

    for i in range(0, n_points, batch_size):
        xb = x[i:i+batch_size].to(x.device)
        yb = y[i:i+batch_size].to(y.device)
        zb = z[i:i+batch_size].to(z.device)
        tb = t[i:i+batch_size].to(t.device)
        with torch.no_grad():
            out_batch = model(xb, yb, zb, tb)
        out_list.append(out_batch if return_tensor else out_batch.cpu().numpy())

    if return_tensor:
        return torch.cat(out_list, dim=0)
    else:
        return np.concatenate(out_list, axis=0)

# =====================================================
# ============ 从位移和温度计算完整应力（分批）========
# =====================================================
def compute_stress_from_model(model, x, y, z, t, E, nu, alpha, T_ref, batch_size=2000):
    """
    利用模型输出和自动微分计算应力张量，分批处理避免显存爆炸。
    返回所有应力分量及 von Mises 的 numpy 数组（扁平化）。
    """
    device = x.device
    n_points = x.shape[0]

    # 预先分配结果列表
    sigma_xx_list, sigma_yy_list, sigma_zz_list = [], [], []
    sigma_xy_list, sigma_yz_list, sigma_xz_list = [], [], []
    sigma_vm_list = []

    for i in range(0, n_points, batch_size):
        # 取出当前批次，并设置 requires_grad
        xb = x[i:i+batch_size].clone().detach().requires_grad_(True).to(device)
        yb = y[i:i+batch_size].clone().detach().requires_grad_(True).to(device)
        zb = z[i:i+batch_size].clone().detach().requires_grad_(True).to(device)
        tb = t[i:i+batch_size].clone().detach().requires_grad_(True).to(device)

        # 前向传播
        out = model(xb, yb, zb, tb)
        T = out[:, 0:1]
        u = out[:, 1:2]
        v = out[:, 2:3]
        w = out[:, 3:4]

        # 位移梯度（一阶导数）
        u_x = torch.autograd.grad(u, xb, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, yb, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_z = torch.autograd.grad(u, zb, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        v_x = torch.autograd.grad(v, xb, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_y = torch.autograd.grad(v, yb, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_z = torch.autograd.grad(v, zb, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        w_x = torch.autograd.grad(w, xb, grad_outputs=torch.ones_like(w), create_graph=True)[0]
        w_y = torch.autograd.grad(w, yb, grad_outputs=torch.ones_like(w), create_graph=True)[0]
        w_z = torch.autograd.grad(w, zb, grad_outputs=torch.ones_like(w), create_graph=True)[0]

        # 应变（小变形）
        eps_xx = u_x
        eps_yy = v_y
        eps_zz = w_z
        eps_xy = 0.5 * (u_y + v_x)
        eps_xz = 0.5 * (u_z + w_x)
        eps_yz = 0.5 * (v_z + w_y)

        # 热应变
        eps_th = alpha * (T - T_ref)

        # 拉梅常数
        mu = E / (2.0 * (1.0 + nu))
        lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        # 应力
        sigma_xx = 2*mu*eps_xx + lmbda*(eps_xx+eps_yy+eps_zz) - (3*lmbda+2*mu)*eps_th
        sigma_yy = 2*mu*eps_yy + lmbda*(eps_xx+eps_yy+eps_zz) - (3*lmbda+2*mu)*eps_th
        sigma_zz = 2*mu*eps_zz + lmbda*(eps_xx+eps_yy+eps_zz) - (3*lmbda+2*mu)*eps_th
        sigma_xy = 2*mu * eps_xy
        sigma_xz = 2*mu * eps_xz
        sigma_yz = 2*mu * eps_yz

        sigma_vm = torch.sqrt(
            0.5*((sigma_xx - sigma_yy)**2 + (sigma_yy - sigma_zz)**2 + (sigma_zz - sigma_xx)**2
                 + 6*(sigma_xy**2 + sigma_xz**2 + sigma_yz**2))
        )

        # 分离结果并转移到CPU
        sigma_xx_list.append(sigma_xx.detach().cpu().numpy().flatten())
        sigma_yy_list.append(sigma_yy.detach().cpu().numpy().flatten())
        sigma_zz_list.append(sigma_zz.detach().cpu().numpy().flatten())
        sigma_xy_list.append(sigma_xy.detach().cpu().numpy().flatten())
        sigma_yz_list.append(sigma_yz.detach().cpu().numpy().flatten())
        sigma_xz_list.append(sigma_xz.detach().cpu().numpy().flatten())
        sigma_vm_list.append(sigma_vm.detach().cpu().numpy().flatten())

        # 清理中间变量以释放显存
        del xb, yb, zb, tb, out, T, u, v, w
        del u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y, w_z
        del eps_xx, eps_yy, eps_zz, eps_xy, eps_xz, eps_yz, eps_th
        del sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_xz, sigma_yz, sigma_vm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 拼接所有批次的结果
    sigma_xx = np.concatenate(sigma_xx_list)
    sigma_yy = np.concatenate(sigma_yy_list)
    sigma_zz = np.concatenate(sigma_zz_list)
    sigma_xy = np.concatenate(sigma_xy_list)
    sigma_yz = np.concatenate(sigma_yz_list)
    sigma_xz = np.concatenate(sigma_xz_list)
    sigma_vm = np.concatenate(sigma_vm_list)

    return sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_xz, sigma_vm

# =====================================================
# ============ 温度动画 ==============================
# =====================================================
def _create_animation(data_path, T_pred, save_path):
    data = pd.read_csv(data_path, delimiter=' ', header=None)
    x, y, z, t, T_true = (data.iloc[:, i] for i in range(5))
    unique_times = np.unique(t)

    fig = plt.figure(figsize=(16,7))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    vmin, vmax = min(T_true.min(), T_pred.min()), max(T_true.max(), T_pred.max())
    scat1 = ax1.scatter([], [], [], c=[], cmap='viridis', vmin=vmin, vmax=vmax, s=60)
    scat2 = ax2.scatter([], [], [], c=[], cmap='viridis', vmin=vmin, vmax=vmax, s=60)
    fig.colorbar(scat1, ax=ax1, shrink=0.6, label='Temperature (K)')
    fig.colorbar(scat2, ax=ax2, shrink=0.6, label='Temperature (K)')

    def update(frame):
        time_point = unique_times[frame]
        mask = np.isclose(t, time_point)
        scat1._offsets3d = (x[mask], y[mask], z[mask])
        scat1.set_array(T_true[mask])
        ax1.set_title(f'Ground Truth, t={time_point:.2f}')
        scat2._offsets3d = (x[mask], y[mask], z[mask])
        scat2.set_array(T_pred[mask])
        ax2.set_title(f'PINNs Prediction, t={time_point:.2f}')
        return scat1, scat2

    for ax in [ax1, ax2]:
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_zlim(0,1)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    anim = FuncAnimation(fig, update, frames=len(unique_times), interval=100)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    anim.save(save_path, writer=PillowWriter(fps=10))
    plt.close(fig)

# =====================================================
# ============ 应力动画 ==============================
# =====================================================
def _create_stress_animation(data_path, sigma_vm, save_path):
    data = pd.read_csv(data_path, delimiter=' ', header=None)
    x, y, z, t, _ = (data.iloc[:, i] for i in range(5))
    unique_times = np.unique(t)

    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(111, projection='3d')
    vmin, vmax = sigma_vm.min(), sigma_vm.max()
    scat = ax.scatter([], [], [], c=[], cmap='inferno', vmin=vmin, vmax=vmax, s=60)
    fig.colorbar(scat, ax=ax, shrink=0.6, label='von Mises Stress (Pa)')

    def update(frame):
        time_point = unique_times[frame]
        mask = np.isclose(t, time_point)
        scat._offsets3d = (x[mask], y[mask], z[mask])
        scat.set_array(sigma_vm[mask])
        ax.set_title(f'Thermal Stress, t={time_point:.2f}')
        return scat,

    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_zlim(0,1)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    anim = FuncAnimation(fig, update, frames=len(unique_times), interval=100)
    anim.save(save_path, writer=PillowWriter(fps=10))
    plt.close(fig)

# =====================================================
# ============ 温度-应力二维图 ========================
# =====================================================
def plot_temperature_stress_relation(T, sigma_vm, save_path):
    plt.figure(figsize=(8,6))
    plt.scatter(T, sigma_vm, s=5, alpha=0.3)
    plt.xlabel("Temperature (K)")
    plt.ylabel("von Mises Stress (Pa)")
    plt.title("Temperature–Stress Relationship (Physics-driven)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# =====================================================
# ============ 误差分析 ===============================
# =====================================================
def compute_error_metrics(T_true, T_pred):
    T_true = np.array(T_true).flatten()
    T_pred = np.array(T_pred).flatten()
    abs_error = np.abs(T_true - T_pred)
    rel_error = abs_error / (np.abs(T_true)+1e-8)
    metrics = {
        'MAE': np.mean(abs_error),
        'RMSE': np.sqrt(np.mean((T_true - T_pred)**2)),
        'MaxAE': np.max(abs_error),
        'MAPE': np.mean(rel_error)*100,
        'R2': 1 - np.sum((T_true - T_pred)**2) / (np.sum((T_true - np.mean(T_true))**2)+1e-8),
        'Relative L2': np.linalg.norm(T_true - T_pred)/np.linalg.norm(T_true)
    }
    return metrics, abs_error, rel_error

def plot_prediction_vs_truth(T_true, T_pred, save_path):
    plt.figure(figsize=(10,8))
    plt.scatter(T_true, T_pred, s=10, alpha=0.5, label='Predictions')
    min_val = min(T_true.min(), T_pred.min())
    max_val = max(T_true.max(), T_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    plt.xlabel('Ground Truth Temperature (K)')
    plt.ylabel('Predicted Temperature (K)')
    plt.title('Prediction vs Ground Truth')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_error_distribution(abs_error, rel_error, save_path):
    fig, axes = plt.subplots(2,2,figsize=(12,10))
    axes[0,0].hist(abs_error,bins=50,alpha=0.7,color='blue',edgecolor='black')
    axes[0,0].set_xlabel('Absolute Error (K)'); axes[0,0].set_ylabel('Frequency'); axes[0,0].set_title('Absolute Error Distribution'); axes[0,0].grid(True,alpha=0.3)
    axes[0,1].hist(rel_error*100,bins=50,alpha=0.7,color='green',edgecolor='black')
    axes[0,1].set_xlabel('Relative Error (%)'); axes[0,1].set_ylabel('Frequency'); axes[0,1].set_title('Relative Error Distribution'); axes[0,1].grid(True,alpha=0.3)
    axes[1,0].scatter(np.arange(len(abs_error)),abs_error,s=5,alpha=0.5,color='red')
    axes[1,0].set_xlabel('Data Point Index'); axes[1,0].set_ylabel('Absolute Error (K)'); axes[1,0].set_title('Absolute Error vs Data Point'); axes[1,0].grid(True,alpha=0.3)
    axes[1,1].scatter(abs_error,np.arange(len(abs_error)),s=5,alpha=0.5,color='purple')
    axes[1,1].set_xlabel('Absolute Error (K)'); axes[1,1].set_ylabel('Data Point Index'); axes[1,1].set_title('Sorted Absolute Errors'); axes[1,1].grid(True,alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path,dpi=150)
    plt.close()

def print_error_metrics(metrics):
    print("\n"+"="*60)
    print("TEMPERATURE PREDICTION ERROR ANALYSIS")
    print("="*60)
    print(f"Mean Absolute Error (MAE):      {metrics['MAE']:.6e} K")
    print(f"Root Mean Square Error (RMSE):  {metrics['RMSE']:.6e} K")
    print(f"Max Absolute Error (MaxAE):     {metrics['MaxAE']:.6e} K")
    print(f"Mean Absolute Percentage Error: {metrics['MAPE']:.4f} %")
    print(f"R2 Score:                       {metrics['R2']:.6f}")
    print(f"Relative L2 Error:              {metrics['Relative L2']:.6e}")
    print("="*60)

# =====================================================
# ============ 3D模型生成函数 =========================
# =====================================================
def create_3d_models(x, y, z, temperatures, stresses, output_dir, args):
    try:
        import trimesh
        from scipy.spatial import ConvexHull
        from matplotlib.colors import PowerNorm
    except ImportError as e:
        print(f"无法导入trimesh或相关库: {e}, 跳过3D模型生成")
        return

    model_dir = os.path.join(output_dir, "3d_models")
    os.makedirs(model_dir, exist_ok=True)

    points = np.column_stack([x,y,z])
    temps = temperatures
    stresses_raw = stresses

    # 采样比例
    sample_ratio = 1.0
    n_sample = int(len(points)*sample_ratio)
    indices = np.random.choice(len(points), n_sample, replace=False) if n_sample<len(points) else np.arange(len(points))
    points_used = points[indices]
    temps_used = temps[indices]
    stresses_used = stresses_raw[indices]

    # 凸包网格
    try:
        n_hull = min(5000,len(points))
        idx_hull = np.random.choice(len(points),n_hull,replace=False)
        points_hull = points[idx_hull]
        temps_hull = temps[idx_hull]
        stresses_hull = stresses_raw[idx_hull]

        hull = ConvexHull(points_hull)
        mesh_temp = trimesh.Trimesh(vertices=points_hull,faces=hull.simplices,process=True)
        norm_temp = PowerNorm(gamma=1.8,vmin=temps.min(),vmax=temps.max())
        colors_temp = yellow_purple_cmap(norm_temp(temps_hull))
        colors_temp[:,3]=1.0
        mesh_temp.visual.vertex_colors = (colors_temp*255).astype(np.uint8)
        mesh_temp.export(os.path.join(model_dir,"temperature_convex_hull.glb"))

        mesh_stress = trimesh.Trimesh(vertices=points_hull,faces=hull.simplices,process=True)
        norm_stress = PowerNorm(gamma=1.8,vmin=stresses_raw.min(),vmax=stresses_raw.max())
        colors_stress = blue_purple_cmap(norm_stress(stresses_hull))
        colors_stress[:,3]=1.0
        mesh_stress.visual.vertex_colors = (colors_stress*255).astype(np.uint8)
        mesh_stress.export(os.path.join(model_dir,"stress_convex_hull.glb"))
    except Exception as e:
        print(f"凸包生成失败: {e}")

    # 点云导出
    norm_temp_cloud = PowerNorm(gamma=1.8,vmin=temps.min(),vmax=temps.max())
    colors_temp = yellow_purple_cmap(norm_temp_cloud(temps_used))
    colors_temp[:,3]=1.0
    temp_cloud = trimesh.PointCloud(vertices=points_used)
    temp_cloud.visual = trimesh.visual.ColorVisuals(vertex_colors=(colors_temp*255).astype(np.uint8))
    temp_cloud.export(os.path.join(model_dir,"temperature_point_cloud.glb"),file_type='glb')

    norm_stress_cloud = PowerNorm(gamma=1.8,vmin=stresses_raw.min(),vmax=stresses_raw.max())
    colors_stress = blue_purple_cmap(norm_stress_cloud(stresses_used))
    colors_stress[:,3]=1.0
    stress_cloud = trimesh.PointCloud(vertices=points_used)
    stress_cloud.visual = trimesh.visual.ColorVisuals(vertex_colors=(colors_stress*255).astype(np.uint8))
    stress_cloud.export(os.path.join(model_dir,"stress_point_cloud.glb"),file_type='glb')

    # 保存 npz
    np.savez(os.path.join(model_dir,"model_data.npz"),points=points_used,temperatures=temps_used,stresses=stresses_used)

# =====================================================
# ============ 主流程 ================================
# =====================================================
def run(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    true_k = args.k
    inferred_k = None

    # 加载评估数据
    x_star, y_star, z_star, t_star, T_star_true = data_loader.get_evaluation_data(args)

    # 将数据移至设备，不需要 requires_grad（将在应力计算中重新设置）
    x_tensor = x_star.clone().detach().to(device)
    y_tensor = y_star.clone().detach().to(device)
    z_tensor = z_star.clone().detach().to(device)
    t_tensor = t_star.clone().detach().to(device)

    # 真值温度转为 numpy 一维数组
    if isinstance(T_star_true, torch.Tensor):
        T_star_true = T_star_true.detach().cpu().numpy().flatten()
    else:
        T_star_true = np.array(T_star_true).flatten()

    # 加载模型
    model = PINNsformer(d_model=args.d_model,d_hidden=args.d_hidden,N=args.n_layers,heads=args.n_heads).to(device)
    checkpoint = torch.load(args.model_save_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # 如果是反演模式且 checkpoint 中包含 loss_computer_state_dict，可加载反演参数（可选）
    inverse_mode = getattr(args, 'inverse_mode', False)
    if inverse_mode and 'loss_computer_state_dict' in checkpoint:
        from .physics import ThermoElasticPINNLoss
        infer_params = getattr(args, 'infer_params', [])
        loss_computer = ThermoElasticPINNLoss(args, trainable_params=infer_params).to(device)
        loss_computer.load_state_dict(checkpoint['loss_computer_state_dict'])
        print("\nLoaded inversion parameters:")
        for name in infer_params:
            val = getattr(loss_computer, name).item()
            print(f"  {name} = {val:.6f}")
            if name == 'k':
                inferred_k = val
        # 可选择性更新 args 中的参数以便后续应力计算使用（可选）
        for name in infer_params:
            if name != 'k':
                setattr(args, name, getattr(loss_computer, name).item())

    # 前向预测温度（无梯度）
    out_np = model_forward_in_batches(model, x_tensor, y_tensor, z_tensor, t_tensor, batch_size=4096)
    T_pred = out_np[:, 0].flatten()
    # 位移预测（可选）
    u_pred = out_np[:, 1]
    v_pred = out_np[:, 2]
    w_pred = out_np[:, 3]

    # 温度误差分析
    metrics, abs_error, rel_error = compute_error_metrics(T_star_true, T_pred)
    print_error_metrics(metrics)

    # 计算应力（分批自动微分）
    sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_xz, sigma_vm = \
        compute_stress_from_model(model, x_tensor, y_tensor, z_tensor, t_tensor,
                                  args.e, args.nu, args.alpha_t, args.t_ref,
                                  batch_size=2000)  # 可根据显存调整

    results_dir = os.path.dirname(args.animation_save_path)
    os.makedirs(results_dir, exist_ok=True)

    # 保存误差报告、动画、图像等
    error_file = os.path.join(results_dir, "error_analysis.txt")
    with open(error_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("PINNsformer Temperature Prediction Error Analysis\n")
        f.write("="*60 + "\n\n")
        for key, value in metrics.items():
            if key in ['MAE', 'RMSE', 'MaxAE', 'Relative L2']:
                f.write(f"{key:25s}: {value:.6e}\n")
            elif key == 'MAPE':
                f.write(f"{key:25s}: {value:.4f} %\n")
            else:
                f.write(f"{key:25s}: {value:.6f}\n")
        f.write("\n" + "-"*60 + "\n")
        f.write("Additional Statistics\n")
        f.write("-"*60 + "\n")
        f.write(f"Number of test points: {len(T_star_true)}\n")
        f.write(f"True temperature range: [{T_star_true.min():.2f}, {T_star_true.max():.2f}] K\n")
        f.write(f"Predicted temperature range: [{T_pred.min():.2f}, {T_pred.max():.2f}] K\n")
    print(f"[OK] Error analysis saved to: {error_file}")

    # 温度 GIF
    _create_animation(args.data_path, T_pred, args.animation_save_path)
    print(f"[OK] Temperature GIF saved to: {args.animation_save_path}")

    # 应力 GIF
    stress_gif = args.animation_save_path.replace(".gif", "_stress.gif")
    _create_stress_animation(args.data_path, sigma_vm, stress_gif)
    print(f"[OK] Thermal stress GIF saved to: {stress_gif}")

    # 温度-应力关系图
    temp_stress_plot = os.path.join(results_dir, "temperature_vs_stress.png")
    plot_temperature_stress_relation(T_pred, sigma_vm, temp_stress_plot)
    print(f"[OK] Temperature-Stress relation plot saved to: {temp_stress_plot}")

    # 预测 vs 真值图
    pred_vs_truth_plot = os.path.join(results_dir, "prediction_vs_truth.png")
    plot_prediction_vs_truth(T_star_true, T_pred, pred_vs_truth_plot)
    print(f"[OK] Prediction vs Truth plot saved to: {pred_vs_truth_plot}")

    # 误差分布图
    error_dist_plot = os.path.join(results_dir, "error_distribution.png")
    plot_error_distribution(abs_error, rel_error, error_dist_plot)
    print(f"[OK] Error distribution plot saved to: {error_dist_plot}")

    # 3D模型生成
    if inferred_k is not None:
        k_compare_plot = os.path.join(results_dir, "k_true_vs_inferred.png")
        plot_k_comparison(true_k, inferred_k, k_compare_plot)
        print(f"k comparison plot saved to: {k_compare_plot} (true={true_k:.6f}, inferred={inferred_k:.6f})")

    x_np = x_star.cpu().numpy().flatten()
    y_np = y_star.cpu().numpy().flatten()
    z_np = z_star.cpu().numpy().flatten()
    create_3d_models(x_np, y_np, z_np, T_pred, sigma_vm, results_dir, args)

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
