import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


RESULTS_DIR = "results"


def plot_results(
    hist,
    model,
    xy_obs_list,
    Eobs_re_list,
    Eobs_im_list,
    x_src,
    TRUE_EPS,
    TRUE_R,
    L,
    BETA,
    contrast_chi,  # 这里应该接收来自外部的contrast_chi函数
    device,
):
    """
    简化可视化 - 只保留loss图和三个对比图
    """
    # 设置全局样式
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'figure.dpi': 150
    })
    
    # 创建2x2的图形布局（第一行：loss，第二行：三张对比图）
    fig = plt.figure(figsize=(16, 10))
    
    # 获取模型参数
    eps_pred, r_pred = model.physical_params()
    eps_pred_val = eps_pred.item()
    r_pred_val = r_pred.item()
    
    # ==================== 1. 损失曲线（占据第一行） ====================
    ax1 = plt.subplot(2, 3, (1, 2))  # 占据前两个位置
    
    # 检查历史数据是否存在且非空
    if hist and 'loss' in hist and len(hist['loss']) > 0:
        epochs = np.arange(len(hist['loss']))
        
        ax1.semilogy(epochs, hist['loss'], label='Total Loss', linewidth=2.5, color='#2E86AB')
        
        # 检查并绘制其他损失项
        if 'loss_pde' in hist and len(hist['loss_pde']) > 0:
            ax1.semilogy(epochs[:len(hist['loss_pde'])], hist['loss_pde'], 
                        label='PDE Loss', linewidth=1.8, color='#A23B72', alpha=0.8)
        
        if 'loss_data' in hist and len(hist['loss_data']) > 0:
            ax1.semilogy(epochs[:len(hist['loss_data'])], hist['loss_data'], 
                        label='Data Loss', linewidth=1.8, color='#F18F01', alpha=0.8)
        
        ax1.set_xlabel('Epoch', fontweight='bold')
        ax1.set_ylabel('Loss (log scale)', fontweight='bold')
        ax1.set_title('Training Losses', fontweight='bold')
        ax1.legend(frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # 在loss图上添加最终损失值
        final_loss = hist['loss'][-1] if len(hist['loss']) > 0 else 0
        ax1.text(0.02, 0.98, f'Final Loss: {final_loss:.2e}', 
                 transform=ax1.transAxes, fontsize=9,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        # 如果没有历史数据，显示提示信息
        ax1.text(0.5, 0.5, 'No training history available\n(Loaded pre-trained model)', 
                 transform=ax1.transAxes, ha='center', va='center', fontsize=12, color='gray')
        ax1.set_title('Training Losses', fontweight='bold')
        ax1.set_xlabel('Epoch', fontweight='bold')
        ax1.set_ylabel('Loss (log scale)', fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
    
    # ==================== 2. 参数信息面板（第一行第三个位置） ====================
    ax2 = plt.subplot(2, 3, 3)
    ax2.axis('off')  # 关闭坐标轴，显示文本信息
    
    # 计算误差
    eps_error = abs(eps_pred_val - TRUE_EPS) / TRUE_EPS * 100 if TRUE_EPS != 0 else 0
    r_error = abs(r_pred_val - TRUE_R) / TRUE_R * 100 if TRUE_R != 0 else 0
    
    # 创建信息文本
    info_text = (
        f"Training Results\n"
        f"═══════════════════\n\n"
        f"Target Parameters:\n"
        f"   ε = {TRUE_EPS:.4f}\n"
        f"   r = {TRUE_R:.4f}\n\n"
        f"Recovered Parameters:\n"
        f"   ε = {eps_pred_val:.4f}\n"
        f"   r = {r_pred_val:.4f}\n\n"
        f"Relative Errors:\n"
        f"   ε = {eps_error:.2f}%\n"
        f"   r = {r_error:.2f}%\n\n"
    )
    
    # 如果有历史参数数据，添加初始值
    if hist and 'eps' in hist and 'r' in hist and len(hist['eps']) > 0:
        initial_eps = hist['eps'][0]
        initial_r = hist['r'][0]
        info_text += f"Initial Parameters:\n"
        info_text += f"   ε = {initial_eps:.4f}\n"
        info_text += f"   r = {initial_r:.4f}\n\n"
    
    # 添加性能评估
    if eps_error < 2.0 and r_error < 10.0:
        info_text += "Excellent recovery!"
    elif eps_error < 5.0 and r_error < 15.0:
        info_text += "Good recovery."
    else:
        info_text += "Needs improvement."
    
    # 显示文本
    ax2.text(0.05, 0.95, info_text,
             transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='whitesmoke', 
                      edgecolor='gray', alpha=0.9))
    
    # ==================== 3. 对比度分布图（第二行三个位置） ====================
    # 提高分辨率
    resolution = 300
    x = torch.linspace(-L, L, resolution)
    y = torch.linspace(-L, L, resolution)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    xy = torch.stack([X.flatten(), Y.flatten()], dim=1).to(device)
    
    # 使用传入的contrast_chi函数计算对比度
    chi_pred = contrast_chi(xy, eps_pred, r_pred)
    chi_true = contrast_chi(xy, torch.tensor(TRUE_EPS, device=device), 
                           torch.tensor(TRUE_R, device=device))
    
    chi_pred_img = chi_pred.reshape(resolution, resolution).cpu().detach().numpy()
    chi_true_img = chi_true.reshape(resolution, resolution).cpu().detach().numpy()
    error_img = np.abs(chi_true_img - chi_pred_img)
    
    # 3.1 真实对比度（第二行第一个）
    ax3 = plt.subplot(2, 3, 4)
    im1 = ax3.imshow(chi_true_img, extent=[-L, L, -L, L], origin='lower', 
                    cmap='viridis', vmin=0, vmax=max(TRUE_EPS-1, 0.1), aspect='auto')
    ax3.set_title('True Contrast', fontweight='bold')
    ax3.set_xlabel('x', fontweight='bold')
    ax3.set_ylabel('y', fontweight='bold')
    plt.colorbar(im1, ax=ax3, fraction=0.046, pad=0.04)
    
    # 在图上添加目标信息
    ax3.text(0.02, 0.98, f'Target\nε={TRUE_EPS}\nr={TRUE_R}', 
             transform=ax3.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3.2 预测对比度（第二行第二个）
    ax4 = plt.subplot(2, 3, 5)
    im2 = ax4.imshow(chi_pred_img, extent=[-L, L, -L, L], origin='lower', 
                    cmap='viridis', vmin=0, vmax=max(TRUE_EPS-1, 0.1), aspect='auto')
    ax4.set_title(f'Predicted Contrast', fontweight='bold')
    ax4.set_xlabel('x', fontweight='bold')
    ax4.set_ylabel('y', fontweight='bold')
    plt.colorbar(im2, ax=ax4, fraction=0.046, pad=0.04)
    
    # 在图上添加预测信息
    ax4.text(0.02, 0.98, f'Recovered\nε={eps_pred_val:.4f}\nr={r_pred_val:.4f}', 
             transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3.3 误差分布（第二行第三个）
    ax5 = plt.subplot(2, 3, 6)
    
    # 使用热图但调整颜色范围
    vmax = np.percentile(error_img, 99)  # 使用99%分位数作为最大值
    if vmax == 0:
        vmax = error_img.max() + 1e-10  # 避免除零错误
    
    im3 = ax5.imshow(error_img, extent=[-L, L, -L, L], origin='lower', 
                    cmap='hot_r', vmin=0, vmax=vmax, aspect='auto')
    ax5.set_title(f'Error Distribution', fontweight='bold')
    ax5.set_xlabel('x', fontweight='bold')
    ax5.set_ylabel('y', fontweight='bold')
    cbar = plt.colorbar(im3, ax=ax5, fraction=0.046, pad=0.04)
    cbar.set_label('|χ_true - χ_pred|')
    
    # 显示最大误差
    ax5.text(0.02, 0.98, f'Max Error\n{error_img.max():.2e}', 
             transform=ax5.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 调整布局
    plt.tight_layout()
    
    # 确保结果目录存在
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 保存高质量图像
    summary_path = os.path.join(RESULTS_DIR, "training_summary_simple.png")
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 打印总结
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Target: ε = {TRUE_EPS}, r = {TRUE_R}")
    
    # 如果有历史数据，显示初始参数
    if hist and 'eps' in hist and 'r' in hist and len(hist['eps']) > 0:
        print(f"Initial: ε = {hist['eps'][0]:.4f}, r = {hist['r'][0]:.4f}")
    
    print(f"Final: ε = {eps_pred_val:.4f}, r = {r_pred_val:.4f}")
    print(f"Error: ε = {eps_error:.2f}%, r = {r_error:.2f}%")
    
    # 性能评估
    print("\nPerformance Evaluation:")
    if eps_error < 2.0 and r_error < 10.0:
        print("Excellent recovery!")
    elif eps_error < 5.0 and r_error < 15.0:
        print("Good recovery.")
    else:
        print("Needs improvement.")
    print("="*60)
    
    # 如果有历史数据，保存结果
    if hist and 'eps' in hist:
        np.savez(os.path.join(RESULTS_DIR, 'training_history_final.npz'),
                 eps=hist['eps'] if 'eps' in hist else [],
                 r=hist['r'] if 'r' in hist else [],
                 loss=hist['loss'] if 'loss' in hist else [],
                 loss_pde=hist['loss_pde'] if 'loss_pde' in hist else [],
                 loss_data=hist['loss_data'] if 'loss_data' in hist else [],
                 loss_radius=hist['loss_radius'] if 'loss_radius' in hist else [],
                 eps_pred=eps_pred_val,
                 r_pred=r_pred_val,
                 eps_target=TRUE_EPS,
                 r_target=TRUE_R)
    
        print(f"\nResults saved to '{RESULTS_DIR}' directory")
        print(f"Image saved: {summary_path}")
    else:
        print(f"\nImage saved: {summary_path}")
    
    return eps_pred_val, r_pred_val


def plot_quick_results(model, device, save_dict=None, save_dir=RESULTS_DIR):
    """
    快速模式下的简单可视化
    
    参数:
        model: 加载的模型
        device: 计算设备
        save_dict: 保存的模型字典（包含参数等信息）
        save_dir: 保存目录
    """
    print("\n生成快速测试可视化...")
    
    # 获取模型参数
    eps_pred, r_pred = model.physical_params()
    eps_pred_val = eps_pred.item()
    r_pred_val = r_pred.item()
    
    if save_dict:
        eps_target = save_dict.get('eps_target', 1.5)
        r_target = save_dict.get('r_target', 0.15)
        eps_error = save_dict.get('eps_error', abs(eps_pred_val - eps_target) / eps_target * 100)
        r_error = save_dict.get('r_error', abs(r_pred_val - r_target) / r_target * 100)
        model_name = save_dict.get('timestamp', 'unknown')
    else:
        # 使用默认值
        eps_target = 1.5
        r_target = 0.15
        eps_error = abs(eps_pred_val - eps_target) / eps_target * 100 if eps_target != 0 else 0
        r_error = abs(r_pred_val - r_target) / r_target * 100 if r_target != 0 else 0
        model_name = 'current_model'
    
    # 设置全局样式
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'figure.dpi': 150
    })
    
    # 生成对比度分布图
    resolution = 250
    L = 1.0  # 默认值，可以根据需要调整
    
    x = torch.linspace(-L, L, resolution)
    y = torch.linspace(-L, L, resolution)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    xy = torch.stack([X.flatten(), Y.flatten()], dim=1).to(device)
    
    # 这里需要contrast_chi函数，如果外部没有传入，我们需要从physics导入
    # 注意：这里假设contrast_chi已经在外部定义，或者我们可以从physics导入
    try:
        # 尝试从physics导入contrast_chi
        from src.physics import contrast_chi
    except ImportError:
        # 如果无法导入，创建一个简单的版本
        BETA = 60.0
        def contrast_chi(xy, eps, r):
            rr = torch.sqrt(xy[:, 0:1] ** 2 + xy[:, 1:2] ** 2)
            mask = torch.sigmoid(BETA * (r - rr))
            return (eps - 1.0) * mask
    
    # 计算预测对比度
    chi_pred = contrast_chi(xy, eps_pred, r_pred)
    chi_true = contrast_chi(xy, torch.tensor(eps_target, device=device), 
                           torch.tensor(r_target, device=device))
    
    chi_pred_img = chi_pred.reshape(resolution, resolution).cpu().detach().numpy()
    chi_true_img = chi_true.reshape(resolution, resolution).cpu().detach().numpy()
    error_img = np.abs(chi_true_img - chi_pred_img)
    
    # 创建图形
    fig = plt.figure(figsize=(15, 5))
    
    # 1. 真实对比度
    ax1 = plt.subplot(1, 3, 1)
    im1 = ax1.imshow(chi_true_img, extent=[-L, L, -L, L], origin='lower', 
                     cmap='viridis', vmin=0, vmax=max(eps_target-1, 0.5))
    ax1.set_title(f'真实对比度\nε={eps_target}, r={r_target}', fontweight='bold')
    ax1.set_xlabel('x', fontweight='bold')
    ax1.set_ylabel('y', fontweight='bold')
    plt.colorbar(im1, ax=ax1)
    
    # 在图上添加误差信息
    ax1.text(0.02, 0.98, f'目标值', 
             transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. 预测对比度
    ax2 = plt.subplot(1, 3, 2)
    im2 = ax2.imshow(chi_pred_img, extent=[-L, L, -L, L], origin='lower', 
                     cmap='viridis', vmin=0, vmax=max(eps_target-1, 0.5))
    ax2.set_title(f'预测对比度\nε={eps_pred_val:.4f}, r={r_pred_val:.4f}', fontweight='bold')
    ax2.set_xlabel('x', fontweight='bold')
    ax2.set_ylabel('y', fontweight='bold')
    plt.colorbar(im2, ax=ax2)
    
    # 在图上添加误差信息
    ax2.text(0.02, 0.98, f'误差: ε={eps_error:.1f}%\nr={r_error:.1f}%', 
             transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. 误差分布
    ax3 = plt.subplot(1, 3, 3)
    
    # 使用热图但调整颜色范围
    vmax = np.percentile(error_img, 99)  # 使用99%分位数作为最大值
    if vmax == 0:
        vmax = error_img.max() + 1e-10  # 避免除零错误
    
    im3 = ax3.imshow(error_img, extent=[-L, L, -L, L], origin='lower', 
                     cmap='hot', vmin=0, vmax=vmax)
    ax3.set_title(f'误差分布\n最大: {error_img.max():.2e}', fontweight='bold')
    ax3.set_xlabel('x', fontweight='bold')
    ax3.set_ylabel('y', fontweight='bold')
    cbar = plt.colorbar(im3, ax=ax3)
    cbar.set_label('|χ_true - χ_pred|')
    
    # 在图上添加模型信息
    ax3.text(0.02, 0.98, f'模型: {model_name[:15]}...', 
             transform=ax3.transAxes, fontsize=8,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存图像
    save_path = os.path.join(save_dir, f'quick_test_result_{model_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"可视化结果已保存: {save_path}")
    
    # 打印简要总结
    print("\n" + "="*60)
    print("QUICK TEST SUMMARY")
    print("="*60)
    print(f"目标参数: ε = {eps_target}, r = {r_target}")
    print(f"预测参数: ε = {eps_pred_val:.4f}, r = {r_pred_val:.4f}")
    print(f"相对误差: ε = {eps_error:.2f}%, r = {r_error:.2f}%")
    print("="*60)
    
    return eps_pred_val, r_pred_val
