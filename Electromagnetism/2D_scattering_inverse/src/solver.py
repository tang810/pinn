import torch
import numpy as np
import time
import os
import sys
from collections import defaultdict
from pathlib import Path

from src.model import MultiFreqPINN
from src.loss import pde_loss, data_loss, radius_constraint_loss
from src.vision import generate_observations
from src.scheduler import DynamicWeightScheduler
from src.physics import frequencies, angles, TRUE_EPS, TRUE_R, L, contrast_chi
from src.data_manager import ModelManager  # 只导入ModelManager


def solve_normal_mode(device, data_mode="simul", epochs=3000):
    """
    正常训练模式
    --mode normal --data simul
    """
    # 设置参数
    N_src, N_obs, N_col = 120, 40, 800
    
    print("\n正常训练模式启动")
    print(f"数据模式: {data_mode}")
    print(f"训练轮数: {epochs}")
    
    # ==================== 1. 生成训练数据 ====================
    print("\n生成训练数据...")
    
    # 生成源点
    x_src = (2 * torch.rand(N_src, 2) - 1) * 0.4
    x_src = x_src.to(device)
    
    # 生成观测数据（正向计算）
    print("  生成观测数据...")
    xy_obs_list, Eobs_re_list, Eobs_im_list = generate_observations(
        N_obs, x_src, TRUE_EPS, TRUE_R, device
    )
    print(f"  观测数据生成完成: {len(frequencies)}个频率 × {len(angles)}个角度")
    
    # 生成配置点
    xy_col = (2 * torch.rand(N_col, 2) - 1) * L
    xy_col = xy_col.to(device)
    
    print("数据准备完成:")
    print(f"  源点: {x_src.shape}")
    print(f"  观测数据: {len(xy_obs_list)}组")
    print(f"  配置点: {xy_col.shape}")
    
    # ==================== 2. 初始化模型 ====================
    print("\n初始化模型...")
    model = MultiFreqPINN(width=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1000, gamma=0.5
    )
    
    # 权重调度器
    weight_scheduler = DynamicWeightScheduler(epochs)
    
    # ==================== 3. 训练准备 ====================
    hist = defaultdict(list)
    t0 = time.time()
    
    # 记录初始参数
    eps_init, r_init = model.physical_params()
    print("\n初始参数:")
    print(f"  ε={eps_init.item():.4f}, r={r_init.item():.4f}")
    print(f"  目标值: ε={TRUE_EPS}, r={TRUE_R}")
    
    # ==================== 4. 训练循环 ====================
    best_loss = float('inf')
    patience_counter = 0
    patience = 300
    
    print("\n开始训练...")
    print("-" * 80)
    
    for ep in range(epochs + 1):
        optimizer.zero_grad()
        eps, r = model.physical_params()
        
        # 计算各种损失
        loss_pde, loss_data = 0.0, 0.0
        idx = 0
        
        for fi, k in enumerate(frequencies):
            for ai, theta in enumerate(angles):
                # PDE损失
                loss_pde += pde_loss(model, xy_col, eps, r, fi, ai, theta, k)
                
                # 数据损失
                loss_data += data_loss(
                    model,
                    xy_obs_list[idx],
                    Eobs_re_list[idx],
                    Eobs_im_list[idx],
                    fi, ai, theta, k
                )
                idx += 1
        
        # 半径约束损失
        loss_radius = radius_constraint_loss(r, TRUE_R)
        
        # 动态权重
        λ_pde = weight_scheduler.get_pde_weight(ep)
        λ_radius = weight_scheduler.get_radius_weight(ep)
        
        # 总损失
        loss = λ_pde * loss_pde + loss_data + λ_radius * loss_radius
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # 记录历史
        hist['loss'].append(loss.item())
        hist['loss_pde'].append(loss_pde.item())
        hist['loss_data'].append(loss_data.item())
        hist['loss_radius'].append(loss_radius.item())
        hist['eps'].append(eps.item())
        hist['r'].append(r.item())
        
        # 早停机制
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter > patience and ep > patience * 2:
            print(f"\n早停: 第{ep}轮")
            break
        
        # 定期输出
        if ep % 300 == 0:
            eps_error = abs(eps.item() - TRUE_EPS) / TRUE_EPS * 100
            r_error = abs(r.item() - TRUE_R) / TRUE_R * 100
            print(f"Epoch {ep:4d} | Loss {loss.item():.3e} | "
                  f"ε={eps.item():.4f} (err={eps_error:.1f}%) | "
                  f"r={r.item():.4f} (err={r_error:.1f}%)")
    
    print("-" * 80)
    print(f"训练时间: {time.time()-t0:.1f}秒")
    
    # ==================== 5. 保存结果 ====================
    eps_final, r_final = model.physical_params()
    eps_error = abs(eps_final.item() - TRUE_EPS) / TRUE_EPS * 100
    r_error = abs(r_final.item() - TRUE_R) / TRUE_R * 100
    
    print("\n训练完成:")
    print(f"  最终参数: ε={eps_final.item():.4f}, r={r_final.item():.4f}")
    print(f"  目标参数: ε={TRUE_EPS}, r={TRUE_R}")
    print(f"  相对误差: ε={eps_error:.2f}%, r={r_error:.2f}%")
    
    # 保存模型
    model_manager = ModelManager()
    metrics = {
        'eps_pred': eps_final.item(),
        'r_pred': r_final.item(),
        'eps_error': eps_error,
        'r_error': r_error
    }
    model_manager.save_best_model(model, eps_final.item(), r_final.item(), metrics, device=device)
    
    print("\n模型已保存到: model/best_models/")
    
    return hist, model, xy_obs_list, Eobs_re_list, Eobs_im_list, x_src


def solve_quick_mode(device, data_mode="simul"):
    """
    快速测试模式
    --mode quick --data simul
    """
    print("\n快速测试模式启动")
    print(f"数据模式: {data_mode}")
    
    # ==================== 1. 加载最新模型 ====================
    print("\n加载最新训练模型...")
    
    model_manager = ModelManager()
    
    # 查找最新模型
    best_models_dir = os.path.join(model_manager.base_dir, "best_models")
    if not os.path.exists(best_models_dir):
        print("没有找到训练好的模型，请先运行正常训练模式")
        return None
    
    model_files = [f for f in os.listdir(best_models_dir) if f.endswith('.pt')]
    if not model_files:
        print("没有找到训练好的模型，请先运行正常训练模式")
        return None
    
    # 选择最新的模型
    model_files.sort(reverse=True)
    latest_model = model_files[0]
    print(f"选择模型: {latest_model}")
    
    # 加载模型
    model = MultiFreqPINN(width=128).to(device)
    save_dict = model_manager.load_best_model(latest_model, model, device=device)
    
    if not save_dict:
        print("模型加载失败")
        return None
    
    # 获取模型参数
    eps_pred = save_dict.get('eps', 0)
    r_pred = save_dict.get('r', 0)
    print(f"加载的模型参数: ε={eps_pred:.4f}, r={r_pred:.4f}")
    
    # ==================== 2. 生成测试数据 ====================
    print("\n生成测试数据...")
    
    # 设置参数
    N_src, N_obs = 120, 40
    
    # 生成新的测试数据
    x_src = (2 * torch.rand(N_src, 2) - 1) * 0.4
    x_src = x_src.to(device)
    
    # 生成观测数据（使用相同的正向计算）
    print("  生成测试观测数据...")
    xy_obs_list, Eobs_re_list, Eobs_im_list = generate_observations(
        N_obs, x_src, TRUE_EPS, TRUE_R, device
    )
    print(f"  测试数据生成完成: {len(frequencies)}个频率 × {len(angles)}个角度")
    
    # ==================== 3. 在测试数据上评估 ====================
    print("\n在测试数据上评估模型...")
    
    model.eval()  # 设置为评估模式
    
    # 计算测试损失
    total_loss = 0.0
    idx = 0
    
    with torch.no_grad():
        for fi, k in enumerate(frequencies):
            for ai, theta in enumerate(angles):
                loss = data_loss(
                    model,
                    xy_obs_list[idx],
                    Eobs_re_list[idx],
                    Eobs_im_list[idx],
                    fi, ai, theta, k
                )
                total_loss += loss.item()
                idx += 1
    
    avg_loss = total_loss / idx if idx > 0 else 0
    
    # 获取当前参数（可能和保存的略有不同）
    eps_current, r_current = model.physical_params()
    eps_current_val = eps_current.item()
    r_current_val = r_current.item()
    
    # 计算误差
    eps_error = abs(eps_current_val - TRUE_EPS) / TRUE_EPS * 100
    r_error = abs(r_current_val - TRUE_R) / TRUE_R * 100
    
    print("\n测试结果:")
    print(f"  测试损失: {avg_loss:.4e}")
    print(f"  当前参数: ε={eps_current_val:.4f}, r={r_current_val:.4f}")
    print(f"  目标参数: ε={TRUE_EPS}, r={TRUE_R}")
    print(f"  相对误差: ε={eps_error:.2f}%, r={r_error:.2f}%")
    
    # 创建简单的历史记录用于绘图
    hist = defaultdict(list)
    hist['eps'] = [eps_pred, eps_current_val]
    hist['r'] = [r_pred, r_current_val]
    hist['loss'] = [avg_loss]
    
    return hist, model, xy_obs_list, Eobs_re_list, Eobs_im_list, x_src


# 导出函数
__all__ = ['solve_normal_mode', 'solve_quick_mode']
