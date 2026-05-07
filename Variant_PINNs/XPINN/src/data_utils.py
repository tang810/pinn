#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import scipy.io

def generate_simulated_data():
    """
    生成模拟数据（当没有真实数据时使用）
    """
    print("生成模拟数据...")
    
    # 定义区域
    N = 50
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)
    
    # 精确解
    u_exact = np.exp(X) + np.exp(Y)
    
    # 定义子区域
    idx1 = X.flatten() < 0.4
    idx2 = (X.flatten() >= 0.4) & (X.flatten() < 0.7)
    idx3 = X.flatten() >= 0.7
    
    # 边界点
    boundary_idx = (X.flatten() == 0) | (X.flatten() == 1) | (Y.flatten() == 0) | (Y.flatten() == 1)
    
    # 界面点
    interface1_idx = np.abs(X.flatten() - 0.4) < 0.01
    interface2_idx = np.abs(X.flatten() - 0.7) < 0.01
    
    # 准备数据
    X_total = np.vstack([X.flatten()[:, None], Y.flatten()[:, None]]).T
    
    x_f1 = X_total[idx1, 0][:, None]
    y_f1 = X_total[idx1, 1][:, None]
    x_f2 = X_total[idx2, 0][:, None]
    y_f2 = X_total[idx2, 1][:, None]
    x_f3 = X_total[idx3, 0][:, None]
    y_f3 = X_total[idx3, 1][:, None]
    
    xi1 = X_total[interface1_idx, 0][:, None]
    yi1 = X_total[interface1_idx, 1][:, None]
    xi2 = X_total[interface2_idx, 0][:, None]
    yi2 = X_total[interface2_idx, 1][:, None]
    
    xb = X_total[boundary_idx, 0][:, None]
    yb = X_total[boundary_idx, 1][:, None]
    
    ub_train = u_exact.flatten()[boundary_idx][:, None]
    
    return {
        'x_f1': x_f1, 'y_f1': y_f1,
        'x_f2': x_f2, 'y_f2': y_f2,
        'x_f3': x_f3, 'y_f3': y_f3,
        'xi1': xi1, 'yi1': yi1,
        'xi2': xi2, 'yi2': yi2,
        'xb': xb, 'yb': yb,
        'ub': ub_train,
        'u_exact': u_exact.flatten()[:, None],
        'u_exact2': u_exact.flatten()[idx2][:, None],
        'u_exact3': u_exact.flatten()[idx3][:, None]
    }

def load_data(data_path, data_source='load'):
    """加载或生成数据"""
    if data_source == 'simul':
        print("使用模拟数据...")
        data = scipy.io.loadmat(data_path)
    else:  # load from file
        if not os.path.exists(data_path):
            print(f"警告: 数据文件不存在: {data_path}")
            print("切换到模拟数据模式...")
            data = generate_simulated_data()
        else:
            print(f"加载数据: {data_path}")
            try:
                data = scipy.io.loadmat(data_path)
                print("数据加载成功!")
            except Exception as e:
                print(f"加载数据失败: {str(e)}")
                print("切换到模拟数据模式...")
                data = generate_simulated_data()
    
    return data

def prepare_training_data(data):
    """准备训练数据"""
    x_f1 = data['x_f1'].flatten()[:, None]
    y_f1 = data['y_f1'].flatten()[:, None]
    x_f2 = data['x_f2'].flatten()[:, None]
    y_f2 = data['y_f2'].flatten()[:, None]
    x_f3 = data['x_f3'].flatten()[:, None]
    y_f3 = data['y_f3'].flatten()[:, None]
    xi1 = data['xi1'].flatten()[:, None]
    yi1 = data['yi1'].flatten()[:, None]
    xi2 = data['xi2'].flatten()[:, None]
    yi2 = data['yi2'].flatten()[:, None]
    xb = data['xb'].flatten()[:, None]
    yb = data['yb'].flatten()[:, None]
    ub_train = data['ub'].flatten()[:, None]
    u_exact = data['u_exact'].flatten()[:, None]
    
    # 准备训练数据
    X_f1_train = np.hstack((x_f1.flatten()[:, None], y_f1.flatten()[:, None]))
    X_f2_train = np.hstack((x_f2.flatten()[:, None], y_f2.flatten()[:, None]))
    X_f3_train = np.hstack((x_f3.flatten()[:, None], y_f3.flatten()[:, None]))
    X_fi1_train = np.hstack((xi1.flatten()[:, None], yi1.flatten()[:, None]))
    X_fi2_train = np.hstack((xi2.flatten()[:, None], yi2.flatten()[:, None]))
    X_ub_train = np.hstack((xb.flatten()[:, None], yb.flatten()[:, None]))
    
    # 测试数据点
    X_star1 = np.hstack((x_f1.flatten()[:, None], y_f1.flatten()[:, None]))
    X_star2 = np.hstack((x_f2.flatten()[:, None], y_f2.flatten()[:, None]))
    X_star3 = np.hstack((x_f3.flatten()[:, None], y_f3.flatten()[:, None]))
    
    # 界面点用于绘图
    X_fi1_train_Plot = np.hstack((xi1.flatten()[:, None], yi1.flatten()[:, None]))
    X_fi2_train_Plot = np.hstack((xi2.flatten()[:, None], yi2.flatten()[:, None]))
    
    return (X_f1_train, X_f2_train, X_f3_train, X_fi1_train, X_fi2_train, 
            X_ub_train, ub_train, X_star1, X_star2, X_star3, 
            X_fi1_train_Plot, X_fi2_train_Plot, u_exact)