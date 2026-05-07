#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import numpy as np

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.xpinn_model import XPINN
from src.data_utils import load_data, prepare_training_data
from src.visualization import create_three_subplots_figure
from src.visualization import plot_training_history

# 设置图片保存路径
FIGURE_SAVE_PATH = r"/data/AI4PDE_CN/src/PINN4Science/Variant_PINNs/XPINN/results"
MODEL_SAVE_PATH = r"/data/AI4PDE_CN/src/PINN4Science/Variant_PINNs/XPINN/models"
DATA_PATH = r"/data/AI4PDE_CN/src/PINN4Science/Variant_PINNs/XPINN/data/XPINN_2D_PoissonEqn.mat"

os.makedirs(FIGURE_SAVE_PATH, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs("results", exist_ok=True)

print(f"图片将保存到: {FIGURE_SAVE_PATH}")
print(f"模型将保存到: {MODEL_SAVE_PATH}")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='XPINN 2D Poisson Equation Solver')
    
    parser.add_argument('--mode', type=str, choices=['quick', 'train'], default='quick',
                       help='运行模式: quick(加载模型直接预测), train(完整训练)')
    
    parser.add_argument('--data', type=str, choices=['simul', 'load'], default='load',
                       help='数据来源: simul(模拟数据), load(加载.mat文件)')
    
    parser.add_argument('--data_path', type=str, default=DATA_PATH,
                       help='数据文件路径（当data=load时使用）')
    
    parser.add_argument('--model_path', type=str, default=None,
                       help='模型文件路径（当mode=quick时使用，默认为models/xpinn_trained_model.ckpt）')
    
    parser.add_argument('--iterations', type=int, default=2000,
                       help='训练迭代次数（当mode=train时使用）')
    
    parser.add_argument('--save_model', action='store_true', default=True,
                       help='是否保存训练后的模型（当mode=train时）')
    
    return parser.parse_args()

def check_for_trained_model():
    """检查是否存在训练好的模型"""
    default_model = os.path.join(MODEL_SAVE_PATH, "xpinn_trained_model.ckpt.index")
    if os.path.exists(default_model):
        return os.path.join(MODEL_SAVE_PATH, "xpinn_trained_model.ckpt")
    
    # 查找最新的模型
    model_files = []
    for file in os.listdir(MODEL_SAVE_PATH):
        if file.endswith('.index'):
            model_name = file.replace('.index', '')
            model_path = os.path.join(MODEL_SAVE_PATH, model_name)
            model_files.append((model_path, os.path.getmtime(model_path + '.index')))
    
    if model_files:
        # 按修改时间排序，返回最新的
        model_files.sort(key=lambda x: x[1], reverse=True)
        return model_files[0][0]
    
    return None

def run_training_mode(args, training_data):
    """运行训练模式"""
    print(f"\n=== 训练模式 ===")
    print(f"迭代次数: {args.iterations}")
    
    # 解包数据
    (X_f1_train, X_f2_train, X_f3_train, X_fi1_train, X_fi2_train, 
     X_ub_train, ub_train, X_star1, X_star2, X_star3, 
     X_fi1_train_Plot, X_fi2_train_Plot, u_exact) = training_data
    
    # 网络架构
    layers1 = [2, 30, 30, 1]
    layers2 = [2, 20, 20, 20, 20, 1]
    layers3 = [2, 25, 25, 25, 1]
    
    # 创建模型
    print("创建XPINN模型...")
    model = XPINN(X_ub_train, ub_train, X_f1_train, X_f2_train, X_f3_train,
                 X_fi1_train, X_fi2_train, layers1, layers2, layers3)
    
    # 训练模型
    print("开始训练模型...")
    start_time = time.time()
    loss_history = model.train(args.iterations)
    elapsed = time.time() - start_time
    print(f"训练完成! 耗时: {elapsed:.2f} 秒")
    
    # 保存模型
    if args.save_model:
        model_path = os.path.join(MODEL_SAVE_PATH, "xpinn_trained_model.ckpt")
        model.save_model(model_path)
    
    return model, loss_history

def run_quick_mode(args, training_data):
    """运行快速模式（直接加载模型预测）"""
    print(f"\n=== 快速模式 ===")
    print("加载预训练模型进行预测...")
    
    # 解包数据
    (X_f1_train, X_f2_train, X_f3_train, X_fi1_train, X_fi2_train, 
     X_ub_train, ub_train, X_star1, X_star2, X_star3, 
     X_fi1_train_Plot, X_fi2_train_Plot, u_exact) = training_data
    
    # 确定模型路径
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = check_for_trained_model()
    
    if not model_path or not os.path.exists(model_path + '.index'):
        print("错误: 未找到预训练模型!")
        print("请先运行训练模式或指定正确的模型路径:")
        print("  python main.py --mode train")
        print("或")
        print(f"  python main.py --mode quick --model_path /path/to/model.ckpt")
        return None, None
    
    print(f"加载模型: {model_path}")
    
    # 网络架构（必须与训练时一致）
    layers1 = [2, 30, 30, 1]
    layers2 = [2, 20, 20, 20, 20, 1]
    layers3 = [2, 25, 25, 25, 1]
    
    # 创建模型（不训练，只用于预测）
    model = XPINN(X_ub_train, ub_train, X_f1_train, X_f2_train, X_f3_train,
                 X_fi1_train, X_fi2_train, layers1, layers2, layers3)
    
    # 加载模型权重
    model.load_model(model_path)
    
    return model, None

def main():
    args = parse_arguments()
    
    print("=== XPINN 2D Poisson Equation Solver ===")
    print(f"运行模式: {args.mode}")
    print(f"数据来源: {args.data}")
    
    # 加载数据
    data = load_data(args.data_path, args.data)
    
    # 准备训练数据
    training_data = prepare_training_data(data)
    (X_f1_train, X_f2_train, X_f3_train, X_fi1_train, X_fi2_train, 
     X_ub_train, ub_train, X_star1, X_star2, X_star3, 
     X_fi1_train_Plot, X_fi2_train_Plot, u_exact) = training_data
    
    # 根据模式运行
    if args.mode == 'train':
        model, loss_history = run_training_mode(args, training_data)
        if model is None:
            return
    else:  # quick mode
        model, loss_history = run_quick_mode(args, training_data)
        if model is None:
            return
    
    # 预测
    print("进行预测...")
    u_pred1, u_pred2, u_pred3 = model.predict(X_star1, X_star2, X_star3)
    
    # 创建对比图像
    print("生成对比图像...")
    create_three_subplots_figure(
        u_exact, 
        (u_pred1, u_pred2, u_pred3),
        X_star1, X_star2, X_star3,
        X_fi1_train_Plot, X_fi2_train_Plot,
        FIGURE_SAVE_PATH,
        args.mode
    )
    
    # 计算并显示误差
    u_pred_all = np.concatenate([u_pred1, u_pred2, u_pred3])
    l2_error = np.linalg.norm(u_exact.flatten() - u_pred_all.flatten()) / np.linalg.norm(u_exact.flatten())
    print(f"相对L2误差: {l2_error:.3e}")
    
    # 绘制损失历史（如果是训练模式）
    if loss_history and args.mode == 'train':
        plot_training_history(loss_history, args.iterations, FIGURE_SAVE_PATH)
    
    print("\n程序执行完成!")
    print(f"所有图像已保存到: {FIGURE_SAVE_PATH}")

if __name__ == "__main__":
    main()