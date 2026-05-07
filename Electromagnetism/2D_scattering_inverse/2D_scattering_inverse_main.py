#!/usr/bin/env python3
"""
2D电磁反散射PINN主程序
简化版：只支持两种模式
"""

import torch
import numpy as np
import os
import sys
import argparse
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent

# 添加 src 到 Python 路径
sys.path.append(str(PROJECT_ROOT / "src"))

from src.solver import solve_normal_mode, solve_quick_mode
from src.plotting import plot_results
from src.physics import TRUE_EPS, TRUE_R, L, BETA, contrast_chi


def main():
    """主函数"""
    os.chdir(PROJECT_ROOT)

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="2D电磁反散射PINN - 简化版")
    parser.add_argument("--mode", choices=["train", "quick"], default="quick",
                       help="运行模式: train-训练模型, quick-测试模型")
    parser.add_argument("--data", choices=["simul"], default="simul",
                       help="数据模式: simul-生成新数据")
    parser.add_argument("--epochs", type=int, default=3000,
                       help="训练轮数（train模式有效）")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("2D电磁反散射PINN - 简化版")
    print("=" * 60)
    print(f"目标参数: ε={TRUE_EPS}, r={TRUE_R}")
    print(f"运行模式: {args.mode}")
    print(f"数据模式: {args.data}")
    if args.mode == "train":
        print(f"训练轮数: {args.epochs}")
    print("=" * 60)
    
    # 创建必要的目录
    os.makedirs(PROJECT_ROOT / "model" / "best_models", exist_ok=True)
    os.makedirs(PROJECT_ROOT / "results", exist_ok=True)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    if args.mode == "train":
        # ==================== 正常训练模式 ====================
        print("\n开始训练模式...")
        
        # 运行训练
        hist, model, xy_obs_list, Eobs_re_list, Eobs_im_list, x_src = solve_normal_mode(
            device=device,
            data_mode=args.data,
            epochs=args.epochs
        )
        
        # 绘制结果
        print("\n生成训练结果图表...")
        final_eps, final_r = plot_results(
            hist,
            model,
            xy_obs_list,
            Eobs_re_list,
            Eobs_im_list,
            x_src,
            TRUE_EPS=TRUE_EPS,
            TRUE_R=TRUE_R,
            L=L,
            BETA=BETA,
            contrast_chi=contrast_chi,
            device=device
        )
        
        print("\n训练完成!")
        print(f"  下次可以使用快速测试模式:")
        print(f"  python 2D_scattering_inverse_main.py --mode quick --data simul")
        
    else:  # quick模式
        # ==================== 快速测试模式 ====================
        print("\n开始快速测试模式...")
        
        # 运行测试
        result = solve_quick_mode(
            device=device,
            data_mode=args.data
        )
        
        if result:
            hist, model, xy_obs_list, Eobs_re_list, Eobs_im_list, x_src = result
            
            # 绘制结果
            print("\n生成测试结果图表...")
            final_eps, final_r = plot_results(
                hist,
                model,
                xy_obs_list,
                Eobs_re_list,
                Eobs_im_list,
                x_src,
                TRUE_EPS=TRUE_EPS,
                TRUE_R=TRUE_R,
                L=L,
                BETA=BETA,
                contrast_chi=contrast_chi,
                device=device
            )
            
            print("\n测试完成!")
            print(f"  可以使用正常模式重新训练:")
            print(f"  python 2D_scattering_inverse_main.py --mode normal --data simul")
        else:
            print("\n测试失败，请先运行正常训练模式")
    
    print("=" * 60)
    print("程序执行完成！")


if __name__ == "__main__":
    main()
