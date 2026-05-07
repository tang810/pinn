#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri


def create_three_subplots_figure(u_exact, u_pred, X_star1, X_star2, X_star3, 
                                X_fi1_train_Plot, X_fi2_train_Plot, 
                                figure_save_path, mode="train"):
    """
    创建包含三个子图的对比图像（覆盖保存）
    """
    # ======================
    # 数据准备
    # ======================
    X1, Y1 = X_star1[:, 0:1], X_star1[:, 1:2]
    X2, Y2 = X_star2[:, 0:1], X_star2[:, 1:2]
    X3, Y3 = X_star3[:, 0:1], X_star3[:, 1:2]
    
    x_tot = np.concatenate([X1, X2, X3])
    y_tot = np.concatenate([Y1, Y2, Y3])
    triang_total = tri.Triangulation(x_tot.flatten(), y_tot.flatten())
    
    u_pred_all = np.concatenate([u_pred[0], u_pred[1], u_pred[2]])
    error = np.abs(np.squeeze(u_exact) - u_pred_all.flatten())

    # ======================
    # 绘图
    # ======================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    contour1 = axes[0].tricontourf(triang_total, np.squeeze(u_exact), 100, cmap='jet')
    axes[0].set_title('$u$ (Exact)', fontsize=16)
    axes[0].set_xlabel('$x$')
    axes[0].set_ylabel('$y$')
    axes[0].grid(True, alpha=0.3)
    axes[0].plot(X_fi1_train_Plot[:, 0], X_fi1_train_Plot[:, 1], 'w-', lw=2)
    axes[0].plot(X_fi2_train_Plot[:, 0], X_fi2_train_Plot[:, 1], 'w-', lw=2)

    contour2 = axes[1].tricontourf(triang_total, u_pred_all.flatten(), 100, cmap='jet')
    axes[1].set_title('$u$ (Predicted)', fontsize=16)
    axes[1].set_xlabel('$x$')
    axes[1].set_ylabel('$y$')
    axes[1].grid(True, alpha=0.3)
    axes[1].plot(X_fi1_train_Plot[:, 0], X_fi1_train_Plot[:, 1], 'w-', lw=2)
    axes[1].plot(X_fi2_train_Plot[:, 0], X_fi2_train_Plot[:, 1], 'w-', lw=2)

    contour3 = axes[2].tricontourf(triang_total, error, 100, cmap='jet')
    axes[2].set_title('Point-wise Error', fontsize=16)
    axes[2].set_xlabel('$x$')
    axes[2].set_ylabel('$y$')
    axes[2].grid(True, alpha=0.3)
    axes[2].plot(X_fi1_train_Plot[:, 0], X_fi1_train_Plot[:, 1], 'w-', lw=2)
    axes[2].plot(X_fi2_train_Plot[:, 0], X_fi2_train_Plot[:, 1], 'w-', lw=2)

    fig.colorbar(contour1, ax=axes[0])
    fig.colorbar(contour2, ax=axes[1])
    fig.colorbar(contour3, ax=axes[2])

    # ======================
    # 覆盖保存（关键修改）
    # ======================
    png_path = os.path.join(
        figure_save_path, f"XPINN_Poisson_{mode}.png"
    )
    pdf_path = os.path.join(
        figure_save_path, f"XPINN_Poisson_{mode}.pdf"
    )

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    # plt.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)

    print("图像已覆盖保存：")
    print(f"  PNG: {png_path}")
    # print(f"  PDF: {pdf_path}")

    return png_path, pdf_path

def plot_training_history(loss_history, iterations, figure_save_path):
    """绘制训练损失历史"""
    plt.figure(figsize=(10, 6))
    iteration_points = range(0, iterations, 100)
    plt.plot(iteration_points[:len(loss_history)], loss_history, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Total Loss', fontsize=12)
    plt.title('Training Loss History', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # 保存损失历史图
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    loss_png_path = os.path.join(figure_save_path, f"training_loss_{timestamp}.png")
    loss_pdf_path = os.path.join(figure_save_path, f"training_loss_{timestamp}.pdf")
    
    plt.savefig(loss_png_path, dpi=150, bbox_inches='tight')
    plt.savefig(loss_pdf_path, bbox_inches='tight')
    plt.show()
    
    print(f"损失历史图已保存到:")
    print(f"  PNG格式: {loss_png_path}")
    print(f"  PDF格式: {loss_pdf_path}")
    
    return loss_png_path, loss_pdf_path