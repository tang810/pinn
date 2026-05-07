
# from sympy import im
import torch
import pandas as pd
import sys
import os
# 将父级路径添加到 sys.path 中
parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_path)
from net import PINNsformer,DNN

from utils import make_sequence
# import time
import numpy as np
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Seeds
torch.manual_seed(123)
np.random.seed(123)

def ref_sol(rho, h1, a, X):
    h2 = 1
    # h1 = 4.13
    g = 9.18
    # rho = 0.977
    h = h2/h1
    # a = -0.77
    c0 = np.sqrt((g * h2 *(1-rho)) / ((h2/h1) + rho))
    c1 = -3/2 * c0 * ((rho-h**2) / (rho * h2 + h*h2))
    c2 = 1/6 * c0 * ((rho * h2**2 * h1 + h1**2 *h2) / (rho * h1 + h2))
    c = c0 + a*c1 / 3
    lam = np.sqrt(12*c2 / (a*c1))
    # X = np.linspace(-20,20,100)
    # # import pdb
    # # pdb.set_trace()
    sol = a/(np.cosh(X/lam)**2)
    return sol
    # plt.plot(X, sol, label=f'a={a}')
    # plt.legend()
    # plt.ylim(-1,0.1)
    # plt.savefig("./sol.png")


def plot_figa(pred, ref, a, save_path, num=5):
    fig, axes = plt.subplots(2, num, figsize=(20, 8))  # 宽度 20 英寸，高度 4 英寸

    for i, ax in enumerate(axes.flat):
        # 计算和格式化 a 的值
        print(i)
        a_value = round(a[i, 0, 0], 3)  # 使用 round 确保三位小数

        # 绘制预测数据
        ax.scatter(x[i], pred[i], label=f'pred', color='b',
                   marker='o', edgecolors='k', alpha=0.7, s=50)

        # 绘制参考数据
        ax.plot(x[i], ref[i], label=f'ref', color='r',
                linestyle='--', alpha=0.7, linewidth=2)

        # 设置图例
        ax.legend(loc='best', fontsize=10)

        # 设置标题，增加字体大小和格式
        ax.set_title(f'a={a_value:.3f}', fontsize=12,
                     fontweight='bold', color='darkblue')

        # 设置坐标轴标签
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)

        # 设置坐标轴范围（可根据数据调整）
        # ax.set_ylim(-0.5, 0.1)

        # 设置网格
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

        # 设置刻度线
        ax.tick_params(axis='both', which='both', length=5, width=1,
                       colors='black', direction='in',
                       grid_color='gray', grid_alpha=0.5)

    # 自动调整子图间距
    plt.tight_layout()

    # 保存图形
    plt.savefig(save_path, dpi=300)


def error(pred, ref):
    error = np.mean(np.abs(pred-ref))
    # L2_error = np.linalg.norm(pred - ref) / np.linalg.norm(ref)
    return error


def plot_fig(pred,pred_noboundary, ref, a, rho, h1, save_path, fig_num=5, num=10):
    fig, axes = plt.subplots(2, fig_num, figsize=(25, 5))  # 宽度 20 英寸，高度 4 英寸
    errors = []
    a_values = []
    rho_values = []
    h1_values = []
    l2_errors_boundary = []
    l2_errors_noboundary = []
    for i, ax in enumerate(axes[0, :]): #axes.flat
        # 计算和格式化 a 的值
        print(i)
        a_value = round(a[i, 0, 0], 3)  # 使用 round 确保三位小数
        rho_value = round(rho[i, 0, 0], 3)
        h1_value = round(h1[i, 0, 0], 3)
        l2_error = error(pred[i], ref[i])
        a_values.append(a_value)
        rho_values.append(rho_value)
        h1_values.append(h1_value)
        errors.append(l2_error)
        print(errors)
        
        # # 绘制无边界数据 (稀疏散点)
        # ax.scatter(x[i][::5], pred_noboundary[i][::5], label='No Boundary', color='b', s=30, alpha=0.8, marker='o', edgecolors='k')
        
        # # 绘制边界数据 (稀疏散点)
        # ax.scatter(x[i][::5], pred[i][::5], label='Boundary', color='g', s=30, alpha=0.8, marker='o', edgecolors='k')
        
        # # 绘制参考数据 (实线)
        # ax.plot(x[i], ref[i], label='Reference', color='r', linestyle='-', linewidth=2, alpha=0.8)
        
        jitter = (0.005 * np.random.randn(len(x[i][::6]))).reshape(-1,1)
        ax.scatter(x[i][::6] + jitter, pred_noboundary[i][::6], 
                label='No Boundary',
                color='b', s=80, alpha=0.9, marker='o',
                facecolors='none', edgecolors='b', linewidth=1.5)
        
        # 边界数据 (实心方块) [::6]
        ax.scatter(x[i][::6] - jitter, pred[i][::6],    
                label='Boundary',
                color='g', s=50, alpha=0.9, marker='^')
        
        # 参考数据 (实线)
        ax.plot(x[i], ref[i], label='Reference', color='r', linestyle='-', linewidth=2, alpha=0.8)
        ax.set_ylim([-0.05,0.1])
        
        # 计算误差
        error_boundary = np.abs(np.array(pred[i]) - np.array(ref[i]))
        error_noboundary = np.abs(np.array(pred_noboundary[i]) - np.array(ref[i]))
        
        # L2 误差
        l2_boundary = np.linalg.norm(error_boundary) / np.linalg.norm(ref[i])
        l2_noboundary = np.linalg.norm(error_noboundary) / np.linalg.norm(ref[i])
        l2_errors_boundary.append(l2_boundary)
        l2_errors_noboundary.append(l2_noboundary)
        
        # 误差曲线 (第二行)
        axes[1, i].plot(x[i], error_boundary, color='g', linewidth=1.5, linestyle='--', label='Boundary Error')
        axes[1, i].plot(x[i], error_noboundary, color='b', linewidth=1, label='No Boundary Error', alpha=0.5)
        axes[1, 0].set_ylabel(r"Error=|pred-ref|", fontsize=12)
            
        # # 绘制无边界数据
        # ax.plot(x[i], pred_noboundary[i], label=f'ref', color='r',
        #         linestyle='--', alpha=0.7, linewidth=2)
        
        # # 绘制边界数据
        # ax.plot(x[i], pred[i], label=f'ref', color='r',
        #         linestyle='--', alpha=0.7, linewidth=2)
        
        # # 绘制参考数据
        # ax.plot(x[i], ref[i], label=f'ref', color='r',
        #         linestyle='--', alpha=0.7, linewidth=2)
        
        
        # # 绘制预测数据
        # ax.scatter(x[i], pred[i], label=f'pred', c='none', edgecolor ='b',
        #            marker='o', linewidth=0.5, alpha=0.7, s=50)

        # # 绘制参考数据
        # ax.plot(x[i], ref[i], label=f'ref', color='r',
        #         linestyle='--', alpha=0.7, linewidth=2)

        # 设置图例
        ax.legend(loc='best', fontsize=10)

        # 设置标题，增加字体大小和格式
        ax.set_title(r'$\rho=\frac{\rho_2}{\rho_1}$' + f'={rho_value:.3f}, h1={h1_value:.3f}, a={a_value:.3f}',
                     fontsize=12, fontweight='bold', color='darkblue')

        # 设置坐标轴标签
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)

        # 设置坐标轴范围（可根据数据调整）
        # ax.set_ylim(-0.5, 0.1)

        # 设置网格
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        axes[1, i].grid(True)
        axes[1, i].set_xlabel("x", fontsize=16)
        axes[1, i].legend(loc='best', fontsize=10)
        # 设置刻度线
        ax.tick_params(axis='both', which='both', length=5, width=1,
                       colors='black', direction='in',
                       grid_color='gray', grid_alpha=0.5)

    # 自动调整子图间距
    plt.tight_layout()

    # 保存图形
    plt.savefig(save_path, dpi=300)
    df = pd.DataFrame({'rho_value': rho_values,
                       'h1_value': h1_values,
                       'a_value': a_values,
                       'error': errors})
    df.to_csv(f'error.csv_{num}', index=True)


if __name__ == "__main__":
    mesh = 100
    # x = make_sequence(num_x=30, step=80, min_x=-100, max_x=100)
    x = make_sequence(num_x=5, step=mesh, min_x=38, max_x=42)
    h2 = 1
    g = 9.18
    # h1 = np.array([7.956,4.120,4.794,7.198,9.892,7.764,4.120,4.132,6.609,7.920])
    # rho = np.array([0.986,0.918,0.932,0.942,0.910,0.940,0.932,0.983,0.934,0.971])
    # a = np.array([-0.102,-0.082,-0.366,-0.275,-0.323,-0.256,-0.177,-0.106,-0.199,-0.202])
    h1 = np.array([4.794,4.12,4.132,7.956,7.764])
    rho = np.array([0.932,0.918,0.983,0.986,0.940])
    a = np.array([-0.366,-0.082,-0.106,-0.102,-0.256])
    h1 = h1.reshape(-1,1)
    rho = rho.reshape(-1,1)
    a = a.reshape(-1,1)
    # h1 = 4.13
    # rho = 0.977
    # h1 = np.linspace(4,10,500)
    # h1 = np.random.choice(h1, 10).reshape(-1,1)
    # h1 = np.linspace(4.13,4.13,10).reshape(-1,1)
    h1 = np.expand_dims(np.tile(h1[:], (mesh)), -1)
    # rho = np.linspace(0.9,0.99,500)
    # rho = np.random.choice(rho, 10).reshape(-1,1)
    # rho = np.linspace(0.977, 0.977, 10).reshape(-1, 1)
    rho = np.expand_dims(np.tile(rho[:], (mesh)), -1)
    # a = np.linspace(-0.3, -0.1, 10).reshape(-1, 1)
    # a = np.array([-0.4, -0.36, -0.77, -0.91, -1.23]).reshape(-1, 1)
    # a = np.linspace(-0.4, -0.05, 500)
    # a = np.random.choice(a, 10).reshape(-1, 1)
    a = np.expand_dims(np.tile(a[:], (mesh)), -1)
    h = h2/h1

    c0 = np.sqrt((g * h2 *(1-rho)) / ((h2/h1) + rho))
    c1 = -3/2 * c0 * ((rho-h**2) / (rho * h2 + h*h2))
    c2 = 1/6 * c0 * ((rho * h2**2 * h1 + h1**2 *h2) / (rho * h1 + h2))
    c = c0 + a*c1 / 3
    lam = np.sqrt(12*c2 / (a*c1))
    ref = ref_sol(rho, h1, a, x)
    x_test_train = torch.tensor(x,
                              requires_grad=True,
                              dtype=torch.float32).to(device)
    rho = torch.tensor(rho,dtype=torch.float32).to(device)
    h1 = torch.tensor(h1,dtype=torch.float32).to(device)
    a = torch.tensor(a,dtype=torch.float32).to(device)
    # ref = torch.tensor(ref,dtype=torch.float32).to(device)
    test_data = torch.tensor(x,dtype=torch.float32).to(device)
    # model_10 = PINNsformer(d_out=1, d_model=513, d_hidden=64, N=1, heads=3).to(device)
    # model_100 = PINNsformer(d_out=1, d_model=513, d_hidden=64, N=1, heads=3).to(device)
    model_1000 = PINNsformer(d_out=1, d_model=513, d_hidden=64, N=1, heads=3).to(device)
    model_1000_noboundary = PINNsformer(d_out=1, d_model=513, d_hidden=64, N=1, heads=3).to(device)
    # checkpoint_path1 = './model_parameters/PINNsformer_num_h1=10/final_model_11018.pth'
    # checkpoint_path2 = './model_parameters/PINNsformer_num_h1=100/final_model_2001.pth'
    # checkpoint_path3 = './model_parameters/PINNsformer_num_h1=1000/model_1500.pth'
    checkpoint_path_noboundary = './model_parameters/PINNsformer_num_h1=1000/model_194_noboundary.pth'
    checkpoint_path = './model_parameters/PINNsformer_num_h1=1000/model_1500.pth'
    # model_10.load_state_dict(torch.load(checkpoint_path1, weights_only=True))
    # pred_10 = model_10(rho, h1, a, x_test_train).detach().cpu().numpy()
    # model_100.load_state_dict(torch.load(checkpoint_path2, weights_only=True))
    # pred_100 = model_100(rho, h1, a, x_test_train).detach().cpu().numpy()
    # model_1000.load_state_dict(torch.load(checkpoint_path3, weights_only=True))
    model_1000_noboundary.load_state_dict(torch.load(checkpoint_path_noboundary, weights_only=True))
    pred_1000_noboundary = model_1000_noboundary(rho, h1, a, x_test_train).detach().cpu().numpy()
    
    model_1000.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    pred_1000 = model_1000(rho, h1, a, x_test_train).detach().cpu().numpy()
    # import pdb
    # pdb.set_trace()

    a = a.detach().cpu().numpy()
    rho = rho.detach().cpu().numpy()
    h1 = h1.detach().cpu().numpy()
    # save_path_10 = "./fig/sol_num_h1=10.png"
    # save_path_100 = "./fig/sol_num_h1=100.png"
    # save_path_1000 = "./fig/sol_num_h1=1000_noboundary.png"
    save_path = "./fig/sol_num_h1=1000_com.pdf"
    # plot_fig(pred_10, ref, a, rho, h1, save_path_10, fig_num=5, num=10)
    # plot_fig(pred_100, ref, a, rho, h1, save_path_100, fig_num=5, num=100)
    plot_fig(pred_1000, pred_1000_noboundary, ref, a, rho, h1, save_path, fig_num=5, num=1000)
