import torch
# from tools import *
import SM_data as SM
import pinn_solver as psolver


def train(net_params=None, loop=None):
    # 材料参数
    E = 100  # 弹性模量
    nu = 0.3  # 泊松比

    # 神经网络和训练参数设置
    N_neu = 80  # 神经网络隐藏层神经元数量
    lam_bcs = 10  # 边界条件损失权重
    lam_equ = 1  # 方程损失权重
    N_f = 10000  # 自定义点数量
    N_HLayer = 6  # 隐藏层数量

    # 网络层配置
    layers = [3] + N_HLayer * [N_neu] + [3]

    # 创建物理信息神经网络（PINN）实例
    PINN = psolver.PysicsInformedNeuralNetwork(
        E=E,
        nu=nu,
        layers=layers,
        bc_weight=lam_bcs,
        eq_weight=lam_equ,
        net_params=net_params,
        checkpoint_path='./checkpoint/'
    )

    # 数据加载路径
    path = './datasets/'
    dataloader = SM.DataLoader(path=path, N_f=N_f, N_b=1000)

    # 加载数据文件
    filename = './data/data.mat'

    # 设置边界数据
    boundary_data = dataloader.loading_boundary_data(filename)
    PINN.set_boundary_data(X=boundary_data)

    # 设置训练数据
    training_data = dataloader.loading_training_data(filename)
    PINN.set_eq_training_data(X=training_data)

    # 加载评估数据
    x_star, y_star, z_star, u_star, v_star, w_star = dataloader.loading_evaluate_data(filename)

    # Training
    PINN.train(num_epoch=30000, lr=1e-3)
    PINN.test(x_star, y_star, z_star, u_star, v_star, w_star, loop, 30000)
    saved_ckpt = 'model_SM_loop%d.pth'%(30000)
    PINN.save(saved_ckpt, N_HLayer=N_HLayer, N_neu=N_neu, N_f=N_f)

    PINN.train(num_epoch=30000, lr=2e-4)
    PINN.test(x_star, y_star, z_star, u_star, v_star, w_star, loop, 60000)
    PINN.train(num_epoch=30000, lr=4e-5)
    PINN.test(x_star, y_star, z_star, u_star, v_star, w_star, loop, 90000)
    saved_ckpt = 'model_SM_loop%d.pth'%(90000)
    PINN.save(saved_ckpt, N_HLayer=N_HLayer, N_neu=N_neu, N_f=N_f)

    PINN.train(num_epoch=50000, lr=1e-5)
    PINN.train(num_epoch=100000, lr=2e-6)
    PINN.test(x_star, y_star, z_star, u_star, v_star, w_star, loop, 240000)
    saved_ckpt = 'model_SM_loop%d.pth'%(240000)
    PINN.save(saved_ckpt, N_HLayer=N_HLayer, N_neu=N_neu, N_f=N_f)

if __name__ == "__main__":
    for loop in range(0, 3):
        train()
