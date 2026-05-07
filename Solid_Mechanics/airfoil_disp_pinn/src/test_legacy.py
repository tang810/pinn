import torch
from tools import *
import SM_data as SM
import pinn_solver as psolver


def train(net_params=None, loop=None):
    E = 100  # Reynolds number
    nu = 0.3
    N_neu = 80
    lam_bcs = 10
    lam_equ = 1
    N_f = 10000
    N_HLayer = 6
    layers = [3] + N_HLayer * [N_neu] + [3]

    PINN = psolver.PysicsInformedNeuralNetwork(
        E=E,
        nu=nu,
        layers=N_HLayer,
        bc_weight=lam_bcs,
        eq_weight=lam_equ,
        net_params=net_params,
        checkpoint_path='./checkpoint/')

    path = './datasets/'
    dataloader = SM.DataLoader(path=path, N_f=N_f, N_b=1000)

    filename = './data/data.mat'
    # Set boundary data,上面十几个
    boundary_data = dataloader.loading_boundary_data(filename)
    PINN.set_boundary_data(X=boundary_data)

    # Set training data, | x, y, z
    training_data = dataloader.loading_training_data(filename)
    PINN.set_eq_training_data(X=training_data)

    x_star, y_star, z_star, u_star, v_star, w_star = dataloader.loading_evaluate_data(filename)
    # evaluate
    PINN.evaluate(x_star, y_star, z_star, u_star, v_star, w_star)

if __name__ == "__main__":
    for loop in range(0,1):
        net_params = './results/model_SM_loop240000.pth'
        train(net_params=net_params)