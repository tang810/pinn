import torch
from tools import *
import cavity_data as cavity
import pinn_solver as psolver


def train(net_params=None, loop=None):
    Re = 2000   # Reynolds number
    Pr = 0.71
    N_neu = 120
    N_neu_1 = 40
    N_neu_2 = 40
    lam_bcs = 10
    lam_equ = 1
    N_f = 40000
    N_b = 2500
    alpha_evm = 0.03
    N_HLayer = 4
    N_HLayer_1 = 4
    N_HLayer_2 = 4

    PINN = psolver.PysicsInformedNeuralNetwork(
        Re=Re,
        Pr=Pr,
        layers=N_HLayer,
        layers_1=N_HLayer_1,
        layers_2=N_HLayer_2,
        hidden_size = N_neu,
        hidden_size_1 = N_neu_1,
        hidden_size_2 = N_neu_2,
        N_f = N_f,
        alpha_evm=alpha_evm,
        bc_weight=lam_bcs,
        eq_weight=lam_equ,
        net_params=net_params,
        checkpoint_path='./checkpoint/')

    path = './datasets/'
    dataloader = cavity.DataLoader(path=path, N_f=N_f, N_b=N_b)

    # Set boundary data, | u, v, x, y
    boundary_data = dataloader.loading_boundary_data()
    PINN.set_boundary_data(X=boundary_data)

    # Set training data, | x, y
    training_data = dataloader.loading_training_data()
    PINN.set_eq_training_data(X=training_data)

    filename = './data/cavity_Re'+str(Re)+'Pr'+str(Pr)+'_256.mat'
    x_star, y_star, u_star, v_star, t_star = dataloader.loading_evaluate_data(filename)

    # Training
    epoch=300000
    PINN.set_alpha_evm(0.05)
    PINN.train(num_epoch=epoch, lr=1e-3)
    PINN.test(x_star, y_star, u_star, v_star, t_star, 300000, loop)

    PINN.set_alpha_evm(0.03)
    PINN.train(num_epoch=epoch, lr=2e-4)
    PINN.test(x_star, y_star, u_star, v_star, t_star, 600000, loop)

    PINN.set_alpha_evm(0.02)
    PINN.train(num_epoch=epoch, lr=5e-5)
    PINN.test(x_star, y_star, u_star, v_star, t_star, 900000, loop)

    PINN.set_alpha_evm(0.01)
    PINN.train(num_epoch=epoch, lr=5e-5)
    PINN.test(x_star, y_star, u_star, v_star, t_star, 1200000,loop)


if __name__ == "__main__":
    for loop in range(0, 1):
        train(loop=loop)
