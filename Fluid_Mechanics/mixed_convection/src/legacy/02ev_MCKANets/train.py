import torch
from tools import *
import cavity_data as cavity
import pinn_solver as psolver
import time

def train(net_params=None, loop=None):
    Re = 2000 # Reynolds number
    Pr = 0.71
    Ri = 0.1
    lam_bcs = 10
    lam_equ = 1
    N_f = 10000
    alpha_evm = 0.03
    N_b = 1000
    loop= loop

    PINN = psolver.PysicsInformedNeuralNetwork(
        Re=Re,
        Pr=Pr,
        Ri=Ri,
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

    filename = './data/cavity_Re'+str(Re)+'Pr'+str(Pr)+'Ri'+str(Ri)+'_256.mat'
    x_star, y_star, u_star, v_star, t_star = dataloader.loading_evaluate_data(filename)

    # Training
    start_time = time.time()
    epoch=200000
    PINN.set_alpha_evm(0.05)
    PINN.train(num_epoch=epoch, lr=1e-3, label=200000)
    PINN.test(x_star, y_star, u_star, v_star, t_star, 200000, loop)

    PINN.set_alpha_evm(0.03)
    PINN.train(num_epoch=epoch, lr=2e-4, label=400000)
    PINN.test(x_star, y_star, u_star, v_star, t_star, 400000, loop)

    PINN.set_alpha_evm(0.02)
    PINN.train(num_epoch=epoch, lr=5e-5, label=600000)
    PINN.test(x_star, y_star, u_star, v_star, t_star, 900000, loop)
    
    PINN.set_alpha_evm(0.01)
    PINN.train(num_epoch=epoch, lr=5e-5, label=800000)
    PINN.test(x_star, y_star, u_star, v_star, t_star, 800000, loop)

    train_time = time.time() - start_time
    print(train_time)

if __name__ == "__main__":
    for loop in range(0,1):
        train(loop=loop)