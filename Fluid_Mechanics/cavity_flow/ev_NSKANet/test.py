import torch
from tools import *
import cavity_data as cavity
import pinn_solver as psolver

def train(net_params=None):
    Re = 2000 # Reynolds number
    lam_bcs = 10
    lam_equ = 1
    N_f = 10000
    N_b = 257
    alpha_evm = 0.03

    PINN = psolver.PysicsInformedNeuralNetwork(
        Re=Re,
        N_f = N_f,
        alpha_evm=alpha_evm,
        bc_weight=lam_bcs,
        eq_weight=lam_equ,
        net_params=net_params,
        checkpoint_path='./checkpoint/')

    path = './datasets/'
    dataloader = cavity.DataLoader(path=path, N_f=N_f, N_b=N_b)

    filename = './data/cavity_Re'+str(Re)+'_256.mat'
    x_star, y_star, u_star, v_star = dataloader.loading_evaluate_data(filename)
    # evaluate
    PINN.evaluate(x_star, y_star, u_star, v_star)

if __name__ == "__main__":
    for loop in range(0,1):
        net_params = 'results/KAN_Re2k_evm01_epoch_500000_ev_net.pth'
        train(net_params=net_params)
