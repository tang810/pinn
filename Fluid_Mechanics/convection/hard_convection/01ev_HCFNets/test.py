import torch
from tools import *
import cavity_data as cavity
import pinn_solver as psolver

def train(net_params=None):
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
    net_params = net_params

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

    filename = './data/cavity_Re'+str(Re)+'Pr'+str(Pr)+'_256.mat'
    x_star, y_star, u_star, v_star, t_star = dataloader.loading_evaluate_data(filename)
    # evaluate
    PINN.evaluate(x_star, y_star, u_star, v_star, t_star)

if __name__ == "__main__":
    for loop in range(0,1):
        net_params = 'results/Re2KPr0.71/4x120_Nf40k_lamB10_alpha0.01/model_cavity_loop290000.pth'
        train(net_params=net_params)
