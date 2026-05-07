import torch
from tools import *
import cavity_data as cavity
import pinn_solver as psolver
import time

def train(loop=loop):
        loop = loop
        Re = 2000 # Reynolds number
        lam_bcs = 10
        lam_equ = 1
        N_f = 10000
        N_b = 257


        PINN = psolver.PysicsInformedNeuralNetwork(
            loop=loop
            Re=Re,
            N_f = N_f,
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
    
        filename = './data/cavity_Re'+str(Re)+'_256.mat'
        x_star, y_star, u_star, v_star = dataloader.loading_evaluate_data(filename)
    
        # Training
        start_time = time.time()
        epoch = 100000
        PINN.train(num_epoch=epoch, lr=1e-3, label=100000)
        PINN.test(x_star, y_star, u_star, v_star, 100000, loop)

        PINN.train(num_epoch=epoch, lr=2e-4, label=200000)
        PINN.test(x_star, y_star, u_star, v_star, 200000, loop)

        PINN.train(num_epoch=epoch, lr=5e-5, label=300000)
        PINN.test(x_star, y_star, u_star, v_star, 300000, loop)

        PINN.train(num_epoch=epoch, lr=5e-5, label=400000)
        PINN.test(x_star, y_star, u_star, v_star, 400000, loop)

        PINN.train(num_epoch=epoch, lr=1e-5, label=500000)
        PINN.test(x_star, y_star, u_star, v_star, 500000, loop)

        train_time = time.time() - start_time
        print(train_time)

if __name__ == "__main__":
    for loop in range(0, 1): #设置循环次数
        train(loop=loop)

