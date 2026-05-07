# import torch
# from tools import *
import cavity_data as cavity
import pinn_solver as psolver
import time


def train(net_params=None, loop=None):
        Re = 2000 # Reynolds number
        Pr = 0.71 # Prandtl number
        layers=4  # 神经网络层数
        hidden_size=120 #神经元个数
        lam_bcs = 10 #边界loss的权重
        lam_equ = 1  #方程loss的权重
        N_f = 40000 #内部采样点个数
        N_b = 2500 #边界采样点个数
        loop=loop
    
        PINN = psolver.PysicsInformedNeuralNetwork(
            Re=Re,
            Pr=Pr,
            layers=layers,
            hidden_size=hidden_size,      
            bc_weight=lam_bcs,
            eq_weight=lam_equ,
            net_params=net_params, #训练参数
            checkpoint_path='./checkpoint/')
    
        path = './datasets/'
        dataloader = cavity.DataLoader(path=path, N_f=N_f, N_b=N_b)
    
        # Set boundary data, | u, v, x, y
        boundary_data = dataloader.loading_boundary_data()
        PINN.set_boundary_data(X=boundary_data)
    
        # Set training data, | x, y
        training_data = dataloader.loading_training_data()
        PINN.set_eq_training_data(X=training_data)
        # Set test data,  | u, v, p, x, y
        filename = './data/cavity_Re'+str(Re)+'_Pr0.71_256.mat'
        x_star, y_star, u_star, v_star, t_star = dataloader.loading_evaluate_data(filename)

        # Training
        start_time = time.time()
        epoch=300000
        PINN.train(num_epoch=epoch, lr=1e-3)
        PINN.test(x_star, y_star, u_star, v_star, t_star, 300000, loop)
        
        PINN.train(num_epoch=epoch, lr=2e-4)
        PINN.test(x_star, y_star, u_star, v_star, t_star, 600000, loop)
    
        PINN.train(num_epoch=epoch, lr=5e-5)
        PINN.test(x_star, y_star, u_star, v_star, t_star, 900000, loop)
        
        PINN.train(num_epoch=epoch, lr=5e-5)
        PINN.test(x_star, y_star, u_star, v_star, t_star, 1200000,loop)
        end_time = time.time()
        training_time = start_time - end_time
        print("training time:", training_time)

if __name__ == "__main__":
    for loop in range(0,1):
        train(loop=loop)
