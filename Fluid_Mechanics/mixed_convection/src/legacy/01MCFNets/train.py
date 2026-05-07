# import torch
# from tools import *
import cavity_data as cavity
import pinn_solver as psolver


def train(net_params=None):
    for loop in range(1,5):
        Re = 2000 # Reynolds number
        Pr = 0.71 # Prandtl number
        Ri = 0.1 # Grashof number
        Pe = Re * Pr # Peclet number
        layers=4  #
        hidden_size=120
        lam_bcs = 10
        lam_equ = 1
        N_f = 40000
        N_HLayer = 4
        N_neu = 120
    
        PINN = psolver.PysicsInformedNeuralNetwork(
            Re=Re,
            Pr=Pr,
            Pe = Pe,
            Ri = Ri,
            layers=layers,
            hidden_size=hidden_size,
            bc_weight=lam_bcs,
            eq_weight=lam_equ,
            net_params=net_params,
            checkpoint_path='./checkpoint/')
    
        path = './datasets/'
        dataloader = cavity.DataLoader(path=path, N_f=N_f, N_b=1000)
    
        # Set boundary data, | u, v, x, y
        boundary_data = dataloader.loading_boundary_data()
        PINN.set_boundary_data(X=boundary_data)
    
        # Set training data, | x, y
        training_data = dataloader.loading_training_data()
        PINN.set_eq_training_data(X=training_data)
    
        filename = './data/cavity_Re'+str(Re)+'_256.mat'
        x_star, y_star, u_star, v_star, p_star, t_star = dataloader.loading_evaluate_data(filename)
    
        # Training
        
        PINN.train(num_epoch=300000,lr=1e-3)
        PINN.test(x_star, y_star, u_star, v_star, p_star, t_star, 300000, loop)
        saved_ckpt = 'model_cavity_loop%d.pth'%(300000)
        PINN.save(saved_ckpt, N_HLayer=N_HLayer, N_neu=N_neu, N_f=N_f) 
       
        PINN.train(num_epoch=300000, lr=2e-4)
        PINN.test(x_star, y_star, u_star, v_star, p_star, t_star, 600000, loop)
        
        PINN.train(num_epoch=300000, lr=5e-5)
        PINN.test(x_star, y_star, u_star, v_star, p_star, t_star, 900000, loop)
        saved_ckpt = 'model_cavity_loop%d.pth'%(900000)
        PINN.save(saved_ckpt, N_HLayer=N_HLayer, N_neu=N_neu, N_f=N_f)
        
        PINN.train(num_epoch=300000, lr=5e-5)
        PINN.test(x_star, y_star, u_star, v_star, p_star, t_star, 1200000, loop)
        saved_ckpt = 'model_cavity_loop%d.pth'%(1200000)
        PINN.save(saved_ckpt, N_HLayer=N_HLayer, N_neu=N_neu, N_f=N_f)
        
        '''
        PINN.train(num_epoch=500, lr=1e-5)
        PINN.test(x_star, y_star, u_star, v_star, p_star, t_star, 1400000, loop)
        PINN.train(num_epoch=1000, lr=2e-6)
        PINN.test(x_star, y_star, u_star, v_star, p_star, t_star, 2400000, loop)
        saved_ckpt = 'model_cavity_loop%d.pth'%(2400000)
        PINN.save(saved_ckpt, N_HLayer=N_HLayer, N_neu=N_neu, N_f=N_f)
        '''
if __name__ == "__main__":
    train()
