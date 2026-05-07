# Copyright (c) 2023 Se42 Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import torch
from tools import *
import cavity_data as cavity
import pinn_solver as psolver
from torch.utils.data import DataLoader
torch.set_num_threads(2)


def main(net_params=None, net_params_1=None):
    Re = 5000   # Reynolds number
    N_neu = 120
    N_neu_1 = 40
    lam_bcs = 10
    lam_equ = 1
    N_f = 120000
    batchsize = 30000
    alpha_evm = 0.03
    N_HLayer = 4
    N_HLayer_1 = 4

    path = './datasets/'
    dataloader = cavity.DataLoader(path=path, N_f=N_f, N_b=2049)
    #train_dl = DataLoader(dataloader, shuffle=True, drop_last=True)
    #print("Training batches: {}".format(len(train_dl)))

    PINN = psolver.PysicsInformedNeuralNetwork(
        Re=Re,
        layers=N_HLayer,
        layers_1=N_HLayer_1,
        hidden_size = N_neu,
        hidden_size_1 = N_neu_1,
        N_f = N_f,
        alpha_evm=alpha_evm,
        bc_weight=lam_bcs,
        eq_weight=lam_equ,
        batchsize=batchsize,
        net_params=net_params,
        net_params_1=net_params_1,
        distributed=True,
        checkpoint_path='./checkpoint/')

    # Set boundary data, | u, v, x, y
    boundary_data = dataloader.loading_boundary_data()
    PINN.set_boundary_data(X=boundary_data)

    # Set training data, | x, y
    training_data = dataloader.loading_training_data()
    PINN.set_eq_training_data(X=training_data)

    filename = './data/cavity_Re'+str(Re)+'_256_Uniform.mat'
    x_star, y_star, u_star, v_star = dataloader.loading_evaluate_data(filename)

    # Training
    PINN.set_alpha_evm(0.05)
    batchsize = 20000
    PINN.batchsize = batchsize
    PINN.train(num_epoch=30000, lr=1e-3, mini_counts=(N_f/batchsize))
    saved_ckpt = 'model_cavity_loop_0_%d.pth'%(1)
    PINN.save(saved_ckpt, N_HLayer=N_HLayer, N_neu=N_neu, N_f=N_f)
    PINN.evaluate(x_star, y_star, u_star, v_star)

    PINN.set_alpha_evm(0.03)
    PINN.train(num_epoch=100000, lr=2e-4, mini_counts=(N_f/batchsize))
    saved_ckpt = 'model_cavity_loop_0_%d.pth'%(2)
    PINN.save(saved_ckpt, N_HLayer=N_HLayer, N_neu=N_neu, N_f=N_f)
    PINN.evaluate(x_star, y_star, u_star, v_star)

    PINN.set_alpha_evm(0.02)
    PINN.train(num_epoch=100000, lr=4e-5, mini_counts=(N_f/batchsize))
    saved_ckpt = 'model_cavity_loop_0_%d.pth'%(3)
    PINN.save(saved_ckpt, N_HLayer=N_HLayer, N_neu=N_neu, N_f=N_f)
    PINN.evaluate(x_star, y_star, u_star, v_star)

    PINN.set_alpha_evm(0.01)
    PINN.train(num_epoch=100000, lr=1e-5, mini_counts=(N_f/batchsize))
    saved_ckpt = 'model_cavity_loop_0_%d.pth'%(4)
    PINN.save(saved_ckpt, N_HLayer=N_HLayer, N_neu=N_neu, N_f=N_f)
    PINN.evaluate(x_star, y_star, u_star, v_star)

if __name__ == "__main__":
    net_params = None #'./simulation_06102023/results/Re5000/4x120_Nf120k_lamB10_alpha0.02/model_cavity_loop_0_3.pth'
    net_params_1 = None #'./simulation_06102023/results/Re5000/4x120_Nf120k_lamB10_alpha0.02/model_cavity_loop_0_3.pth_evm'
    main(net_params=net_params, net_params_1=net_params_1)
