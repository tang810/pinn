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
import cylinder_re100_data as data
import pinn_solver as psolver


def train(net_params=None, net_params_1=None):

    # 设置PINN网络参数
    N_neu = 120
    N_neu_1 = 20
    lam_bcs = 10
    lam_equ = 10
    lam_ic = 10
    N_b = 4096
    N_train = 15000

    alpha_evm = 0.03
    N_HLayer = 6
    N_HLayer_1 = 6
    #layers = [2] + N_HLayer*[N_neu] + [4]
    
    # 加载数据
    filename = './datasets/cylinder_nektar_Re100.mat'
    dataloader = data.DataLoader(filename=filename,time_window=10)
 #   dataloader.sample_geometry(N_train=N_train, N_b=N_b)

    PINN = psolver.PysicsInformedNeuralNetwork(
        Re=100,
        layers=N_HLayer,
        layers_1=N_HLayer_1,
        hidden_size = N_neu,
        hidden_size_1 = N_neu_1,
        bc_weight=lam_bcs,
        eq_weight=lam_equ,
        ic_weight=lam_ic,
        outlet_weight=lam_bcs,
        checkpoint_path='./checkpoint/',
        net_params=net_params,
        net_params_1 = net_params_1)

    # Evaluate data 
    t_list = dataloader.get_t_list()
    evaluate_data = dataloader.get_evaluate_data(snap=10)
    _, u_star, v_star, t_star, x_star, y_star = evaluate_data

    # Training from T ~ [1,10]
    # the time index range of dataset
    # e.g. random select 10 time frames from index[1,30)
    time_frames = 10
    time_range = [1,10]
    dataloader.set_time_range(time_frames, time_range)
    # Set initial data, | p, u, v, t, x, y
    initial_data = dataloader.get_initial_data()
    PINN.set_initial_data(X=initial_data)
    # sample bc and training data
    dataloader.sample_geometry(N_train=N_train, N_b=N_b)
    boundary_data, outlet_data = dataloader.get_boundary_data()
    # Set boundary data, | u, v, t, x, y
    PINN.set_boundary_data(X=boundary_data)
    # Set outlet data, | p, t, x, y
    PINN.set_outlet_data(X=outlet_data)
    # Set training data, | t, x, y
    training_data = dataloader.get_training_data()
    PINN.set_eq_training_data(X=training_data)

    PINN.set_alpha_evm(0.05)
    PINN.train(num_epoch=50000, lr=1e-3)
    saved_ckpt = 'model_cylinder_loop%d.pth'%(0)
    PINN.save(saved_ckpt, N_HLayer=N_HLayer, N_neu=N_neu, N_f=N_train)
    PINN.evaluate(t_star, x_star, y_star, u_star, v_star)
    next_ic = PINN.prepare_next_ic(t_list[time_range[1]], x_star, y_star)
    
    # Training from T ~ [11,20]
    time_frames = 10
    time_range = [11,20]
    dataloader.set_time_range(time_frames, time_range)
    # Set initial data, | p, u, v, t, x, y
    PINN.set_initial_data(X=next_ic)
    # sample bc and training data
    dataloader.sample_geometry(N_train=N_train, N_b=N_b)
    boundary_data, outlet_data = dataloader.get_boundary_data()
    # Set boundary data, | u, v, t, x, y
    PINN.set_boundary_data(X=boundary_data)
    # Set outlet data, | p, t, x, y
    PINN.set_outlet_data(X=outlet_data)
    # Set training data, | t, x, y
    training_data = dataloader.get_training_data()
    PINN.set_eq_training_data(X=training_data)
    PINN.set_alpha_evm(0.03)
    PINN.train(num_epoch=50000, lr=1e-3)
    saved_ckpt = 'model_cylinder_loop%d.pth'%(1)
    PINN.save(saved_ckpt, N_HLayer=N_HLayer, N_neu=N_neu, N_f=N_train)
    PINN.evaluate(t_star, x_star, y_star, u_star, v_star)
    next_ic = PINN.prepare_next_ic(t_list[time_range[1]], x_star, y_star)

    # Training from T ~ [21,30]
    time_frames = 10
    time_range = [21,30]
    dataloader.set_time_range(time_frames, time_range)
    # Set initial data, | p, u, v, t, x, y
    PINN.set_initial_data(X=next_ic)
    # sample bc and training data
    dataloader.sample_geometry(N_train=N_train, N_b=N_b)
    boundary_data, outlet_data = dataloader.get_boundary_data()
    # Set boundary data, | u, v, t, x, y
    PINN.set_boundary_data(X=boundary_data)
    # Set outlet data, | p, t, x, y
    PINN.set_outlet_data(X=outlet_data)
    # Set training data, | t, x, y
    training_data = dataloader.get_training_data()
    PINN.set_eq_training_data(X=training_data)
    PINN.set_alpha_evm(0.03)
    PINN.train(num_epoch=50000, lr=1e-3)
    saved_ckpt = 'model_cylinder_loop%d.pth'%(2)
    PINN.save(saved_ckpt, N_HLayer=N_HLayer, N_neu=N_neu, N_f=N_train)
    PINN.evaluate(t_star, x_star, y_star, u_star, v_star)
    next_ic = PINN.prepare_next_ic(t_list[time_range[1]], x_star, y_star)

    # Training from T ~ [31,40]
    time_frames = 10
    time_range = [31,40]
    dataloader.set_time_range(time_frames, time_range)
    # Set initial data, | p, u, v, t, x, y
    PINN.set_initial_data(X=next_ic)
    # sample bc and training data
    dataloader.sample_geometry(N_train=N_train, N_b=N_b)
    boundary_data, outlet_data = dataloader.get_boundary_data()
    # Set boundary data, | u, v, t, x, y
    PINN.set_boundary_data(X=boundary_data)
    # Set outlet data, | p, t, x, y
    PINN.set_outlet_data(X=outlet_data)
    # Set training data, | t, x, y
    training_data = dataloader.get_training_data()
    PINN.set_eq_training_data(X=training_data)
    PINN.set_alpha_evm(0.03)
    PINN.train(num_epoch=50000, lr=1e-3)
    saved_ckpt = 'model_cylinder_loop%d.pth'%(3)
    PINN.save(saved_ckpt, N_HLayer=N_HLayer, N_neu=N_neu, N_f=N_train)
    PINN.evaluate(t_star, x_star, y_star, u_star, v_star)
    next_ic = PINN.prepare_next_ic(t_list[time_range[1]], x_star, y_star)

    # Training from T ~ [41,50]
    time_frames = 10
    time_range = [41,50]
    dataloader.set_time_range(time_frames, time_range)
    # Set initial data, | p, u, v, t, x, y
    PINN.set_initial_data(X=next_ic)
    # sample bc and training data
    dataloader.sample_geometry(N_train=N_train, N_b=N_b)
    boundary_data, outlet_data = dataloader.get_boundary_data()
    # Set boundary data, | u, v, t, x, y
    PINN.set_boundary_data(X=boundary_data)
    # Set outlet data, | p, t, x, y
    PINN.set_outlet_data(X=outlet_data)
    # Set training data, | t, x, y
    training_data = dataloader.get_training_data()
    PINN.set_eq_training_data(X=training_data)
    PINN.set_alpha_evm(0.03)
    PINN.train(num_epoch=50000, lr=1e-3)
    saved_ckpt = 'model_cylinder_loop%d.pth'%(4)
    PINN.save(saved_ckpt, N_HLayer=N_HLayer, N_neu=N_neu, N_f=N_train)
    PINN.evaluate(t_star, x_star, y_star, u_star, v_star)
    next_ic = PINN.prepare_next_ic(t_list[time_range[1]], x_star, y_star)


if __name__ == "__main__":
    net_params = None
    net_params_1 = None
    #net_params = './simulation_05272023/results/Re100/4x120_Nf15k_lamB10_alpha0.05/model_cavity_loop100000.pth'
    train(net_params=net_params, net_params_1=net_params_1)
