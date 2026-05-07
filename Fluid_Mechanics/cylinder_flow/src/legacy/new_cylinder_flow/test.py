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
import os
import vtk
import copy
import time
import torch
import numpy as np

#from pyevtk.hl import pointsToVTK

import cylinder_re100_data as data
import pinn_solver as psolver

import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.animation import FuncAnimation, FFMpegWriter
from IPython.display import HTML

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def predict_once_for_all(net_params=None, vtk_path='./vtk/', time_window=10):
    # Loading checkpoint state_dict params
    
    # 设置PINN网络参数
    vtk_filename = vtk_path + 'uvp_t_'
    net_params = net_params

    N_neu = 120
    N_neu_1 = 20
    lam_bcs = 10
    lam_equ = 10
    lam_ic = 10
    # N_b = 4096
    # N_train = 15000
    N_b = 163840 # 4096 x 40
    N_train = 15000

    alpha_evm = 0.03
    N_HLayer = 6
    N_HLayer_1 = 6

    # 加载数据
    filename = './datasets/cylinder_nektar_Re100.mat'
    dataloader = data.DataLoader(filename=filename,time_window=15)
    dataloader.sample_geometry(N_train=N_train, N_b=N_b)
    
    # Test data
    evaluate_data = dataloader.get_evaluate_data(snap=time_window)
    test_p, test_u, test_v, test_t, test_x, test_y = evaluate_data
    test_x = test_x.reshape(-1,time_window)
    test_y = test_y.reshape(-1,time_window)
    test_u = test_u.reshape(-1,time_window)
    test_v = test_v.reshape(-1,time_window)
    test_p = test_p.reshape(-1,time_window)
    print("Groundtruth: ", test_x.shape, test_y.shape, test_u.shape, test_v.shape, test_p.shape)

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

    boundary_data, outlet_data = dataloader.get_boundary_data()
    interior_data = dataloader.sample_interior_data(N_train=15000)

    o_p, o_t, o_x, o_y = outlet_data
    num_outlet_data = int(outlet_data[0].shape[0] / time_window)

    b_u, b_v, b_t, b_x, b_y = boundary_data
    num_boundary_data = int(b_t.shape[0] / time_window)

    t_t, t_x, t_y = interior_data
    num_interior_data = int(t_x.shape[0] / time_window)
    #start_time = time.perf_counter()


    for i in [t_x, t_y, t_t, o_x, o_y, o_t, b_x, b_y, b_t]:
        i = i.astype('float32')

    all_x = np.concatenate((t_x, o_x, b_x))
    all_y = np.concatenate((t_y, o_y, b_y))
    all_t = np.concatenate((t_t, o_t, b_t))

    tensor_t = torch.tensor(all_t).float().to(device)
    tensor_x = torch.tensor(all_x).float().to(device)
    tensor_y = torch.tensor(all_y).float().to(device)

    # start_time = time.perf_counter()
    u, v, p, e = PINN.predict(net_params, (tensor_t, tensor_x, tensor_y))
    # stop_time = time.perf_counter()
    # print('Spend %.3f seconds to predict 50 time steps.'%(stop_time - start_time))
    u = u.cpu().detach().numpy()
    v = v.cpu().detach().numpy()
    p = p.cpu().detach().numpy()
    
    

    if not os.path.exists(vtk_path):
        os.makedirs(vtk_path)

    vtk_x = all_x.reshape(-1,time_window)
    vtk_y = all_y.reshape(-1,time_window)
    vtk_z = np.zeros_like(vtk_x)
    #vtk_z = np.zeros_like(vtk_x, dtype="float32")
    vtk_t = all_t.reshape(-1,time_window)
    u = u.reshape(-1,time_window)
    v = v.reshape(-1,time_window)
    p = p.reshape(-1,time_window)
    print("Predict: ", vtk_x.shape, vtk_y.shape, u.shape, v.shape, p.shape)


    for i in range(time_window):
        filename = vtk_filename + str(i)
        pointsToVTK(
            filename,
            vtk_x[:,i].flatten().copy(),
            vtk_y[:,i].flatten().copy(),
            vtk_z[:,i].flatten().copy(),
            data={
                "u": u[:,i].copy(),
                "v": v[:,i].copy(),
                "p": p[:,i].copy()})
        
    return vtk_x, vtk_y, u, v, p, test_x, test_y, test_u, test_v, test_p

def test(net_params=None, vtk_filename='./vtk/uvp_t_', time_window=10):
    # Loading checkpoint state_dict params
    # N_neu = 120
    # lam_bcs = 10
    # lam_equ = 1
    # N_b = 1024
    # N_f = 15000

    
    N_neu = 120
    N_neu_1 = 20
    lam_bcs = 10
    lam_equ = 10
    lam_ic = 10
    # N_b = 4096
    # N_train = 15000
    N_b = 4096
    #N_b = 163840 # 4096 x 40
    N_f = 15000

    alpha_evm = 0.03
    N_HLayer = 6
    N_HLayer_1 = 6

    # alpha_evm = 0.03
    # N_HLayer = 4
    # layers = [2] + N_HLayer*[N_neu] + [4]
    
    # Loading data from openfoam
    filename = './datasets/cylinder_nektar_Re100.mat'
    dataloader = data.DataLoader(filename=filename,time_window=time_window)
    dataloader.sample_geometry(N_train=N_f, N_b=N_b)

    # PINN = psolver.PysicsInformedNeuralNetwork(
    #     Re=100,
    #     layers=N_HLayer,
    #     hidden_size = N_neu,
    #     bc_weight=lam_bcs,
    #     eq_weight=lam_equ,
    #     ic_weight=10,
    #     outlet_weight=lam_bcs,
    #     checkpoint_path='./checkpoint/',
    #     net_params=net_params)
    
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
        net_params=net_params)
        #net_params_1 = net_params_1)

    # Set initial data, | p, u, v, t, x, y
    initial_data = dataloader.get_initial_data()
    PINN.set_initial_data(X=initial_data)

    boundary_data, outlet_data = dataloader.get_boundary_data()
    # Set boundary data, | u, v, t, x, y
    PINN.set_boundary_data(X=boundary_data)
    # Set outlet data, | p, t, x, y
    PINN.set_outlet_data(X=outlet_data)

    # Set supervised data, | p, u, v, t, x, y
    #supervised_data = dataloader.loading_supervised_data(training_time_list)
    #PINN.set_supervised_data(X=supervised_data)

    # Test data
    evaluate_data = dataloader.get_evaluate_data(snap=time_window)
    test_p, test_u, test_v, test_t, test_x, test_y = evaluate_data

    tensor_t = torch.tensor(test_t).float().to(device)
    tensor_x = torch.tensor(test_x).float().to(device)
    tensor_y = torch.tensor(test_y).float().to(device)

    # start_time = time.perf_counter()
    u, v, p, e = PINN.predict(net_params, (tensor_t, tensor_x, tensor_y))
    # stop_time = time.perf_counter()
    # print('Spend %.3f seconds to predict 50 time steps.'%(stop_time - start_time))
    u = u.cpu().detach().numpy().reshape(-1,1)
    v = v.cpu().detach().numpy().reshape(-1,1)
    p = p.cpu().detach().numpy().reshape(-1,1)

    # Error
  #  error_u = np.linalg.norm(test_u-u,2)/np.linalg.norm(test_u,2)
  #  error_v = np.linalg.norm(test_v-v,2)/np.linalg.norm(test_v,2)
    xmin=-5
    xmax=10
    ymin=-3
    ymax=3
    error_u=0.0
    error_v=0.0
    error_p=0.0
    l2_u=0.0
    l2_v=0.0
    l2_p=0.0
    num=0
    for  i in range(len(tensor_x)):
        if tensor_x[i]>xmin and tensor_x[i]<xmax and tensor_y[i]>ymin and tensor_y[i]<ymax:
            error_u = error_u+(test_u[i]-u[i])*(test_u[i]-u[i])
            error_v = error_v+(test_v[i]-v[i])*(test_v[i]-v[i])
            error_p = error_p+(test_p[i]-p[i])*(test_p[i]-p[i])
            l2_u =l2_u+test_u[i]*test_u[i]
            l2_v =l2_v+test_v[i]*test_v[i]
            l2_p =l2_p+test_p[i]*test_p[i]
            num = num+1
   
    error_u=np.sqrt(error_u/num)
    error_v=np.sqrt(error_v/num)
    error_p=np.sqrt(error_p/num)
    l2_u=np.sqrt(l2_u/num)
    l2_v=np.sqrt(l2_v/num)
    l2_p=np.sqrt(l2_p/num)

    error_u = error_u/l2_u
    error_v = error_v/l2_v
    error_p = error_p/l2_p
           
    print('------------------------')
    print('Error u: %e' % (error_u))
    print('Error v: %e' % (error_v))
    print('Error p: %e' % (error_p))
    print('L2 u: %e' % (l2_u))
    print('L2 v: %e' % (l2_v))
    print('L2 p: %e' % (l2_p))
    print('------------------------')

fig, axs = plt.subplots(2, 3, figsize=(30, 10))

def plot_frame(frame, x, y, u, v, p, test_x, test_y, test_u, test_v, test_p):
    
    # 原始预测数据
    cur_x = x[:,frame].reshape(-1,1).flatten()
    cur_y = y[:,frame].reshape(-1,1).flatten()
    cur_u = u[:,frame].reshape(-1,1).flatten()
    cur_v = v[:,frame].reshape(-1,1).flatten()
    cur_p = p[:,frame].reshape(-1,1).flatten()
    
    # 原始groundtruth数据
    cur_test_x = test_x[:,frame].reshape(-1,1).flatten()
    cur_test_y = test_y[:,frame].reshape(-1,1).flatten()
    cur_test_u = test_u[:,frame].reshape(-1,1).flatten()
    cur_test_v = test_v[:,frame].reshape(-1,1).flatten()
    cur_test_p = test_p[:,frame].reshape(-1,1).flatten()
    
    # 创建网格
    grid_x, grid_y = np.meshgrid(np.linspace(cur_x.min(), cur_x.max(), 1000), np.linspace(cur_y.min(), cur_y.max(), 1000))
    
    grid_test_x, grid_test_y = np.meshgrid(np.linspace(cur_test_x.min(), cur_test_x.max(), 1000), np.linspace(cur_test_y.min(), cur_test_y.max(), 1000))
    
    # 插值到网格
    grid_u = griddata((cur_x, cur_y), cur_u, (grid_x, grid_y), method='nearest')
    grid_v = griddata((cur_x, cur_y), cur_v, (grid_x, grid_y), method='nearest')
    grid_p = griddata((cur_x, cur_y), cur_p, (grid_x, grid_y), method='nearest')
    
    grid_test_u = griddata((cur_test_x, cur_test_y), cur_test_u, (grid_test_x, grid_test_y), method='nearest')
    grid_test_v = griddata((cur_test_x, cur_test_y), cur_test_v, (grid_test_x, grid_test_y), method='nearest')
    grid_test_p = griddata((cur_test_x, cur_test_y), cur_test_p, (grid_test_x, grid_test_y), method='nearest')
    
    # d = data[frame]
    # d = zoom(d, (1, 0.4 , 0.4), order=1)

    axs[0][0].clear()
    axs[0][0].imshow(grid_u, extent=(cur_x.min(), cur_x.max(), cur_y.min(), cur_y.max()), cmap="jet", vmin=cur_u.min(), vmax=cur_u.max(),interpolation="catrom")
    axs[0][0].set_title("Predict Velocity Field $U$")
    
    axs[0][1].clear()
    axs[0][1].imshow(grid_v, extent=(cur_x.min(), cur_x.max(), cur_y.min(), cur_y.max()), cmap="jet", vmin=cur_v.min(), vmax=cur_v.max(),interpolation="catrom")
    axs[0][1].set_title("Predict Velocity Field $V$")
    
    axs[0][2].clear()
    axs[0][2].imshow(grid_p, extent=(cur_x.min(), cur_x.max(), cur_y.min(), cur_y.max()), cmap="jet", vmin=cur_p.min(), vmax=cur_p.max(),interpolation="catrom")
    axs[0][2].set_title("Predict Pressure Field $P$")
    
    axs[1][0].clear()
    axs[1][0].imshow(grid_test_u, extent=(cur_test_x.min(), cur_test_x.max(), cur_test_y.min(), cur_test_y.max()), cmap="jet", vmin=cur_test_u.min(), vmax=cur_test_u.max(),interpolation="catrom")
    axs[1][0].set_title("Groundtruth Velocity Field $U$")
    
    axs[1][1].clear()
    axs[1][1].imshow(grid_test_v, extent=(cur_test_x.min(), cur_test_x.max(), cur_test_y.min(), cur_test_y.max()), cmap="jet", vmin=cur_test_v.min(), vmax=cur_test_v.max(),interpolation="catrom")
    axs[1][1].set_title("Groundtruth Velocity Field $V$")
    
    axs[1][2].clear()
    axs[1][2].imshow(grid_test_p, extent=(cur_test_x.min(), cur_test_x.max(), cur_test_y.min(), cur_test_y.max()), cmap="jet", vmin=cur_test_p.min(), vmax=cur_test_p.max(),interpolation="catrom")
    axs[1][2].set_title("Groundtruth Pressure Field $P$")
    
    
    # for i in range(3):
    #     axs[i].clear()
    #     # axs[i].imshow(d[i], cmap="jet", vmin=min_max[i][0], vmax=min_max[i][1])
    #     axs[i].imshow(
    #         d[i][150:150+64, 150:150+128] * mask[150:150+64, 150:150+128], cmap="jet",
    #         vmin=min_max[i][0], vmax=min_max[i][1],
    #         interpolation="catrom"
    #     )


if __name__ == "__main__":
    #net_params = None
    #net_params = 'results/Re100/6x120_Nf15k_lamB10_alpha0.03/model_cylinder_loop4.pth'
    #net_params_1 = 'results/Re100/6x120_Nf15k_lamB10_alpha0.03/model_cylinder_loop4.pth_evm'
    net_params = 'checkpoint/net_params_40000.pth'
    time_window = 10
    test(net_params=net_params, time_window=time_window)
    
    # x, y, u, v, p, test_x, test_y, test_u, test_v, test_p = predict_once_for_all(net_params=net_params, time_window=time_window)
    # # print(x.min(),x.max(),y.min(),y.max(),u.min(),u.max())
    
    # anim = FuncAnimation(fig, plot_frame, frames=100, fargs=(x, y, u, v, p, test_x, test_y, test_u, test_v, test_p) ,interval=50)
    # writervideo = FFMpegWriter(fps=60) 
    # anim.save('8.mp4', writer=writervideo) 
    
    # print(anim.to_html5_video())
    # HTML(anim.to_html5_video())
