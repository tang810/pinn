
import os
import numpy as np
import scipy.io
from src.tools import LHSample
from src.tools import sort_pts



# geometry
inlet_channel_length = (-2, 0)
inlet_channel_width = (1, 8)
outlet_channel_length = (0, 2)
outlet_channel_width = (0, 8)


class DataLoader:
    def __init__(self, path=None, N_f=20000, N_b=1000):

        '''
        N_f: Num of residual points
        N_b: Num of boundary points
        '''
        self.N_b = N_b
        self.x_min = 0.5
        self.x_max = 1.5
        self.y_min = 0.0
        self.y_max = 1.0
        self.N_f = N_f # equation points
        self.pts_bc = None

    def loading_boundary_data(self, bc_UV, bc_P, nu_0):
        # boundary points
        Nx = 257
        Ny = 257
        dx = 1.0/(Nx-1)

        num_inlet = 224
        num_outlet = 256

#
        num_wall_1 = 257
        yb_wall_1 = np.empty(num_wall_1)
        yb_wall_1.fill(inlet_channel_width[1])
        yb_wall_1 = yb_wall_1.reshape([-1, 1]);
        xb_wall_1 = np.linspace(inlet_channel_length[0],outlet_channel_length[1] , num=num_wall_1).reshape([-1, 1]);
        #bottom wall on the outlet
        num_wall_2 = 257
        yb_wall_2 = np.empty(num_wall_2)
        yb_wall_2.fill(outlet_channel_width[0])
        yb_wall_2 = yb_wall_2.reshape([-1, 1]);
        xb_wall_2 = np.linspace(outlet_channel_length[0],outlet_channel_length[1] , num=num_wall_2).reshape([-1, 1]);

        #bottom wall on the inlet channel
        num_wall_3 = 257
        yb_wall_3 = np.empty(num_wall_3)
        yb_wall_3.fill(inlet_channel_width[0])
        yb_wall_3 = yb_wall_3.reshape([-1, 1]);
        xb_wall_3 = np.linspace(inlet_channel_length[0],inlet_channel_length[1] , num=num_wall_3).reshape([-1, 1]);

        num_wall_4 = 129
        xb_wall_4 = np.empty(num_wall_4)
        xb_wall_4.fill(inlet_channel_length[1])
        xb_wall_4 = xb_wall_4.reshape([-1, 1]);
        yb_wall_4 = np.linspace(inlet_channel_width[0],outlet_channel_length[0] , num=num_wall_4).reshape([-1, 1]);

        xb_wall = np.concatenate([xb_wall_1, xb_wall_2, xb_wall_3, xb_wall_4],
                              axis=0).reshape([-1, 1])
        yb_wall = np.concatenate([yb_wall_1, yb_wall_2, yb_wall_3, yb_wall_4],
                              axis=0).reshape([-1, 1])
        ub_wall = np.zeros([xb_wall.shape[0]]).reshape([-1, 1])
        vb_wall = np.zeros([xb_wall.shape[0]]).reshape([-1, 1])
        kb_wall = np.zeros([xb_wall.shape[0]]).reshape([-1, 1])
      
        data_UV = scipy.io.loadmat(bc_UV)
        xb_io = data_UV['X_data'].reshape([-1,1])
        yb_io = data_UV['Y_data'].reshape([-1,1])
        ub_io = data_UV['U_data'].reshape([-1,1])
        vb_io = data_UV['V_data'].reshape([-1,1])
        kb_io = data_UV['K_data'].reshape([-1,1])
        ob_io = data_UV['O_data'].reshape([-1,1])
        nb_io = data_UV['N_data'].reshape([-1,1])

        data_P = scipy.io.loadmat(bc_P)
        xb_p = data_P['X_data_p'].reshape([-1,1])
        yb_p = data_P['Y_data_p'].reshape([-1,1])
        pb_p = data_P['P_data'].reshape([-1,1])


        self.boundary_data = [(xb_wall,yb_wall,ub_wall, vb_wall,kb_wall),(xb_io,yb_io,ub_io,vb_io, kb_io, ob_io, nb_io),(xb_p,yb_p,pb_p)]

        return self.boundary_data

    def loading_training_data(self):
        inlet_channel = LHSample(2, [[-4, 4], [1, 8 ]], 60000)

        outlet_channel = LHSample(2, [[0, 4], [0, 1 ]], 20000)
#        y_train_f = xye_sorted[:, 1:2]
        x_train_1 = inlet_channel[:, 0:1]
        y_train_1 = inlet_channel[:, 1:2]
        x_train_2 = outlet_channel[:, 0:1]
        y_train_2 = outlet_channel[:, 1:2]

        x_train_f = np.concatenate([x_train_1, x_train_2], axis=0).reshape([-1, 1])
        y_train_f = np.concatenate([y_train_1, y_train_2], axis=0).reshape([-1, 1])

        print(np.shape(x_train_f));
        #print(y_train_f);

        return x_train_f, y_train_f

    def loading_evaluate_data(self, filename):
        """ preparing training data """
        data = scipy.io.loadmat(filename)
        x = data['X_ref']
        y = data['Y_ref']
        u = data['U_ref']
        v = data['V_ref']
        p = data['P_ref']
        k = data['K_ref']
        o = data['O_ref']
        n = data['N_ref']
        x_star = x.reshape(-1,1)
        y_star = y.reshape(-1,1)
        u_star = u.reshape(-1,1)
        v_star = v.reshape(-1,1)
        p_star = p.reshape(-1,1)
        k_star = k.reshape(-1,1)
        o_star = o.reshape(-1,1)
        n_star = n.reshape(-1,1)
        return x_star, y_star, u_star, v_star, p_star, k_star, o_star, n_star


