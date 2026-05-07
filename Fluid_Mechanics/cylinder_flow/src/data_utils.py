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
import math
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

from .geometry.primitives_2d import Line, Circle, Channel2D


# geometry
channel_length = (-7.5, 22.5)
channel_width = (-10, 10)
densy_channel_length = (-2.5, 5)
densy_channel_width = (-3, 3)
cylinder_center = (0, 0)
cylinder_radius = 0.5

class DataLoader:
    def __init__(self, filename=None, time_window=30):
        self.loading_data(filename, time_window=time_window)

    def loading_data(self, filename, time_window=30):
        assert filename is not None
        data = scipy.io.loadmat(filename) # 读取Matlab文件
        
        self.T = time_window
        self.U_star = data['U_star'] # N x 2 x T
        self.P_star = data['P_star'] # N x T
        self.t_star = data['t'][:time_window] # T x 1
        self.t_star_all = data['t'] # T x 1
        self.X_star = data['X_star'] # N x 2

        self.N = self.X_star.shape[0]
        #self.T = self.t_star.shape[0]

    def set_time_range(self, time_frames, time_range):
        self.T = time_frames
        start = time_range[0]
        stop = time_range[1]
        indexes = np.arange(start, stop+1)
        idx = np.random.choice(indexes, time_frames, replace=False)
        self.t_star = self.t_star_all[idx] # T x 1

        ptf = np.round(self.t_star.reshape(-1), decimals=1).tolist()
        print('time_frames:', ptf)

    def get_t_list(self):
        return self.t_star_all

    def get_initial_data(self):

        # Rearrange Data 
        XX = self.X_star[:,0:1] # N x (T=1)
        YY = self.X_star[:,1:2] # N x (T=1)
        TT = np.tile(self.t_star_all[0], (1,XX.shape[0])).T # N x T=0

        UU = self.U_star[:,0,0].reshape((-1,1)) # N x T=0
        VV = self.U_star[:,1,0].reshape((-1,1)) # N x T=0
        PP = self.P_star[:,0].reshape((-1,1)) # N x T=0

        return [PP, UU, VV, TT, XX, YY]

    def sample_geometry(self, N_train=10000, N_b=1024):
        channel = Channel2D(
                (channel_length[0], channel_width[0]),
                (channel_length[1], channel_width[1]),
        )
        densy_channel = Channel2D(
                (densy_channel_length[0], densy_channel_width[0]),
                (densy_channel_length[1], densy_channel_width[1]),
        )
        inlet = Line(
                (channel_length[0], channel_width[0]),
                (channel_length[0], channel_width[1]),
                normal=1,
        )
        outlet = Line(
                (channel_length[1], channel_width[0]),
                (channel_length[1], channel_width[1]),
                normal=1,
        )
        cylinder = Circle(cylinder_center, cylinder_radius)

        T = self.T
        inlet_data = inlet.sample_boundary(nr_points=N_b)
        inlet_x = np.tile(inlet_data['x'], (1,T)) # N x T
        inlet_y = np.tile(inlet_data['y'], (1,T)) # N x T
        inlet_t = np.tile(self.t_star, (1,inlet_x.shape[0])).T # N x T
        inlet_u = np.zeros_like(inlet_x)
        inlet_u[:,:] = 1
        inlet_v = np.zeros_like(inlet_x)
        
        outlet_data = outlet.sample_boundary(nr_points=N_b)
        outlet_x = np.tile(outlet_data['x'], (1,T)) # N x T
        outlet_y = np.tile(outlet_data['y'], (1,T)) # N x T
        outlet_t = np.tile(self.t_star, (1,outlet_x.shape[0])).T # N x T
        outlet_p = np.zeros_like(outlet_x)
        
        wall_data = channel.sample_boundary(nr_points=N_b)
        wall_x = np.tile(wall_data['x'], (1,T)) # N x T
        wall_y = np.tile(wall_data['y'], (1,T)) # N x T
        wall_t = np.tile(self.t_star, (1,wall_x.shape[0])).T # N x T
        wall_u = np.zeros_like(wall_x)
        wall_u[:,:] = 1
        wall_v = np.zeros_like(wall_x)

        cylinder_data = cylinder.sample_boundary(nr_points=N_b*2)
        cylinder_x = np.tile(cylinder_data['x'], (1,T)) # N x T
        cylinder_y = np.tile(cylinder_data['y'], (1,T)) # N x T
        cylinder_t = np.tile(self.t_star, (1, cylinder_x.shape[0])).T # N x T
        cylinder_u = np.zeros_like(cylinder_x)
        cylinder_v = np.zeros_like(cylinder_x)

        self.boundary_data = [(inlet_x, inlet_y, inlet_t, inlet_u, inlet_v),
                              (outlet_x, outlet_y, outlet_t, outlet_p),
                              (wall_x, wall_y, wall_t, wall_u, wall_v),
                              (cylinder_x, cylinder_y, cylinder_t, cylinder_u, cylinder_v)]

        volume_geo = channel - cylinder
        densy_volume_geo = densy_channel - cylinder
        training_data = volume_geo.sample_interior(nr_points=int(N_train/2), quasirandom=True)
        training_data_densy = densy_volume_geo.sample_interior(nr_points=N_train, quasirandom=True)
        for k in training_data.keys():
            training_data[k] = np.concatenate((training_data[k], training_data_densy[k]))
        self.tile_interior_data(training_data)

    def tile_interior_data(self, training_data):
        T = self.T
        training_x = np.tile(training_data['x'], (1,T)) # N x T
        training_y = np.tile(training_data['y'], (1,T)) # N x T
        training_t = np.tile(self.t_star, (1, training_x.shape[0])).T # N x T
        training_t = training_t.reshape((-1, 1))
        training_x = training_x.reshape((-1, 1))
        training_y = training_y.reshape((-1, 1))
        self.training_data = (training_t, training_x, training_y)

    def get_boundary_data(self):
        [(inlet_x, inlet_y, inlet_t, inlet_u, inlet_v),
         (outlet_x, outlet_y, outlet_t, outlet_p), 
         (wall_x, wall_y, wall_t, wall_u, wall_v),
         (cylinder_x, cylinder_y, cylinder_t, cylinder_u, cylinder_v)] = self.boundary_data

        b_x = np.concatenate((inlet_x, cylinder_x, wall_x)).reshape((-1, 1))
        b_y = np.concatenate((inlet_y, cylinder_y, wall_y)).reshape((-1, 1))
        b_t = np.concatenate((inlet_t, cylinder_t, wall_t)).reshape((-1, 1))
        b_u = np.concatenate((inlet_u, cylinder_u, wall_u)).reshape((-1, 1))
        b_v = np.concatenate((inlet_v, cylinder_v, wall_v)).reshape((-1, 1))

        o_x = outlet_x.reshape((-1, 1))
        o_y = outlet_y.reshape((-1, 1))
        o_t = outlet_t.reshape((-1, 1))
        o_p = outlet_p.reshape((-1, 1))

        self.boundary_data = ((b_u, b_v, b_t, b_x, b_y), (o_p, o_t, o_x, o_y))
        return self.boundary_data

    def get_training_data(self):
        return self.training_data

    def load_training_data(self, N_train):
        # Rearrange Data 
        N = self.N
        T = self.T
        XX = np.tile(self.X_star[:,0:1], (1,T)) # N x T
        YY = np.tile(self.X_star[:,1:2], (1,T)) # N x T
        TT = np.tile(self.t_star, (1,N)).T # N x T

        UU = self.U_star[:,0,:] # N x T
        VV = self.U_star[:,1,:] # N x T
        PP = self.P_star # N x T
        
        x = XX.flatten()[:,None] # NT x 1
        y = YY.flatten()[:,None] # NT x 1
        t = TT.flatten()[:,None] # NT x 1

        u = UU.flatten()[:,None] # NT x 1
        v = VV.flatten()[:,None] # NT x 1
        p = PP.flatten()[:,None] # NT x 1

        # Training Data    
        idx = np.random.choice(self.N*self.T, N_train, replace=False)
        x_train = x[idx,:]
        y_train = y[idx,:]
        t_train = t[idx,:]
        u_train = u[idx,:]
        v_train = v[idx,:]
        
        return x_train, y_train, t_train, u_train, v_train 

    def get_evaluate_data(self, snap=10):     
        x_star = self.X_star[:,0:1]
        x_star = np.tile(x_star, (1,snap)) # N x T
        y_star = self.X_star[:,1:2]
        y_star = np.tile(y_star, (1,snap)) # N x T
        t_star = np.tile(self.t_star_all[:snap], (1,x_star.shape[0])).T # N x T
                        
        x_star = x_star.reshape((-1,1))
        y_star = y_star.reshape((-1,1))
        t_star = t_star.reshape((-1,1))
        u_star = self.U_star[:,0,:snap].reshape((-1,1))
        v_star = self.U_star[:,1,:snap].reshape((-1,1))
        p_star = self.P_star[:,:snap].reshape((-1,1))

        return p_star, u_star, v_star, t_star, x_star, y_star

    def sample_interior_data(self, N_train=10000):     
        T = self.T
        channel = Channel2D(
                (channel_length[0], channel_width[0]),
                (channel_length[1], channel_width[1]),
        )
        densy_channel = Channel2D(
                (densy_channel_length[0], densy_channel_width[0]),
                (densy_channel_length[1], densy_channel_width[1]),
        )

        cylinder = Circle(cylinder_center, cylinder_radius)

        volume_geo = channel - cylinder
        densy_volume_geo = densy_channel - cylinder
        
        interior_data = volume_geo.sample_interior(nr_points=int(N_train/2), quasirandom=True)
        interior_data_densy = densy_volume_geo.sample_interior(nr_points=N_train, quasirandom=True)
        for k in interior_data.keys():
            interior_data[k] = np.concatenate((interior_data[k], interior_data_densy[k]))

        x = np.tile(interior_data['x'], (1,T)) # N x T
        y = np.tile(interior_data['y'], (1,T)) # N x T
        t = np.tile(self.t_star, (1, x.shape[0])).T # N x T
        t = t.reshape((-1, 1))
        x = x.reshape((-1, 1))
        y = y.reshape((-1, 1))
        return (t, x, y)
