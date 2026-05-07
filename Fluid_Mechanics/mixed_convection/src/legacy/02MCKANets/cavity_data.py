import os
import numpy as np
import scipy.io
from tools import LHSample
from tools import sort_pts



class DataLoader:
    def __init__(self, path=None, N_f=20000, N_b=1000):

        '''
        N_f: Num of residual points
        N_b: Num of boundary points
        '''
        self.N_b = N_b
        self.x_min = 0.0
        self.x_max = 1.0
        self.y_min = 0.0
        self.y_max = 1.0
        self.N_f = N_f # equation points
        self.pts_bc = None

    def loading_boundary_data(self):
        Nx = self.N_b
        Ny = self.N_b
        r_const = 10
        
        upper_x = np.random.uniform(self.x_min, self.x_max, Nx)
        lower_x = np.random.uniform(self.x_min, self.x_max, Nx)
        left_y = np.random.uniform(self.y_min, self.y_max, Ny)
        right_y = np.random.uniform(self.y_min, self.y_max, Ny)
        
        u_upper = 1 -  np.cosh(r_const*(upper_x-0.5)) / np.cosh(r_const*0.5)
        t_lower = np.sin(np.pi * lower_x)
        
        #  lower upper left right
        x_b = np.concatenate([lower_x,
                              upper_x,
                              self.x_min * np.ones_like(left_y),
                              self.x_max * np.ones_like(right_y)], 
                              axis=0).reshape([-1, 1])
        y_b = np.concatenate([self.y_min * np.ones_like(lower_x),
                              self.y_max * np.ones_like(upper_x),
                              left_y,
                              right_y],
                              axis=0).reshape([-1, 1])
        
        u_b = np.concatenate([np.zeros_like(lower_x),
                              u_upper,
                              np.zeros_like(left_y),
                              np.zeros_like(right_y)],
                              axis=0).reshape([-1, 1])

        v_b = np.zeros([u_b.shape[0]]).reshape([-1, 1])
        
        t_b = np.concatenate([t_lower,
                              np.zeros_like(upper_x),
                              np.zeros_like(left_y),
                              np.zeros_like(right_y)],
                              axis=0).reshape([-1, 1])

        self.pts_bc = np.hstack((x_b,y_b))
      
        N_train_bcs = x_b.shape[0]
        print('-----------------------------')
        print('N_train_bcs: ' + str(N_train_bcs) )
        print('N_train_equ: ' + str(self.N_f) )
        print('-----------------------------')     
        return x_b, y_b, u_b, v_b, t_b 

    def loading_training_data(self):
        xye = LHSample(2, [[self.x_min, self.x_max], [self.y_min, self.y_max]], self.N_f)
        if self.pts_bc is not None:
            xye_sorted, _ = sort_pts(xye, self.pts_bc)
        else:
            print("need to load boundary data first!")
            raise 
        x_train_f = xye_sorted[:, 0:1]
        y_train_f = xye_sorted[:, 1:2]
        return x_train_f, y_train_f

    def loading_evaluate_data(self, filename):
        """ preparing training data """
        data = scipy.io.loadmat(filename)
        x = data['X_ref']
        y = data['Y_ref']
        u = data['U_ref']
        v = data['V_ref']
        t = data['T_ref']
        x_star = x.reshape(-1,1)
        y_star = y.reshape(-1,1)
        u_star = u.reshape(-1,1)
        v_star = v.reshape(-1,1)
        t_star = t.reshape(-1,1)
        return x_star, y_star, u_star, v_star, t_star
    
