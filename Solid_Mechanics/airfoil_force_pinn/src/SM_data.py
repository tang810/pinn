import os
import numpy as np
import scipy.io
from scipy.stats import qmc


class DataLoader:
    def __init__(self, path=None, N_f=20000, N_b=1000):

        '''
        N_f: Num of residual points
        N_b: Num of boundary points
        '''
        self.N_b = N_b
        self.N_f = N_f # equation points

    def loading_boundary_data(self, filename):
        # boundary points
        data = scipy.io.loadmat(filename)
        # coordinate 
        bx = data['bx']
        by = data['by']
        bz = data['bz']
        lx = data['lx']
        ly = data['ly']
        lz = data['lz']
        ux = data['ux']
        uy = data['uy']
        uz = data['uz']
        b_u = data['b_u']
        b_v = data['b_v']
        b_w = data['b_w']
        l_u = data['l_u']
        l_v = data['l_v']
        l_w = data['l_w']
        u_u = data['u_u']
        u_v = data['u_v']
        u_w = data['u_w']
        
        x_b = np.concatenate([bx, lx, ux], axis=0).reshape([-1, 1])
        y_b = np.concatenate([by, ly, uy], axis=0).reshape([-1, 1])
        z_b = np.concatenate([bz, lz, uz], axis=0).reshape([-1, 1])
        
        x_u = np.concatenate([b_u, l_u, u_u], axis=0).reshape([-1, 1])
        y_v = np.concatenate([b_v, l_v, u_v], axis=0).reshape([-1, 1])
        z_w = np.concatenate([b_w, l_w, u_w], axis=0).reshape([-1, 1])
              
        N_train_bcs = x_b.shape[0]
        print('-----------------------------')
        print('N_train_bcs: ' + str(N_train_bcs))
        print('N_train_equ: ' + str(self.N_f))
        print('-----------------------------')     
        return x_b, y_b, z_b, x_u, y_v, z_w
    
    def loading_training_data(self, filename):
        data = scipy.io.loadmat(filename)
        x = data['x']    
        y = data['y']
        z = data['z']
        x_train_f = x
        y_train_f = y
        z_train_f = z
        return x_train_f, y_train_f, z_train_f

    def loading_evaluate_data(self, filename):
        """ preparing training data """
        data = scipy.io.loadmat(filename)
        x_star = data['x']
        y_star = data['y']
        z_star = data['z'] 
        u_star = data['u_ref']
        v_star = data['v_ref']
        w_star = data['w_ref']
        return x_star, y_star, z_star, u_star, v_star, w_star
