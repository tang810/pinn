import os
import torch
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
from .net import FCNet
from typing import Dict, List, Set, Optional, Union, Callable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PysicsInformedNeuralNetwork:
    # Initialize the class
    # training_type:  'unsupervised' | 'half-supervised'
    def __init__(self,
                 opt=None,
                 E = 100,
                 nu = 0.3,
                 layers=10,
                 learning_rate=0.001,
                 weight_decay=0.9,
                 outlet_weight=1,
                 bc_weight=1,
                 eq_weight=1,
                 ic_weight=1,
                 num_ins=3,
                 num_outs=3,
                 supervised_data_weight=1,
                 training_type='unsupervised',
                 net_params=None,
                 checkpoint_freq=2000,
                 checkpoint_path='./checkpoint/',
                 model_dir="./model",
                 results_dir="./results"):

        self.E = E
        self.nu = nu

        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_path = checkpoint_path
        self.model_dir = model_dir
        self.results_dir = results_dir
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        self.training_type = training_type
        self.alpha_b = bc_weight
        self.alpha_e = eq_weight
        self.alpha_i = ic_weight
        self.alpha_o = outlet_weight
        self.alpha_s = supervised_data_weight
        self.loss_i = self.loss_o = self.loss_b = self.loss_e = self.loss_s = 0.0

        # initialize NN
        self.net = self.initialize_NN(
                num_ins=num_ins, num_outs=num_outs, num_layers=layers).to(device)
        if net_params:
            load_params = torch.load(net_params)
            self.net.load_state_dict(load_params)

        self.opt = torch.optim.Adam(
            params=self.net.parameters(),
            lr=learning_rate,
            weight_decay=0) if not opt else opt

    def set_boundary_data(self, X=None, time=False):
        # boundary training data
        requires_grad = False
        
        self.x_b = torch.tensor(X[0], requires_grad=requires_grad).float().to(device)
        self.y_b = torch.tensor(X[1], requires_grad=requires_grad).float().to(device)
        self.z_b = torch.tensor(X[2], requires_grad=requires_grad).float().to(device)        
        self.x_u = torch.tensor(X[3], requires_grad=requires_grad).float().to(device)        
        self.y_v = torch.tensor(X[4], requires_grad=requires_grad).float().to(device)
        self.z_w = torch.tensor(X[5], requires_grad=requires_grad).float().to(device)        


    def set_eq_training_data(self, X=None, time=False):
        requires_grad = True
        
        self.x_f = torch.tensor(X[0], requires_grad=requires_grad).float().to(device)
        self.y_f = torch.tensor(X[1], requires_grad=requires_grad).float().to(device)
        self.z_f = torch.tensor(X[2], dtype=torch.float32, requires_grad=requires_grad).to(device)


    def set_optimizers(self, opt):
        self.opt = opt

    def initialize_NN(self,
                      num_ins=3,
                      num_outs=3,
                      num_layers=10,
                      hidden_size=50):
        
        return FCNet(num_ins=num_ins,
                     num_outs=num_outs,
                     num_layers=num_layers,
                     hidden_size=hidden_size,
                     activation=torch.nn.Tanh)
 
    def set_eq_training_func(self, train_data_func):
        self.train_data_func = train_data_func

    def neural_net_u(self, x, y, z):
        X = torch.cat((x, y, z), dim=1)               
        uvw = self.net(X)
        u = uvw[:, 0:1]
        v = uvw[:, 1:2]
        w = uvw[:, 2:3]
        return u, v, w
    
    def neural_net_equations(self, x, y, z):
        X = torch.cat((x, y, z), dim=1)
        uvw = self.net(X)
        u = uvw[:, 0:1]
        v = uvw[:, 1:2]
        w = uvw[:, 2:3]

        u_x, u_y, u_z = self.autograd(u, [x, y, z])
        u_xx = self.autograd(u_x, [x])[0]
        u_xy = self.autograd(u_x, [y])[0]
        u_xz = self.autograd(u_x, [z])[0]
        u_yy = self.autograd(u_y, [y])[0]
        u_zz = self.autograd(u_z, [z])[0]

        v_x, v_y, v_z = self.autograd(v, [x, y, z])
        v_xx = self.autograd(v_x, [x])[0]
        v_xy = self.autograd(v_x, [y])[0]
        v_yy = self.autograd(v_y, [y])[0]
        v_yz = self.autograd(v_y, [z])[0]
        v_zz = self.autograd(v_z, [z])[0]

        w_x, w_y, w_z = self.autograd(w, [x, y, z])
        w_xx = self.autograd(w_x, [x])[0]
        w_xz = self.autograd(w_x, [z])[0]
        w_yy = self.autograd(w_y, [y])[0]
        w_yz = self.autograd(w_y, [z])[0]
        w_zz = self.autograd(w_z, [z])[0]
        
        G = ((self.E/(2*(1+self.nu))))                              #lame 
        la = (self.E * self.nu)/((1 + self.nu)*(1 - 2 * self.nu)) 
        
        # body force 
        f_1 = 0.1  #设置体力
        f_2 = 0.1  #设置体力
        f_3 = 0.1  #设置体力
        
        eq1 = ((2 * G + la) * ( u_xx ) + G * (u_yy + u_zz) + ( G + la )*(v_xy + w_xz) + f_1)    #loss1
        eq2 = ((2 * G + la) * ( v_yy ) + G * (v_xx + v_zz) + ( G + la )*(u_xy + w_yz) + f_2)    #loss2 
        eq3 = ((2 * G + la) * ( w_zz ) + G * (w_xx + w_yy) + ( G + la )*(u_xz + v_yz) + f_3)    #loss3                                                                        

        return eq1, eq2, eq3

    @torch.jit.script
    def autograd(y: torch.Tensor, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        TorchScript function to compute the gradient of a tensor wrt multople inputs
        """
        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(y, device=y.device)]
        grad = torch.autograd.grad(
            [
                y,
            ],
            x,
            grad_outputs=grad_outputs,
            create_graph=True,
            allow_unused=True,
            #retain_graph=True,
        )

        if grad is None:
            grad = [torch.zeros_like(xx) for xx in x]
        assert grad is not None
        grad = [g if g is not None else torch.zeros_like(x[i]) for i, g in enumerate(grad)]
        return grad

    def predict(self, net_params, X):
        x, y, z = X
        return self.neural_net_u(x, y, z)

    def shuffle(self, tensor):
        tensor_to_numpy = tensor.detach().cpu()
        return torch.tensor(tensor_to_numpy, requires_grad=True).float()

    def fwd_computing_loss_3d(self, loss_mode='MSE'):
        # boundary data
        (self.u_pred_b, self.v_pred_b, self.w_pred_b) = self.neural_net_u(self.x_b, self.y_b, self.z_b)
        
        # BC loss
        if loss_mode == 'L2':
            self.loss_b = torch.norm((self.x_u.reshape([-1]) - self.u_pred_b.reshape([-1])), p=2) + \
                          torch.norm((self.y_v.reshape([-1]) - self.v_pred_b.reshape([-1])), p=2) + \
                          torch.norm((self.z_w.reshape([-1]) - self.w_pred_b.reshape([-1])), p=2)
        if loss_mode == 'MSE':
            self.loss_b = torch.mean(torch.square(self.x_u.reshape([-1]) - self.u_pred_b.reshape([-1]))) + \
                          torch.mean(torch.square(self.y_v.reshape([-1]) - self.v_pred_b.reshape([-1]))) + \
                          torch.mean(torch.square(self.z_w.reshape([-1]) - self.w_pred_b.reshape([-1])))                              
        # equation
        assert self.x_f is not None and self.y_f is not None

        (self.eq11_pred, self.eq22_pred,
         self.eq33_pred) = self.neural_net_equations(self.x_f, self.y_f, self.z_f)
        if loss_mode == 'L2':
            self.loss_e = torch.norm(self.eq11_pred.reshape([-1]), p=2) + \
                          torch.norm(self.eq22_pred.reshape([-1]), p=2) + \
                          torch.norm(self.eq33_pred.reshape([-1]), p=2)
        if loss_mode == 'MSE':
            self.loss_e = torch.mean(torch.square(self.eq11_pred.reshape([-1]))) + \
                          torch.mean(torch.square(self.eq22_pred.reshape([-1]))) + \
                          torch.mean(torch.square(self.eq33_pred.reshape([-1])))

        self.loss = self.alpha_b * self.loss_b + self.alpha_e * self.loss_e

        return self.loss, [self.loss_e, self.loss_b]

    def train(self,
              num_epoch=1,
              lr=1e-4,
              optimizer=None,
              scheduler=None,
              batchsize=None):
        if optimizer is not None:
            self.opt = optimizer
        else:
            self.opt = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        return self.solve_Adam(self.fwd_computing_loss_3d, num_epoch, batchsize, scheduler)

    def solve_Adam(self,
                   loss_func,
                   num_epoch=1000,
                   batchsize=None,
                   scheduler=None):
        for epoch_id in range(num_epoch):
            loss, losses = loss_func()
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
            if scheduler:
                scheduler.step()

            if epoch_id == 0 or (epoch_id + 1)%100 == 0:
                self.print_log(loss, losses, epoch_id, num_epoch)

    def print_log(self, loss, losses, epoch_id, num_epoch):
        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']

        print("current lr is {}".format(get_lr(self.opt)))
        if isinstance(losses[0], int):
            eq_loss = losses[0]
        else:
            eq_loss = losses[0].detach().cpu().item()

        print("epoch/num_epoch: ", epoch_id + 1, "/", num_epoch,
              "loss[Adam]: %.3e"
              %(loss.detach().cpu().item()), "eq_loss: %.3e " %(eq_loss), "bc_loss: %.3e"
              %(losses[1].detach().cpu().item())) 
        
    def evaluate(self, x, y, z, u, v, w, plot_name="result.png"):
        """ testing all points in the domain """
        x_test = x.reshape(-1,1)
        y_test = y.reshape(-1,1)
        z_test = z.reshape(-1,1) 
        u_test = u.reshape(-1,1)
        v_test = v.reshape(-1,1)
        w_test = w.reshape(-1,1)

        x_test = torch.tensor(x_test).float().to(device)
        y_test = torch.tensor(y_test).float().to(device)
        z_test = torch.tensor(z_test).float().to(device)
        u_pred, v_pred, w_pred = self.neural_net_u(x_test, y_test, z_test)

        x_test = x_test.detach().cpu().numpy().reshape(-1,1)
        y_test = y_test.detach().cpu().numpy().reshape(-1,1)
        z_test = z_test.detach().cpu().numpy().reshape(-1,1)

        u_pred = u_pred.detach().cpu().numpy().reshape(-1,1)
        v_pred = v_pred.detach().cpu().numpy().reshape(-1,1)
        w_pred = w_pred.detach().cpu().numpy().reshape(-1,1)
        # Error
        error_u = np.linalg.norm(u_test-u_pred,2)/np.linalg.norm(u_test,2)
        error_v = np.linalg.norm(v_test-v_pred,2)/np.linalg.norm(v_test,2)
        error_w = np.linalg.norm(w_test-w_pred,2)/np.linalg.norm(w_test,2)
        print('------------------------')
        print('Error u: %e' % (error_u))
        print('Error v: %e' % (error_v))
        print('Error w: %e' % (error_w))
        print('------------------------')

        X_ref = x_test
        Y_ref = y_test
        Z_ref = z_test
        u_ref = u_test
        v_ref = v_test
        w_ref = w_test

        u_pred_4 = u_pred
        v_pred_4 = v_pred
        w_pred_4 = w_pred

        # Calculate vmin and vmax for color scales
        vmax_u = np.max(u_ref)
        vmin_u = np.min(u_ref)
        vmax_v = np.max(v_ref)
        vmin_v = np.min(v_ref)
        vmax_w = np.max(w_ref)
        vmin_w = np.min(w_ref)

        vmax_u_4 = np.max(u_pred_4)
        vmin_u_4 = np.min(u_pred_4)
        vmax_v_4 = np.max(v_pred_4)
        vmin_v_4 = np.min(v_pred_4)
        vmax_w_4 = np.max(w_pred_4)
        vmin_w_4 = np.min(w_pred_4)

        vlim_u = np.linspace(vmin_u, vmax_u, 10)
        vlim_v = np.linspace(vmin_v, vmax_v, 10)
        vlim_w = np.linspace(vmin_w, vmax_w, 10)
        vlim_u_4 = np.linspace(vmin_u_4, vmax_u_4, 10)
        vlim_v_4 = np.linspace(vmin_v_4, vmax_v_4, 10)
        vlim_w_4 = np.linspace(vmin_w_4, vmax_w_4, 10)

        font = 'Times New Roman'
        fig = plt.figure(figsize=(20, 10), facecolor='white')

        # Plot u_ref
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        scatter1 = ax1.scatter(Y_ref, X_ref, Z_ref, c=u_ref, cmap='jet')
        ax1.set_xlabel('y [m]', fontname=font, fontsize=14)
        ax1.set_ylabel('x [m]', fontname=font, fontsize=14)
        ax1.set_zlabel('z [m]', fontname=font, fontsize=14)
        ax1.set_title('U_ref', fontname=font, fontsize=20, fontweight='bold')
        ax1.set_box_aspect([15, 5, 5])

        # Plot v_ref
        ax2 = fig.add_subplot(2, 3, 2, projection='3d')
        scatter2 = ax2.scatter(Y_ref, X_ref, Z_ref, c=v_ref, cmap='jet')
        ax2.set_xlabel('y [m]', fontname=font, fontsize=14)
        ax2.set_ylabel('x [m]', fontname=font, fontsize=14)
        ax2.set_zlabel('z [m]', fontname=font, fontsize=14)
        ax2.set_title('V_ref', fontname=font, fontsize=20, fontweight='bold')
        ax2.set_box_aspect([15, 5, 5])

        # Plot w_ref
        ax3 = fig.add_subplot(2, 3, 3, projection='3d')
        scatter3 = ax3.scatter(Y_ref, X_ref, Z_ref, c=w_ref, cmap='jet')
        ax3.set_xlabel('y [m]', fontname=font, fontsize=14)
        ax3.set_ylabel('x [m]', fontname=font, fontsize=14)
        ax3.set_zlabel('z [m]', fontname=font, fontsize=14)
        ax3.set_title('W_ref', fontname=font, fontsize=20, fontweight='bold')
        ax3.set_box_aspect([15, 5, 5])

        # Plot u_pred_4
        ax4 = fig.add_subplot(2, 3, 4, projection='3d')
        scatter4 = ax4.scatter(Y_ref, X_ref, Z_ref, c=u_pred_4, cmap='jet')
        ax4.set_xlabel('y [m]', fontname=font, fontsize=14)
        ax4.set_ylabel('x [m]', fontname=font, fontsize=14)
        ax4.set_zlabel('z [m]', fontname=font, fontsize=14)
        ax4.set_title('U_PINN', fontname=font, fontsize=20, fontweight='bold')
        ax4.set_box_aspect([15, 5, 5])

        # Plot v_pred_4
        ax5 = fig.add_subplot(2, 3, 5, projection='3d')
        scatter5 = ax5.scatter(Y_ref, X_ref, Z_ref, c=v_pred_4, cmap='jet')
        ax5.set_xlabel('y [m]', fontname=font, fontsize=14)
        ax5.set_ylabel('x [m]', fontname=font, fontsize=14)
        ax5.set_zlabel('z [m]', fontname=font, fontsize=14)
        ax5.set_title('V_PINN', fontname=font, fontsize=20, fontweight='bold')
        ax5.set_box_aspect([15, 5, 5])

        # Plot w_pred_4
        ax6 = fig.add_subplot(2, 3, 6, projection='3d')
        scatter6 = ax6.scatter(Y_ref, X_ref, Z_ref, c=w_pred_4, cmap='jet')

        ax6.set_xlabel('y [m]', fontname=font, fontsize=14)
        ax6.set_ylabel('x [m]', fontname=font, fontsize=14)
        ax6.set_zlabel('z [m]', fontname=font, fontsize=14)
        ax6.set_title('W_PINN', fontname=font, fontsize=20, fontweight='bold')
        ax6.set_box_aspect([15, 5, 5])

        plt.suptitle('Results', fontname=font, fontsize=20, fontweight='bold')
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        plt.savefig(os.path.join(self.results_dir, plot_name), dpi=300)
        plt.close()


    def test(self, x, y, z, u, v, w, num_epoch, loop=None):
        """ testing all points in the domain """
        x_test = x.reshape(-1,1)
        y_test = y.reshape(-1,1)
        z_test = z.reshape(-1,1)
        u_test = u.reshape(-1,1)
        v_test = v.reshape(-1,1)
        w_test = w.reshape(-1,1)
        # Prediction
        x_test = torch.tensor(x_test).float().to(device)
        y_test = torch.tensor(y_test).float().to(device)
        z_test = torch.tensor(z_test).float().to(device)
        
        u_pred, v_pred, w_pred = self.neural_net_u(x_test, y_test, z_test)
        u_pred = u_pred.detach().cpu().numpy().reshape(-1,1)
        v_pred = v_pred.detach().cpu().numpy().reshape(-1,1)
        w_pred = w_pred.detach().cpu().numpy().reshape(-1,1)
        # Error
        error_u = np.linalg.norm(u_test-u_pred,2)/np.linalg.norm(u_test,2)
        error_v = np.linalg.norm(v_test-v_pred,2)/np.linalg.norm(v_test,2)
        error_w = np.linalg.norm(w_test-w_pred,2)/np.linalg.norm(w_test,2)
        print('------------------------')
        print('Error u: %e' % (error_u))
        print('Error v: %e' % (error_v))
        print('Error w: %e' % (error_w))
        print('------------------------')

        scipy.io.savemat(os.path.join(self.results_dir, 'cavity_result_loop%d_epoch%d.mat'%(loop, num_epoch)),
                    {'Error_u':error_u,
                     'Error_v':error_v,
                     'Error_w':error_w,
                     'x':x_test,
                     'y':y_test,
                     'z':z_test,
                     'U_pred':u_pred,
                     'V_pred':v_pred,
                     'W_pred':w_pred,
                     'lam_bcs':self.alpha_b,
                     'lam_equ':self.alpha_e})

    def save(self, filename, directory=None, N_HLayer=None, N_neu=None, N_f=None):
        target_dir = directory if directory else self.model_dir
        os.makedirs(target_dir, exist_ok=True)
        torch.save(self.net.state_dict(), os.path.join(target_dir, filename))
