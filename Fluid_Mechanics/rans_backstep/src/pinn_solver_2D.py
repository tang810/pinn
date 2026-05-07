import os
import torch
import scipy.io
import numpy as np
import math

import matplotlib.pyplot as plt

from src.net import FCNet
from src.net import ChebyKAN
from src.net import ChebyKAN_1
from tqdm.auto import tqdm
from typing import Dict, List, Set, Optional, Union, Callable
device = torch.device("cpu")

from torch.utils.tensorboard import SummaryWriter# Create an instance of the object 


def tonp(tensor): #将torch张量转换为numpy数组
    """ Torch to Numpy """
    if isinstance(tensor, torch.Tensor):   #将torch.Tensor从GPU转移到CPU上
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or '\
            'np.ndarray, but got {}'.format(type(input)))

class PysicsInformedNeuralNetwork:
    # Initialize the class
    def __init__(self,
                 device="cpu", 
                 lr=1e-3,
                 opt=None,
                 nu_0 = 1e-5,
                 U_0 = 1.0,
                 rho_0 = 1.0,
                 L_0 = 1.0,
                 layers=6,
                 layers_1=6,
                 hidden_size=80,
                 hidden_size_1=20,
                 N_f = 40000,
                 alpha_evm=0.03,
                 beta_evm=100.0,
                 learning_rate=0.001,
                 weight_decay=0.9,
                 outlet_weight=1,
                 bc_weight=1,
                 eq_weight=1,
                 ic_weight=1,
                 num_ins=2,
                 num_outs=3,
                 num_outs_1=1,
                 supervised_data_weight=1,
                 net_params=None,
                 net_params_1=None,
                 checkpoint_freq=2000,
                 checkpoint_path='./checkpoint/'):

        self.nu_0 = nu_0
        self.U_0 = U_0
        self.rho_0 = rho_0
        self.L_0 = L_0
        self.device = device
        self.lr = lr
        self.my_max = 1e7
        self.my_eps = 1e-7
        # SST constants
        self.sigma_k1 = 0.85
        self.sigma_o1 = 0.5
        self.beta_1 = 0.075

        self.sigma_k2 = 1.00
        self.sigma_o2 = 0.865
        self.beta_2 = 0.0828

        self.beta_star = 0.09
        self.a1 = 0.31

        self.alpha_1 = 5.0/9.0
        self.alpha_2 = 0.44

        self.evm = None
        self.vis_t0 = 5.0*self.nu_0

        self.layers = layers
        self.layers_1 = layers_1
        self.hidden_size = hidden_size
        self.hidden_size_1 = hidden_size_1
        self.N_f = N_f

        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_path = checkpoint_path

        self.alpha_evm = alpha_evm
        self.beta_evm = beta_evm

        self.alpha_b = bc_weight
        self.alpha_e = eq_weight
        self.alpha_i = ic_weight
        self.alpha_o = outlet_weight
        self.alpha_s = supervised_data_weight
        self.loss_i = self.loss_o = self.loss_b = self.loss_e = self.loss_s = 0.0
        
        self.L2_u = []
        self.L2_v = []
        self.loss_e1 = []
        self.loss_e2 = []
        self.loss_e3 = []
        self.loss_e4 = []
        self.loss_io = []
        self.loss_p = []
        self.loss_wall = []


        self.net = self.initialize_NN(
                 num_ins=num_ins, num_outs=num_outs, num_layers=layers, hidden_size=hidden_size).to(device)
        self.net_1 = self.initialize_NN(
                 num_ins=num_ins, num_outs=num_outs_1, num_layers=layers_1, hidden_size=hidden_size_1).to(device)
        if net_params:
            load_params = torch.load(net_params)
            self.net.load_state_dict(load_params)
#
        if net_params_1:
            load_params_1 = torch.load(net_params_1)
            self.net_1.load_state_dict(load_params_1)

        self.opt = torch.optim.Adam(
            list(self.net.parameters())+list(self.net_1.parameters()),
            lr=learning_rate,
            weight_decay=0.0) if not opt else opt

    def set_d_data(self):
        #self.inv_d = torch.tensor(1.0/(np.clip(np.abs(self.y_f), self.my_eps, self.my_max)), requires_grad=True).float().to(device)
        self.inv_d = 1.0/(np.abs(self.y_f.detach().cpu().numpy().clip(min=1e-3)))

    def init_rans_parameters(self):
        (_,_,_,e) = self.neural_net_u(self.x_f, self.y_f)
        self.nu_u      = self.alpha_evm*np.abs(e.detach().cpu().numpy())

    def set_sym_boundary_data(self, X=None, time=False):
        # symmetric boundary training data | v, x, y
        requires_grad = False
        self.xb_up = torch.tensor(X[0], requires_grad=requires_grad).float().to(device)
        self.yb_up = torch.tensor(X[1], requires_grad=requires_grad).float().to(device)
        self.vb_up = torch.tensor(X[2], requires_grad=requires_grad).float().to(device)

    def set_io_boundary_data(self, X=None, time=False):
        # io boundary training data |u, v, x, y
        requires_grad = False
        self.xb_io = torch.tensor(X[0], requires_grad=requires_grad).float().to(device)
        self.yb_io = torch.tensor(X[1], requires_grad=requires_grad).float().to(device)
        self.ub_io = torch.tensor(X[2], requires_grad=requires_grad).float().to(device)
        self.vb_io = torch.tensor(X[3], requires_grad=requires_grad).float().to(device)
        self.kb_io = torch.tensor(X[4], requires_grad=requires_grad).float().to(device)
        self.ob_io = torch.tensor(X[5], requires_grad=requires_grad).float().to(device)
        self.nb_io = torch.tensor(X[6], requires_grad=requires_grad).float().to(device)

    def set_pres_boundary_data(self, X=None, time=False):
        # io boundary training data |p, x, y
        requires_grad = False
        self.xb_p = torch.tensor(X[0], requires_grad=requires_grad).float().to(device)
        self.yb_p = torch.tensor(X[1], requires_grad=requires_grad).float().to(device)
        self.pb_p = torch.tensor(X[2], requires_grad=requires_grad).float().to(device)

    def set_ko_boundary_data(self, X=None, time=False):
        # ko boundary training data |k, omega, x, y
        requires_grad = False
        self.xb_ko = torch.tensor(X[0], requires_grad=requires_grad).float().to(device)
        self.yb_ko = torch.tensor(X[1], requires_grad=requires_grad).float().to(device)
        self.kb_ko = torch.tensor(X[2], requires_grad=requires_grad).float().to(device)
        self.ob_ko = torch.tensor(X[3], requires_grad=requires_grad).float().to(device)

    def set_wall_boundary_data(self, X=None, time=False):
        # ko boundary training data |k, omega, x, y
        requires_grad = False
        self.xb_wall = torch.tensor(X[0], requires_grad=requires_grad).float().to(device)
        self.yb_wall = torch.tensor(X[1], requires_grad=requires_grad).float().to(device)
        self.ub_wall = torch.tensor(X[2], requires_grad=requires_grad).float().to(device)
        self.vb_wall = torch.tensor(X[3], requires_grad=requires_grad).float().to(device)
        self.kb_wall = torch.tensor(X[4], requires_grad=requires_grad).float().to(device)
        
    def set_eq_training_data(self,
                             X=None,
                             time=False):
        requires_grad = True
        self.x_f = torch.tensor(X[0], requires_grad=requires_grad).float().to(device)
        self.y_f = torch.tensor(X[1], requires_grad=requires_grad).float().to(device)
        if time:
            self.t_f = torch.tensor(X[2], requires_grad=requires_grad).float().to(device)

        self.set_d_data()
        self.init_rans_parameters()

    def set_testing_data(self,
                         x_star=None,
                         y_star=None,
                         u_star=None,
                         v_star=None,
                         p_star=None,
                         k_star=None,
                         o_star=None,
                         time=False):
        requires_grad = False
        self.x_star = torch.tensor(x_star, requires_grad=requires_grad).float().to(device)
        self.y_star = torch.tensor(y_star, requires_grad=requires_grad).float().to(device)
        self.u_star = torch.tensor(u_star, requires_grad=requires_grad).float().to(device)
        self.v_star = torch.tensor(v_star, requires_grad=requires_grad).float().to(device)
        self.p_star = torch.tensor(p_star, requires_grad=requires_grad).float().to(device)
        self.k_star = torch.tensor(k_star, requires_grad=requires_grad).float().to(device)
        self.o_star = torch.tensor(o_star, requires_grad=requires_grad).float().to(device)


    def set_optimizers(self, opt):
        self.opt = opt

    def set_alpha_evm(self, alpha):
        self.alpha_evm = alpha

    def initialize_NN(self,
                       num_ins=2,
                       num_outs=4,
                       num_layers=10,
                       hidden_size=50):
         return FCNet(num_ins=num_ins,
                      num_outs=num_outs,
                      num_layers=num_layers,
                      hidden_size=hidden_size,
                      activation=torch.nn.Tanh)

    def set_eq_training_func(self, train_data_func):
        self.train_data_func = train_data_func

    def set_equation_parameters(self):
        X = torch.cat((self.x_f, self.y_f), dim=1)
        uvp = self.net(X)
        ko = self.net_1(X)

        u = uvp[:, 0:1]
        v = uvp[:, 1:2]
        p = uvp[:, 2:3]
        e = ko[:, 0:1]
#        e = ko[:, 1:2]

        self.nu_u_torch = self.alpha_evm*(e)

        self.nu_u = self.nu_u_torch.detach().cpu().numpy().clip(min=0.0, max=self.beta_evm*self.nu_0)
       
#        self.nu_k = self.nu_u

 
        u_x, u_y = self.autograd(u, [self.x_f,self.y_f])

        v_x, v_y = self.autograd(v, [self.x_f,self.y_f])

        nu_x, nu_y = self.autograd(self.nu_u_torch, [self.x_f,self.y_f])

 
        self.nu_ux = nu_x.detach().cpu().numpy().clip(min=-100*self.nu_u,max=100*self.nu_u)
        self.nu_uy = nu_y.detach().cpu().numpy().clip(min=-100*self.nu_u,max=100*self.nu_u)




    def neural_net_u(self, x, y):
        X = torch.cat((x, y), dim=1)
        uvp = self.net(X)
        ee = self.net_1(X)
        u = uvp[:, 0:1]
        v = uvp[:, 1:2]
        p = uvp[:, 2:3]
        e =  ee[:, 0:1]
#        o =  ee[:, 1:2]
        return u, v, p, e

    def neural_net_equations(self, x, y):
        X = torch.cat((x, y), dim=1)
        uvp = self.net(X)
        ko = self.net_1(X)

        u = uvp[:, 0:1]
        v = uvp[:, 1:2]
        p = uvp[:, 2:3]
        e = ko[:, 0:1]
#        e = ko[:, 1:2]

        u_x, u_y = self.autograd(u, [x,y])
        u_xx = self.autograd(u_x, [x])[0]
        u_yy = self.autograd(u_y, [y])[0]

        v_x, v_y = self.autograd(v, [x,y])
        v_xx = self.autograd(v_x, [x])[0]
        v_yy = self.autograd(v_y, [y])[0]

        p_x, p_y = self.autograd(p, [x,y])



        self.nu_u_old = torch.from_numpy(self.nu_u).float().to(device)
#        self.nu_k_old = torch.from_numpy(self.nu_k).float().to(device)
        self.nu_ux_old = torch.from_numpy(self.nu_ux).float().to(device)
        self.nu_uy_old = torch.from_numpy(self.nu_uy).float().to(device)


        # NS
        eq1 = (u*u_x + v*u_y ) + p_x/self.rho_0 - (self.nu_0+self.nu_u_old)*(u_xx + u_yy) \
                 - (2.0*self.nu_ux_old*u_x+self.nu_uy_old*(u_y+v_x))

        eq2 = (u*v_x + v*v_y) + p_y/self.rho_0 - (self.nu_0+self.nu_u_old)*(v_xx + v_yy) \
                 - (self.nu_ux_old*(v_x+u_y) + 2.0*self.nu_uy_old*v_y)

        eq3 = u_x + v_y



        eq4 = (u-0.5)*eq1+ (v-0.5)*eq2 - e



        return eq1, eq2, eq3, eq4

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

    def predict(self, x_star, y_star, u_star, v_star, num_epoch=None, p_star=None, e_star=None):
        x_test = torch.tensor(x_star).float().to(device)
        y_test = torch.tensor(y_star).float().to(device)

        u_pred, v_pred, p_pred, e_pred = self.neural_net_u(x_test, y_test)

        # 转 numpy
        u_pred = tonp(u_pred)
        v_pred = tonp(v_pred)
        p_pred = tonp(p_pred)
        e_pred = tonp(e_pred)

        u_true = tonp(u_star)
        v_true = tonp(v_star)
        p_true = tonp(p_star) if p_star is not None else None
        e_true = tonp(e_star) if e_star is not None else None

        result_folder = 'result_uvpdata'
        os.makedirs(result_folder, exist_ok=True)

        # ====== 1. 训练原始数据图 ======
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        im1 = axs[0, 0].tricontourf(x_star.flatten(), y_star.flatten(), u_true.flatten(), 100, cmap="jet")
        fig.colorbar(im1, ax=axs[0, 0])
        axs[0, 0].set_title("Ground Truth U velocity")

        im2 = axs[0, 1].tricontourf(x_star.flatten(), y_star.flatten(), v_true.flatten(), 100, cmap="jet")
        fig.colorbar(im2, ax=axs[0, 1])
        axs[0, 1].set_title("Ground Truth V velocity")

        if p_true is not None:
            im3 = axs[1, 0].tricontourf(x_star.flatten(), y_star.flatten(), p_true.flatten(), 100, cmap="jet")
            fig.colorbar(im3, ax=axs[1, 0])
            axs[1, 0].set_title("Ground Truth Pressure")

        if e_true is not None:
            im4 = axs[1, 1].tricontourf(x_star.flatten(), y_star.flatten(), e_true.flatten(), 100, cmap="jet")
            fig.colorbar(im4, ax=axs[1, 1])
            axs[1, 1].set_title("Ground Truth Eddy Viscosity")

        for ax in axs.flat:
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        plt.tight_layout()
        fig_path = os.path.join(result_folder, f'truth_epoch{num_epoch}.png')
        plt.savefig(fig_path, dpi=300)
        plt.close(fig)

        # ====== 2. 预测结果图 ======
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        im1 = axs[0, 0].tricontourf(x_star.flatten(), y_star.flatten(), u_pred.flatten(), 100, cmap="jet")
        fig.colorbar(im1, ax=axs[0, 0])
        axs[0, 0].set_title("Predicted U velocity")

        im2 = axs[0, 1].tricontourf(x_star.flatten(), y_star.flatten(), v_pred.flatten(), 100, cmap="jet")
        fig.colorbar(im2, ax=axs[0, 1])
        axs[0, 1].set_title("Predicted V velocity")

        im3 = axs[1, 0].tricontourf(x_star.flatten(), y_star.flatten(), p_pred.flatten(), 100, cmap="jet")
        fig.colorbar(im3, ax=axs[1, 0])
        axs[1, 0].set_title("Predicted Pressure")

        im4 = axs[1, 1].tricontourf(x_star.flatten(), y_star.flatten(), e_pred.flatten(), 100, cmap="jet")
        fig.colorbar(im4, ax=axs[1, 1])
        axs[1, 1].set_title("Predicted Eddy Viscosity")

        for ax in axs.flat:
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        plt.tight_layout()
        fig_path = os.path.join(result_folder, f'prediction_epoch{num_epoch}.png')
        plt.savefig(fig_path, dpi=300)
        plt.close(fig)

        # ====== 3. 误差图（预测 - 真实） ======
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        im1 = axs[0, 0].tricontourf(x_star.flatten(), y_star.flatten(), (u_pred - u_true).flatten(), 100, cmap="seismic")
        fig.colorbar(im1, ax=axs[0, 0])
        axs[0, 0].set_title("U Error (Pred - Truth)")

        im2 = axs[0, 1].tricontourf(x_star.flatten(), y_star.flatten(), (v_pred - v_true).flatten(), 100, cmap="seismic")
        fig.colorbar(im2, ax=axs[0, 1])
        axs[0, 1].set_title("V Error (Pred - Truth)")

        if p_true is not None:
            im3 = axs[1, 0].tricontourf(x_star.flatten(), y_star.flatten(), (p_pred - p_true).flatten(), 100, cmap="seismic")
            fig.colorbar(im3, ax=axs[1, 0])
            axs[1, 0].set_title("P Error (Pred - Truth)")

        if e_true is not None:
            im4 = axs[1, 1].tricontourf(x_star.flatten(), y_star.flatten(), (e_pred - e_true).flatten(), 100, cmap="seismic")
            fig.colorbar(im4, ax=axs[1, 1])
            axs[1, 1].set_title("E Error (Pred - Truth)")

        for ax in axs.flat:
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        plt.tight_layout()
        fig_path = os.path.join(result_folder, f'error_epoch{num_epoch}.png')
        plt.savefig(fig_path, dpi=300)
        plt.close(fig)





    def shuffle_graph(self):
        self.nu_u_old = torch.from_numpy(self.nu_u).float().to(device)


    def shuffle(self, tensor):
        tensor_to_numpy = tensor.detach().cpu()
        shuffle_numpy = np.random.shuffle(tensor_to_numpy)
        return torch.tensor(tensor_to_numpy, requires_grad=True).float()

    def fwd_computing_loss_2d(self, loss_mode='MSE'):
        # symmetric boundary data

        (self.ub_io_pred, self.vb_io_pred, _, _) = self.neural_net_u(self.xb_io, self.yb_io)

        # BC loss
        if loss_mode == 'L2':
            self.loss_b_io = torch.norm((self.ub_io.reshape([-1]) - self.ub_io_pred.reshape([-1])), p=2) + \
                             torch.norm((self.vb_io.reshape([-1]) - self.vb_io_pred.reshape([-1])), p=2)                           
                          
        if loss_mode == 'MSE':
            self.loss_b_io = torch.mean(torch.square(self.ub_io.reshape([-1]) - self.ub_io_pred.reshape([-1]))) + \
                             torch.mean(torch.square(self.vb_io.reshape([-1]) - self.vb_io_pred.reshape([-1])))

        # pressure boundary data
        (_, _, self.pb_p_pred, _) = self.neural_net_u(self.xb_p, self.yb_p)

        # BC loss
        if loss_mode == 'L2':
            self.loss_b_p = torch.norm((self.pb_p.reshape([-1]) - self.pb_p_pred.reshape([-1])), p=2)
        if loss_mode == 'MSE':
            self.loss_b_p = torch.mean(torch.square(self.pb_p.reshape([-1]) - self.pb_p_pred.reshape([-1])))

        # wall boundary data
        (self.ub_wall_pred, self.vb_wall_pred, _, _) = self.neural_net_u(self.xb_wall, self.yb_wall)

        # BC loss
        if loss_mode == 'L2':
            self.loss_b_wall = torch.norm((self.ub_wall.reshape([-1]) - self.ub_wall_pred.reshape([-1])), p=2) + \
                               torch.norm((self.vb_wall.reshape([-1]) - self.vb_wall_pred.reshape([-1])), p=2) 
                          
        if loss_mode == 'MSE':
            self.loss_b_wall = torch.mean(torch.square(self.ub_wall.reshape([-1]) - self.ub_wall_pred.reshape([-1]))) + \
                               torch.mean(torch.square(self.vb_wall.reshape([-1]) - self.vb_wall_pred.reshape([-1])))
            
        self.loss_b = self.loss_b_io + self.loss_b_p + self.loss_b_wall
        # equation
        assert self.x_f is not None and self.y_f is not None

        (self.eq1_pred, self.eq2_pred,
         self.eq3_pred, self.eq4_pred) = self.neural_net_equations(self.x_f, self.y_f)
        if loss_mode == 'L2':
            self.loss_e = torch.norm(self.eq1_pred.reshape([-1]), p=2) + \
                          torch.norm(self.eq2_pred.reshape([-1]), p=2) + \
                          torch.norm(self.eq3_pred.reshape([-1]), p=2) + \
                          torch.norm(self.eq4_pred.reshape([-1]), p=2)
        if loss_mode == 'MSE':
            self.loss_eq1 = torch.mean(torch.square(self.eq1_pred.reshape([-1])))
            self.loss_eq2 = torch.mean(torch.square(self.eq2_pred.reshape([-1])))
            self.loss_eq3 = torch.mean(torch.square(self.eq3_pred.reshape([-1])))
            self.loss_eq4 = torch.mean(torch.square(self.eq4_pred.reshape([-1])))

            self.loss_e = self.loss_eq1+self.loss_eq2+self.loss_eq3 + 0.1*self.loss_eq4

        self.loss = self.alpha_b * self.loss_b + self.alpha_e * self.loss_e

        return self.loss, [self.loss_eq1, self.loss_eq2, self.loss_eq3, self.loss_eq4, \
                                    self.loss_b_io, self.loss_b_p, self.loss_b_wall]

    def train(self,
              start_epoch=0,
              num_epoch=1,
              lr=1e-4,
              optimizer=None,
              scheduler=None,
              batchsize=None):
        if self.opt is not None:
            self.opt.param_groups[0]['lr'] = lr
        else:
            self.opt = torch.optim.Adam(list(self.net.parameters())+list(self.net_1.parameters()), lr=lr)
        return self.solve_Adam(self.fwd_computing_loss_2d, start_epoch, num_epoch, batchsize, scheduler)

    def solve_Adam(self,
                   loss_func,
                   start_epoch,
                   num_epoch=1000,
                   batchsize=None,
                   scheduler=None):
        self.freeze_evm_net(0)
        self.set_equation_parameters()
        for epoch_id in range(start_epoch,num_epoch):
            # train evm net every 10000 step
            if epoch_id !=0 and epoch_id % 5000 == 0:
                self.defreeze_evm_net(epoch_id)
            if (epoch_id - 1) % 5000 == 0:
                self.freeze_evm_net(epoch_id)

            loss, losses = loss_func()

#            self.shuffle_graph()

            loss.backward()
            self.opt.step()
            self.opt.zero_grad()

            self.set_equation_parameters()
            if scheduler:
                scheduler.step()

            if epoch_id == 0 or (epoch_id + 1)%100 == 0:
                l2_u, l2_v = self.test()
               # l2_u = l2u.detach().cpu().numpy()
               # l2_v = l2v.detach().cpu().numpy()
                self.print_log(loss, losses, l2_u, l2_v, epoch_id, num_epoch)

            if epoch_id == 0 or (epoch_id)%10000 == 0:
                saved_ckpt = 'model_loop%d.pth'%(epoch_id)
                layers = self.layers
                hidden_size = self.hidden_size
                N_f = self.N_f




    def freeze_evm_net(self, epoch_id):
        for para in self.net_1.parameters():
            para.requires_grad = False
        self.opt.param_groups[0]['params'] = list(self.net.parameters())

    def defreeze_evm_net(self, epoch_id):
        for para in self.net_1.parameters():
            para.requires_grad = True
        self.opt.param_groups[0]['params'] = list(self.net.parameters())+list(self.net_1.parameters())


    def print_log(self, loss, losses, l2_u, l2_v, epoch_id, num_epoch):
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
              %(loss.detach().cpu().item()), 
              "eq1_loss: %.3e " %(losses[0].detach().cpu().item()),
              "eq2_loss: %.3e " %(losses[1].detach().cpu().item()),
              "eq3_loss: %.3e " %(losses[2].detach().cpu().item()),
              "eq4_loss: %.3e " %(losses[3].detach().cpu().item()),
              "bc_loss_io: %.3e" %(losses[4].detach().cpu().item()),
              "bc_loss_p: %.3e" %(losses[5].detach().cpu().item()),
              "bc_loss_wall: %.3e" %(losses[6].detach().cpu().item()),
              "l2_u: %.3e" %(l2_u.item()),
              "l2_v: %.3e" %(l2_v.item()))

        self.loss_e1.append(losses[0].detach().cpu().item())
        self.loss_e2.append(losses[1].detach().cpu().item())
        self.loss_e3.append(losses[2].detach().cpu().item())
        self.loss_e4.append(losses[3].detach().cpu().item())
        self.loss_io.append(losses[4].detach().cpu().item())
        self.loss_p.append(losses[5].detach().cpu().item())
        self.loss_wall.append(losses[6].detach().cpu().item())
        self.L2_u.append(l2_u.item())
        self.L2_v.append(l2_v.item())

    def test(self):
        """ testing all points in the domain """
        # Prediction
        u_pred, v_pred, _, _ = self.neural_net_u(self.x_star, self.y_star)
        u_pred = u_pred.detach().cpu().numpy().reshape(-1,1)
        v_pred = v_pred.detach().cpu().numpy().reshape(-1,1)
        u_star = self.u_star.detach().cpu().numpy().reshape(-1,1)
        v_star = self.v_star.detach().cpu().numpy().reshape(-1,1)
        # Error
        error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
        error_v = np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)

        self.L2_u.append(error_u)
        self.L2_v.append(error_v)

        return error_u, error_v

    def save(self, filename, directory=None, N_HLayer=None, N_neu=None, N_f=None):
        Re_folder = 'Re'+str(math.ceil(1.0/self.nu_0))
        NNsize = str(N_HLayer) + 'x' + str(N_neu) + '_Nf'+str(np.int32(N_f/1000)) + 'k'
        lambdas = 'lamB'+str(self.alpha_b) + '_alpha'+str(self.alpha_evm)

        relative_path = '/results/' +  Re_folder + '/' + NNsize + '_' + lambdas + '/'

        if not directory:
            directory = os.getcwd()
        save_results_to = directory + relative_path
        if not os.path.exists(save_results_to):
            os.makedirs(save_results_to)
        
        torch.save(self.net.state_dict(), save_results_to+filename)
        torch.save(self.net_1.state_dict(), save_results_to+filename+'_evm')
