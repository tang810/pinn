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
import torch
import numpy as np
from .net import FCNet
from tqdm.auto import tqdm
from typing import Dict, List, Set, Optional, Union, Callable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PysicsInformedNeuralNetwork:
    # Initialize the class
    # training_type:  'unsupervised' | 'half-supervised'
    def __init__(self,
                 opt=None,
                 Re=100,
                 layers=4,
                 layers_1=4,
                 hidden_size=120,
                 hidden_size_1=20,
                 alpha_evm=0.03,
                 learning_rate=0.001,
                 weight_decay=0.9,
                 outlet_weight=1,
                 bc_weight=1,
                 eq_weight=1,
                 ic_weight=1,
                 num_ins=3,
                 num_outs=3,
                 num_outs_1=1,
                 supervised_data_weight=1,
                 training_type='unsupervised',
                 net_params=None,
                 net_params_1=None,
                 checkpoint_freq=20000,
                 checkpoint_path='./checkpoint/'):

        self.Re = Re
        self.vis_t0 = 1.0/self.Re

        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_path = checkpoint_path

        self.training_type = training_type
        self.alpha_evm = alpha_evm
        self.alpha_b = bc_weight
        self.alpha_e = eq_weight
        self.alpha_i = ic_weight
        self.alpha_o = outlet_weight
        self.alpha_s = supervised_data_weight
        self.loss_i = self.loss_o = self.loss_b = self.loss_e = self.loss_s = 0.0
        self.epoch_id = 0

        # initialize NN
        self.net = self.initialize_NN(
                num_ins=num_ins, num_outs=num_outs, num_layers=layers,
                hidden_size=hidden_size).to(device)

        self.net_1 = self.initialize_NN(
                num_ins=num_ins, num_outs=num_outs_1, num_layers=layers_1, 
                hidden_size=hidden_size_1).to(device)

        if net_params:
            load_params = torch.load(net_params)
            self.net.load_state_dict(load_params)

        if net_params_1:
            load_params_1 = torch.load(net_params_1)
            self.net_1.load_state_dict(load_params_1)

        #self.count_parameters(self.net)
        #self.count_parameters(self.net_evm)

        self.opt = torch.optim.Adam(
            #list(self.net.parameters()),
            list(self.net.parameters())+list(self.net_1.parameters()),
            lr=learning_rate,
            weight_decay=0) if not opt else opt

    def count_parameters(self, model):
        from prettytable import PrettyTable
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params+=params
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params
                                                                
    def set_alpha_evm(self, alpha):
        self.alpha_evm = alpha

    def init_vis_t(self):
        (_, _,_, e) = self.neural_net_u(self.t_f, self.x_f, self.y_f)
        self.vis_t_minus  = self.alpha_evm*torch.abs(e).detach().cpu().numpy()

    def set_outlet_data(self, X=None, continuous_time=True):
        # p, t, x, y
        requires_grad = False
        self.p_o = torch.tensor(X[0], requires_grad=requires_grad).float().to(device)
        self.t_o = torch.tensor(X[1], requires_grad=requires_grad).float().to(device)
        self.x_o = torch.tensor(X[2], requires_grad=requires_grad).float().to(device)
        self.y_o = torch.tensor(X[3], requires_grad=requires_grad).float().to(device)

    def set_initial_data(self, X=None, continuous_time=True):
        # initial training data | u, v, x, y
        requires_grad = False
        self.p_i = torch.tensor(X[0], requires_grad=requires_grad).float().to(device)
        self.u_i = torch.tensor(X[1], requires_grad=requires_grad).float().to(device)
        self.v_i = torch.tensor(X[2], requires_grad=requires_grad).float().to(device)
        self.t_i = torch.tensor(X[3], requires_grad=requires_grad).float().to(device)
        self.x_i = torch.tensor(X[4], requires_grad=requires_grad).float().to(device)
        self.y_i = torch.tensor(X[5], requires_grad=requires_grad).float().to(device)

    def set_boundary_data(self, X=None, continuous_time=True):
        # boundary training data | u, v, t, x, y
        requires_grad = False
        self.u_b = torch.tensor(X[0], requires_grad=requires_grad).float().to(device)
        self.v_b = torch.tensor(X[1], requires_grad=requires_grad).float().to(device)
        self.t_b = torch.tensor(X[2], requires_grad=requires_grad).float().to(device)
        self.x_b = torch.tensor(X[3], requires_grad=requires_grad).float().to(device)
        self.y_b = torch.tensor(X[4], requires_grad=requires_grad).float().to(device)

    def set_supervised_data(self, X=None, continuous_time=True):
        # p, u, v, t, x, y
        requires_grad = False
        self.p_s = torch.tensor(X[0], requires_grad=requires_grad).float().to(device)
        self.u_s = torch.tensor(X[1], requires_grad=requires_grad).float().to(device)
        self.v_s = torch.tensor(X[2], requires_grad=requires_grad).float().to(device)
        self.t_s = torch.tensor(X[3], requires_grad=requires_grad).float().to(device)
        self.x_s = torch.tensor(X[4], requires_grad=requires_grad).float().to(device)
        self.y_s = torch.tensor(X[5], requires_grad=requires_grad).float().to(device)

    def set_eq_training_data(self, X=None):
        requires_grad = True
        self.t_f = torch.tensor(X[0], requires_grad=requires_grad).float().to(device)
        self.x_f = torch.tensor(X[1], requires_grad=requires_grad).float().to(device)
        self.y_f = torch.tensor(X[2], requires_grad=requires_grad).float().to(device)

        self.init_vis_t()

    def set_optimizers(self, opt):
        self.opt = opt

    def initialize_NN(self,
                      num_ins=3,
                      num_outs=4,
                      num_layers=10,
                      hidden_size=120):
        return FCNet(num_ins=num_ins,
                     num_outs=num_outs,
                     num_layers=num_layers,
                     hidden_size=hidden_size,
                     activation=torch.nn.Tanh)

    def neural_net_u(self, t, x, y):
        X = torch.cat((t, x, y), dim=1)
        uvp = self.net(X)
        ee = self.net_1(X)
        u = uvp[:, 0:1]
        v = uvp[:, 1:2]
        p = uvp[:, 2:3]
        e = ee[:, 0:1]
        return u, v, p, e

    def neural_net_equations(self, t, x, y):
        X = torch.cat((t, x, y), dim=1)
        uvp = self.net(X)
        ee = self.net_1(X)
        u = uvp[:, 0:1]
        v = uvp[:, 1:2]
        p = uvp[:, 2:3]
        e = ee[:, 0:1]

        u_t, u_x, u_y = self.autograd(u, [t,x,y])
        u_xx = self.autograd(u_x, [x])[0]
        u_yy = self.autograd(u_y, [y])[0]

        v_t, v_x, v_y = self.autograd(v, [t,x,y])
        v_xx = self.autograd(v_x, [x])[0]
        v_yy = self.autograd(v_y, [y])[0]

        p_x, p_y = self.autograd(p, [x,y])

        # Get the minum between (vis_t0, vis_t_mius(calculated with last step e))
        self.vis_t = torch.tensor(
                np.minimum(self.vis_t0, self.vis_t_minus)).float().to(device)
        # Save vis_t_minus for computing vis_t in the next step
        self.vis_t_minus  = self.alpha_evm*torch.abs(e).detach().cpu().numpy()
        # NS
        eq1 = (u * u_x + v * u_y) + p_x - (1.0/self.Re+self.vis_t).reshape([-1,1]) * (u_xx + u_yy) + u_t
        eq2 = (u * v_x + v * v_y) + p_y - (1.0/self.Re+self.vis_t).reshape([-1,1]) * (v_xx + v_yy) + v_t
        # Continuty
        eq3 = u_x + v_y

        residual = (eq1 * (u-0.5) + eq2 * (v-0.5))-e
        # return eq1, eq2, eq3, residual
        return eq1, eq2, eq3, residual

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
        t, x, y = X
        return self.neural_net_u(t, x, y)

    def fwd_computing_loss_2d(self, loss_mode='MSE'):
        # physics informed neural networks (inside the domain)
        # initial data
        (self.u_pred_i, self.v_pred_i,
         self.p_pred_i,_) = self.neural_net_u(self.t_i, self.x_i, self.y_i)

        # IC loss
        if loss_mode == 'L2':
            self.loss_i = torch.norm((self.u_i.reshape([-1]) - self.u_pred_i.reshape([-1])), p=2) + \
                          torch.norm((self.v_i.reshape([-1]) - self.v_pred_i.reshape([-1])), p=2) + \
                          torch.norm((self.p_i.reshape([-1]) - self.p_pred_i.reshape([-1])), p=2)
        if loss_mode == 'MSE':
            self.loss_i = torch.mean(torch.square(self.u_i.reshape([-1]) - self.u_pred_i.reshape([-1]))) + \
                          10*torch.mean(torch.square(self.v_i.reshape([-1]) - self.v_pred_i.reshape([-1]))) 
         #                 torch.mean(torch.square(self.p_i.reshape([-1]) - self.p_pred_i.reshape([-1])))

        # boundary data
        (self.u_pred_b, self.v_pred_b,
         self.p_pred_b,_) = self.neural_net_u(self.t_b, self.x_b, self.y_b)

        # BC loss
        if loss_mode == 'L2':
            self.loss_b = torch.norm((self.u_b.reshape([-1]) - self.u_pred_b.reshape([-1])), p=2) + \
                          torch.norm((self.v_b.reshape([-1]) - self.v_pred_b.reshape([-1])), p=2)
        if loss_mode == 'MSE':
            self.loss_b = torch.mean(torch.square(self.u_b.reshape([-1]) - self.u_pred_b.reshape([-1]))) + \
                          10*torch.mean(torch.square(self.v_b.reshape([-1]) - self.v_pred_b.reshape([-1])))

        # outlet data
        (self.u_pred_o, self.v_pred_o,
         self.p_pred_o, _) = self.neural_net_u(self.t_o, self.x_o, self.y_o)

        # outlet loss
        if loss_mode == 'L2':
            self.loss_o = torch.norm(
                (self.p_o.reshape([-1]) - self.p_pred_o.reshape([-1])), p=2)
        if loss_mode == 'MSE':
            self.loss_o = torch.mean(
                torch.square(
                    self.p_o.reshape([-1]) - self.p_pred_o.reshape([-1])))
        
        # supervised interior data
        '''
        (self.u_pred_s, self.v_pred_s,
         self.p_pred_s,_) = self.neural_net_u(self.t_s, self.x_s, self.y_s)
        # supervised data loss
        if loss_mode == 'L2':
            self.loss_s = torch.norm((self.u_s.reshape([-1]) - self.u_pred_s.reshape([-1])), p=2) + \
                          torch.norm((self.v_s.reshape([-1]) - self.v_pred_s.reshape([-1])), p=2)
        if loss_mode == 'MSE':
            self.loss_s = torch.mean(torch.square(self.u_s.reshape([-1]) - self.u_pred_s.reshape([-1]))) + \
                          torch.mean(torch.square(self.v_s.reshape([-1]) - self.v_pred_s.reshape([-1])))
        '''

        # equation
        (self.eq1_pred, self.eq2_pred,
         self.eq3_pred, self.eq4_pred) = self.neural_net_equations(self.t_f, self.x_f, self.y_f)
        # equation residual loss
        if loss_mode == 'L2':
            self.loss_e = torch.norm(self.eq1_pred.reshape([-1]), p=2) + \
                          torch.norm(self.eq2_pred.reshape([-1]), p=2) + \
                          torch.norm(self.eq3_pred.reshape([-1]), p=2) + \
                          torch.norm(self.eq4_pred.reshape([-1]), p=2)
        if loss_mode == 'MSE':
            #self.loss_e = torch.mean(torch.square(self.eq1_pred.reshape([-1]))) + \
            #              torch.mean(torch.square(self.eq2_pred.reshape([-1]))) + \
            #              torch.mean(torch.square(self.eq3_pred.reshape([-1]))) + \
            #              torch.mean(torch.square(self.eq4_pred.reshape([-1])))
            self.loss_eq1 = torch.mean(torch.square(self.eq1_pred.reshape([-1])))
            self.loss_eq2 = torch.mean(torch.square(self.eq2_pred.reshape([-1])))
            self.loss_eq3 = torch.mean(torch.square(self.eq3_pred.reshape([-1])))
            self.loss_eq4 = 0.1*torch.mean(torch.square(self.eq4_pred.reshape([-1])))
            self.loss_e = self.loss_eq1 + self.loss_eq2 + self.loss_eq3 + self.loss_eq4

        self.loss = self.alpha_b * self.loss_b + \
                    self.alpha_e * self.loss_e + \
                    self.alpha_i * self.loss_i + \
                    self.alpha_o * self.loss_o

        return self.loss, [self.loss_e, self.loss_b, self.loss_s, self.loss_o, self.loss_i]

    def train(self,
              lr=1e-4,
              num_epoch=1,
              optimizer=None,
              scheduler=None,
              batchsize=None):
        if self.opt is not None:
            self.opt.param_groups[0]['lr'] = lr
        else:
            self.opt = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        return self.solve_Adam(self.fwd_computing_loss_2d, num_epoch, batchsize, scheduler)

    def solve_Adam(self,
                   loss_func,
                   num_epoch=1000,
                   batchsize=None,
                   scheduler=None):
        epoch_id = 0
        self.freeze_evm_net(0)
        with tqdm(initial=epoch_id, total=num_epoch) as pbar:
            while epoch_id < num_epoch:
                # train evm net every 10000 step
                if epoch_id !=0 and epoch_id % 1000 == 0:
                   self.defreeze_evm_net(epoch_id)
                if (epoch_id - 1) % 1000 == 0:
                   self.freeze_evm_net(epoch_id)

                loss, losses = loss_func()
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()
                if scheduler:
                    scheduler.step()

                epoch_id = epoch_id + 1
                self.epoch_id = epoch_id
                pbar.set_description(f'total_loss: {loss.detach().cpu().item():.6f},'+
                                     f'eq_loss: {losses[0].detach().cpu().item():.3e},'+
                                     f'b_loss: {losses[1].detach().cpu().item():.3e},'+
                                     f'o_loss: {losses[3].detach().cpu().item():.3e},'+
                                     f'i_loss: {losses[4].detach().cpu().item():.3e}, \n')
                                     #f'eq1_loss: {self.loss_eq1.detach().cpu().item():.3e},'+
                                     #f'eq2_loss: {self.loss_eq2.detach().cpu().item():.3e},'+
                                     #f'eq3_loss: {self.loss_eq3.detach().cpu().item():.3e},\n'+
                                     #f'eq4_loss: {self.loss_eq4.detach().cpu().item():.3e}')
                pbar.update(1)
                
                if epoch_id % 1000 == 0:
                    self.print_log(loss, losses, epoch_id, num_epoch)

    def freeze_evm_net(self, epoch_id):
        #print("*"*20)
        #print("freeze evm net")
        #print(epoch_id)
        #if self.evm is not None:
        #    print(self.evm)
        for para in self.net_1.parameters():
            para.requires_grad = False
        self.opt.param_groups[0]['params'] = list(self.net.parameters())
        #print("*"*20)

    def defreeze_evm_net(self, epoch_id):
        #print("*"*20)
        #print("defreeze evm net")
        #print(epoch_id)
        #print(self.evm)
        for para in self.net_1.parameters():
            para.requires_grad = True
        self.opt.param_groups[0]['params'] = list(self.net.parameters())+list(self.net_1.parameters())
        #print("*"*20)

    def print_log(self, loss, losses, epoch_id, num_epoch):
        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']

        if isinstance(losses[0], int):
            eq_loss = losses[0]
        else:
            eq_loss = losses[0].detach().cpu().item()

        print("current lr is: %.6f " % (get_lr(self.opt)),
              "epoch/num_epoch: ", epoch_id + 1, "/", num_epoch,
              #"eq_loss: %.3e " %(eq_loss),
              "eq1_loss: %.3e " %(self.loss_eq1.detach().cpu().item()),
              "eq2_loss: %.3e " %(self.loss_eq2.detach().cpu().item()),
              "eq3_loss: %.3e " %(self.loss_eq3.detach().cpu().item()),
              "eq4_loss: %.3e " %(self.loss_eq4.detach().cpu().item()))

        if epoch_id % self.checkpoint_freq == 0:
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            torch.save(
                self.net.state_dict(),
                self.checkpoint_path + 'net_params_' + str(epoch_id) + '.pth')

    def evaluate(self, t, x, y, u, v):
        """ testing all points in the domain """
        t_test = t.reshape(-1,1)
        x_test = x.reshape(-1,1)
        y_test = y.reshape(-1,1)
        u_test = u.reshape(-1,1)
        v_test = v.reshape(-1,1)

        t_test = torch.tensor(t_test).float().to(device)
        x_test = torch.tensor(x_test).float().to(device)
        y_test = torch.tensor(y_test).float().to(device)
        u_pred, v_pred, _, _ = self.neural_net_u(t_test, x_test, y_test)
        u_pred = u_pred.detach().cpu().numpy().reshape(-1,1)
        v_pred = v_pred.detach().cpu().numpy().reshape(-1,1)
        # Error
        error_u = np.linalg.norm(u_test-u_pred,2)/np.linalg.norm(u_test,2)
        error_v = np.linalg.norm(v_test-v_pred,2)/np.linalg.norm(v_test,2)
        print('------------------------')
        print('Error u: %e' % (error_u))
        print('Error v: %e' % (error_v))
        print('------------------------')

    def test(self, x, y, u, v,loop=None):
        """ testing all points in the domain """
        x_test = x.reshape(-1,1)
        y_test = y.reshape(-1,1)
        u_test = u.reshape(-1,1)
        v_test = v.reshape(-1,1)
        # Prediction
        x_test = torch.tensor(x_test).float().to(device)
        y_test = torch.tensor(y_test).float().to(device)
        u_pred, v_pred, _, _ = self.neural_net_u(x_test, y_test)
        u_pred = u_pred.detach().cpu().numpy().reshape(-1,1)
        v_pred = v_pred.detach().cpu().numpy().reshape(-1,1)
        # Error
        error_u = np.linalg.norm(u_test-u_pred,2)/np.linalg.norm(u_test,2)
        error_v = np.linalg.norm(v_test-v_pred,2)/np.linalg.norm(v_test,2)
        print('------------------------')
        print('Error u: %e' % (error_u))
        print('Error v: %e' % (error_v))
        print('------------------------')

        div_pred = self.divergence(x_test, y_test)
        u_pred = u_pred.reshape(257,257)
        v_pred = v_pred.reshape(257,257)
        p_pred = p_pred.reshape(257,257)
        e_pred = e_pred.reshape(257,257)
        div_pred = div_pred.detach().cpu().numpy().reshape(257,257)

        scipy.io.savemat(save_results_to+'cavity_result_loop%d.mat'%(loop),
                    {'U_pred':u_pred,
                     'V_pred':v_pred,
                     'P_pred':p_pred,
                     'E_pred':e_pred,
                     'div_pred':div_pred,
                     'lam_bcs':self.alpha_b,
                     'lam_equ':self.alpha_e})

    def save(self, filename, directory=None, N_HLayer=None, N_neu=None, N_f=None):
        Re_folder = 'Re'+str(self.Re)
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

    def divergence(self, x_star, y_star):
        (self.eq1_pred, self.eq2_pred,
         self.eq3_pred, self.eq4_pred) = self.neural_net_equations(self.x_star, self.y_star)
        div = self.eq3_pred
        return div

    def prepare_next_ic(self, t, x, y):
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
        t = np.tile(t, (1,x.shape[0])).T # N x T=0
        t_test = torch.tensor(t).float().to(device)
        x_test = torch.tensor(x).float().to(device)
        y_test = torch.tensor(y).float().to(device)
        u, v, p, _ = self.neural_net_u(t_test, x_test, y_test)
        u = u.detach().cpu().numpy().reshape(-1,1)
        v = v.detach().cpu().numpy().reshape(-1,1)
        p = p.detach().cpu().numpy().reshape(-1,1)
        return p,u,v,t,x,y
