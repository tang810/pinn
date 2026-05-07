import os
import torch
import scipy.io
import numpy as np
from net import FCNet
from typing import Dict, List, Set, Optional, Union, Callable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PysicsInformedNeuralNetwork:
    # Initialize the class
    def __init__(self,
                 opt=None,
                 Re = 1000,
                 Pr = 0.71,
                 Ri = 0.1,
                 layers=6,
                 layers_1=6,
                 layers_2=6,
                 hidden_size=80,
                 hidden_size_1=20,
                 hidden_size_2=20,
                 N_f = 100000,
                 alpha_evm=0.03,
                 learning_rate=0.001,
                 weight_decay=0.9,
                 outlet_weight=1,
                 bc_weight=1,
                 eq_weight=1,
                 ic_weight=1,
                 num_ins=2,
                 num_outs=4,
                 num_outs_1=1,
                 num_outs_2=1,
                 supervised_data_weight=1,
                 net_params=None,
                 net_params_1=None,
                 net_params_2=None,
                 checkpoint_freq=2000,
                 checkpoint_path='./checkpoint/'):

        self.evm = None
        self.Re = Re
        self.Pr = Pr
        self.Ri = Ri
        self.vis_t0 = 10.0/self.Re
        self.vis_t1 = 7.1/(self.Re * self.Pr)

        self.layers = layers
        self.layers_1 = layers_1
        self.hidden_size = hidden_size
        self.hidden_size_1 = hidden_size_1
        self.N_f = N_f

        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_path = checkpoint_path

        self.alpha_evm = alpha_evm
        self.alpha_b = bc_weight
        self.alpha_e = eq_weight
        self.alpha_i = ic_weight
        self.alpha_o = outlet_weight
        self.alpha_s = supervised_data_weight
        self.loss_i = self.loss_o = self.loss_b = self.loss_e = self.loss_s = 0.0
        self.loss_bcs_all = []
        self.loss_equ_all = []
        self.loss_sum_all = []

        # initialize NN
        self.net = self.initialize_NN(
                num_ins=num_ins, num_outs=num_outs, num_layers=layers, hidden_size=hidden_size).to(device)
        self.net_1 = self.initialize_NN(
                num_ins=num_ins, num_outs=num_outs_1, num_layers=layers_1, hidden_size=hidden_size_1).to(device)
        self.net_2 = self.initialize_NN(
                num_ins=num_ins, num_outs=num_outs_2, num_layers=layers_2, hidden_size=hidden_size_2).to(device)
        
        # if net_params:
        #     load_params = torch.load(net_params)
        #     self.net.load_state_dict(load_params)
        if net_params:
            try:
                # 灏濊瘯鐩存帴浣跨敤 torch.load 鍔犺浇妯″瀷鍙傛暟锛屾槧灏勫埌褰撳墠鍙敤鐨勮澶?
                load_params = torch.load(net_params,
                                         map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))  # 鎴栬€呬娇鐢?map_location='cuda'锛屾牴鎹渶瑕侀€夋嫨鍚堥€傜殑鏂瑰紡
            except AttributeError as e:
                if "'collections.OrderedDict' object has no attribute 'seek'" in str(e):
                    # 濡傛灉 net_params 鏄竴涓?OrderedDict 鑰屼笉鏄枃浠惰矾寰勶紝闇€瑕侀噸鏂板姞杞?
                    buffer = io.BytesIO()
                    torch.save(net_params, buffer)
                    buffer.seek(0)
                    load_params = torch.load(buffer, map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))  # 鎴栬€呬娇鐢?map_location='cuda'
            except RuntimeError as e:
                if "Attempting to deserialize object on CUDA device" in str(e):
                    # 濡傛灉鎻愮ず灏濊瘯鍦?CUDA 璁惧涓婂弽搴忓垪鍖栦絾璁惧鏁伴噺涓嶅尮閰嶇殑閿欒锛屾墜鍔ㄦ槧灏勫埌褰撳墠璁惧
                    load_params = torch.load(net_params,
                                             map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))  # 鎴栬€呬娇鐢?map_location='cuda'

            # 灏嗗姞杞界殑鍙傛暟搴旂敤鍒扮綉缁滄ā鍨嬬殑鐘舵€佸瓧鍏镐腑
            self.net.load_state_dict(load_params)

        if net_params_1:
            load_params_1 = torch.load(net_params_1)
            self.net_1.load_state_dict(load_params_1)
        
        if net_params_2:
            load_params_2 = torch.load(net_params_2)
            self.net_2.load_state_dict(load_params_2)

        self.opt = torch.optim.Adam(
            list(self.net.parameters())+list(self.net_1.parameters())+list(self.net_2.parameters()),
            lr=learning_rate,
            weight_decay=0.0) if not opt else opt

    def init_vis_t(self):
        (_, _, _, _, e0, e1) = self.neural_net_u(self.x_f, self.y_f)
        self.vis_t0_minus  = self.alpha_evm*torch.abs(e0).detach().cpu().numpy()
        self.vis_t1_minus  = self.alpha_evm*torch.abs(e1).detach().cpu().numpy()

    def set_boundary_data(self, X=None, time=False):
        # boundary training data | u, v, t, x, y
        requires_grad = False
        self.x_b = torch.tensor(X[0], requires_grad=requires_grad).float().to(device)
        self.y_b = torch.tensor(X[1], requires_grad=requires_grad).float().to(device)
        self.u_b = torch.tensor(X[2], requires_grad=requires_grad).float().to(device)
        self.v_b = torch.tensor(X[3], requires_grad=requires_grad).float().to(device)
        self.t_b = torch.tensor(X[4], requires_grad=requires_grad).float().to(device)
        if time:
            self.t_b = torch.tensor(X[4], requires_grad=requires_grad).float().to(device)

    def set_eq_training_data(self,
                             X=None,
                             time=False):
        requires_grad = True
        self.x_f = torch.tensor(X[0], requires_grad=requires_grad).float().to(device)
        self.y_f = torch.tensor(X[1], requires_grad=requires_grad).float().to(device)
        if time:
            self.t_f = torch.tensor(X[2], requires_grad=requires_grad).float().to(device)

        self.init_vis_t()

    def set_optimizers(self, opt):
        self.opt = opt

    def set_alpha_evm(self, alpha):
        self.alpha_evm = alpha

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

    def neural_net_u(self, x, y):
        X = torch.cat((x, y), dim=1)
        uvpt = self.net(X)
        e0 = self.net_1(X)
        e1 = self.net_2(X) 
        u = uvpt[:, 0:1]
        v = uvpt[:, 1:2]
        p = uvpt[:, 2:3]
        t = uvpt[:, 3:4]
        e0 = e0[:, 0:1]
        e1 = e1[:, 0:1]
        return u, v, p, t, e0, e1
    

    def neural_net_equations(self, x, y):
        X = torch.cat((x, y), dim=1)
        uvpt = self.net(X)
        e0 = self.net_1(X)
        e1 = self.net_2(X) 
        u = uvpt[:, 0:1]
        v = uvpt[:, 1:2]
        p = uvpt[:, 2:3]
        t = uvpt[:, 3:4]
        e0 = e0[:, 0:1]
        e1 = e1[:, 0:1]
        self.evm0 = e0
        self.evm1 = e1

        u_x, u_y = self.autograd(u, [x,y])
        u_xx = self.autograd(u_x, [x])[0]
        u_yy = self.autograd(u_y, [y])[0]

        v_x, v_y = self.autograd(v, [x,y])
        v_xx = self.autograd(v_x, [x])[0]
        v_yy = self.autograd(v_y, [y])[0]

        p_x, p_y = self.autograd(p, [x,y])
        
        t_x, t_y = self.autograd(t, [x,y])
        t_xx = self.autograd(t_x, [x])[0]
        t_yy = self.autograd(t_y, [y])[0]

        # Get the minum between (vis_t0, vis_t_mius(calculated with last step e))
        self.vis_e0 = torch.tensor(np.minimum(self.vis_t0, self.vis_t0_minus)).float().to(device)
        # Save vis_t_minus for computing vis_t in the next step
        self.vis_t0_minus  = self.alpha_evm*torch.abs(e0).detach().cpu().numpy()
        
        self.vis_e1 = torch.tensor(np.minimum(self.vis_t1, self.vis_t1_minus)).float().to(device)
        # Save vis_t_minus for computing vis_t in the next step
        self.vis_t1_minus  = self.alpha_evm*torch.abs(e1).detach().cpu().numpy()

        # NS
        eq1 = (u*u_x + v*u_y) + p_x - (1.0/self.Re + self.vis_e0)*(u_xx + u_yy)
        eq2 = (u*v_x + v*v_y) + p_y - (1.0/self.Re + self.vis_e0)*(v_xx + v_yy) - self.Ri*t
        eq3 = (u*t_x + v*t_y) - ((1.0/self.Re)*(1.0/self.Pr) + self.vis_e1)*(t_xx + t_yy)
        eq4 = u_x + v_y

        residual_1 = (eq1*(u-0.5) + eq2*(v-0.5)) - e0
        residual_2 = (eq3*(t-0.5)) - e1
        return eq1, eq2, eq3, eq4, residual_1, residual_2

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
        x, y = X
        return self.neural_net_u(x, y)

    def shuffle(self, tensor):
        tensor_to_numpy = tensor.detach().cpu()
        shuffle_numpy = np.random.shuffle(tensor_to_numpy)
        return torch.tensor(tensor_to_numpy, requires_grad=True).float()

    def fwd_computing_loss_2d(self, loss_mode='MSE'):
        # boundary data
        (self.u_pred_b, self.v_pred_b, _, self.t_pred_b, _, _) = self.neural_net_u(self.x_b, self.y_b)

        # BC loss
        if loss_mode == 'L2':
            self.loss_b = torch.norm((self.u_b.reshape([-1]) - self.u_pred_b.reshape([-1])), p=2) + \
                          torch.norm((self.v_b.reshape([-1]) - self.v_pred_b.reshape([-1])), p=2)
        if loss_mode == 'MSE':
            self.loss_b = torch.mean(torch.square(self.u_b.reshape([-1]) - self.u_pred_b.reshape([-1]))) + \
                          torch.mean(torch.square(self.v_b.reshape([-1]) - self.v_pred_b.reshape([-1]))) + \
                          torch.mean(torch.square(self.t_b.reshape([-1]) - self.t_pred_b.reshape([-1])))

        # equation
        assert self.x_f is not None and self.y_f is not None

        (self.eq1_pred, self.eq2_pred,
         self.eq3_pred, self.eq4_pred, self.eq5_pred, self.eq6_pred) = self.neural_net_equations(self.x_f, self.y_f)
        if loss_mode == 'L2':
            self.loss_e = torch.norm(self.eq1_pred.reshape([-1]), p=2) + \
                          torch.norm(self.eq2_pred.reshape([-1]), p=2) + \
                          torch.norm(self.eq3_pred.reshape([-1]), p=2)
        if loss_mode == 'MSE':
            self.loss_eq1 = torch.mean(torch.square(self.eq1_pred.reshape([-1])))
            self.loss_eq2 = torch.mean(torch.square(self.eq2_pred.reshape([-1])))
            self.loss_eq3 = torch.mean(torch.square(self.eq3_pred.reshape([-1])))
            self.loss_eq4 = torch.mean(torch.square(self.eq4_pred.reshape([-1])))
            self.loss_eq5 = torch.mean(torch.square(self.eq5_pred.reshape([-1])))
            self.loss_eq6 = torch.mean(torch.square(self.eq6_pred.reshape([-1])))
            self.loss_e = self.loss_eq1 + self.loss_eq2 + self.loss_eq3 + self.loss_eq4 + 0.1*self.loss_eq5 + 0.1*self.loss_eq6

        self.loss = self.alpha_b * self.loss_b + self.alpha_e * self.loss_e

        return self.loss, [self.loss_e, self.loss_b]

    def train(self,
              num_epoch=1,
              lr=1e-4,
              optimizer=None,
              scheduler=None,
              batchsize=None):
        if self.opt is not None:
            self.opt.param_groups[0]['lr'] = lr
        return self.solve_Adam(self.fwd_computing_loss_2d, num_epoch, batchsize, scheduler)

    def solve_Adam(self,
                   loss_func,
                   num_epoch=1000,
                   batchsize=None,
                   scheduler=None):
        self.freeze_evm_net(0)
        for epoch_id in range(num_epoch):
            # train evm net every 10000 step
            if epoch_id !=0 and epoch_id % 10000 == 0:
                self.defreeze_evm_net(epoch_id)
            if (epoch_id - 1) % 10000 == 0:
                self.freeze_evm_net(epoch_id)

            loss, losses = loss_func()
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
            e_loss = losses[0].detach().cpu().item()
            all_loss = loss.detach().cpu().item()
            bc_loss = losses[1].detach().cpu().item()

            self.loss_bcs_all.append([bc_loss])
            self.loss_equ_all.append([e_loss])
            self.loss_sum_all.append([all_loss])
            if scheduler:
                scheduler.step()

            if epoch_id == 0 or (epoch_id + 1)%100 == 0:
                self.print_log(loss, losses, epoch_id, num_epoch)

    def freeze_evm_net(self, epoch_id):
        for para in self.net_1.parameters():
            para.requires_grad = False
        self.opt.param_groups[0]['params'] = list(self.net.parameters())

    def defreeze_evm_net(self, epoch_id):
        for para in self.net_1.parameters():
            para.requires_grad = True
        self.opt.param_groups[0]['params'] = list(self.net.parameters()) + list(self.net_1.parameters()) + list(self.net_2.parameters())

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
              %(loss.detach().cpu().item()), 
              "eq_loss: %.3e " %(losses[0].detach().cpu().item()),
              "bc_loss: %.3e" %(losses[1].detach().cpu().item())) 


    def evaluate(self, x, y, u, v, t):
        """ testing all points in the domain """
        x_test = x.reshape(-1, 1)
        y_test = y.reshape(-1, 1)
        u_test = u.reshape(-1, 1)
        v_test = v.reshape(-1, 1)
        t_test = t.reshape(-1, 1)
        num_elements = x_test.size
        sqrt_num = np.sqrt(num_elements)
        # 鎵撳嵃缁撴灉
        # print('x_test鐨勫厓绱犱釜鏁?', num_elements)
        # print('鍏冪礌涓暟鐨勫钩鏂规牴:', sqrt_num)
        # Prediction
        x_test = torch.tensor(x_test).float().to(device)
        y_test = torch.tensor(y_test).float().to(device)
        u_pred, v_pred, p_pred, t_pred, _, _ = self.neural_net_u(x_test, y_test)
        u_pred = u_pred.detach().cpu().numpy().reshape(-1, 1)
        v_pred = v_pred.detach().cpu().numpy().reshape(-1, 1)
        t_pred = t_pred.detach().cpu().numpy().reshape(-1, 1)
        # Error
        error_u = np.linalg.norm(u_test - u_pred, 2) / np.linalg.norm(u_test, 2)
        error_v = np.linalg.norm(v_test - v_pred, 2) / np.linalg.norm(v_test, 2)
        error_t = np.linalg.norm(t_test - t_pred, 2) / np.linalg.norm(t_test, 2)
        print('------------------------')
        print('Error u: %e' % (error_u))
        print('Error v: %e' % (error_v))
        print('Error t: %e' % (error_t))
        print('------------------------')
        # plot picture

        error_u = np.abs(u_test - u_pred).reshape(int(sqrt_num), int(sqrt_num))
        error_v = np.abs(v_test - v_pred).reshape(int(sqrt_num), int(sqrt_num))
        error_t = np.abs(t_test - t_pred).reshape(int(sqrt_num), int(sqrt_num))
        u_test = u_test.reshape(257, 257)
        v_test = v_test.reshape(257, 257)
        t_test = t_test.reshape(257, 257)
        u_pred = u_pred.reshape(257,257)
        v_pred = v_pred.reshape(257,257)
        t_pred = t_pred.reshape(257,257)
        x_test = x_test.cpu().numpy().reshape(int(sqrt_num), int(sqrt_num))
        y_test = y_test.cpu().numpy().reshape(int(sqrt_num), int(sqrt_num))

        plt.figure(figsize=(14, 10))
        plt.subplot(3, 3, 1)
        plt.pcolormesh(x_test, y_test, u_test, shading='auto',cmap='jet')
        plt.colorbar()
        plt.title('Reference_U')

        plt.subplot(3, 3, 2)
        plt.pcolormesh(x_test, y_test, u_pred, shading='auto',cmap='jet')
        plt.colorbar()
        plt.title('Pred_PINN_U')

        plt.subplot(3, 3, 3)
        plt.pcolormesh(x_test, y_test, error_u, shading='auto',cmap='jet')
        plt.colorbar()
        plt.title('Error_U')

        plt.subplot(3, 3, 4)
        plt.pcolormesh(x_test, y_test, v_test, shading='auto', cmap='jet')
        plt.colorbar()
        plt.title('Reference_V')

        plt.subplot(3, 3, 5)
        plt.pcolormesh(x_test, y_test, v_pred, shading='auto', cmap='jet')
        plt.colorbar()
        plt.title('Pred_PINN_V')

        plt.subplot(3, 3, 6)
        plt.pcolormesh(x_test, y_test, error_v, shading='auto', cmap='jet')
        plt.colorbar()
        plt.title('Error_V')

        plt.subplot(3, 3, 7)
        plt.pcolormesh(x_test, y_test, t_test, shading='auto', cmap='jet')
        plt.colorbar()
        plt.title('Reference_T')

        plt.subplot(3, 3, 8)
        plt.pcolormesh(x_test, y_test, t_pred, shading='auto', cmap='jet')
        plt.colorbar()
        plt.title('Pred_PINN_T')

        plt.subplot(3, 3, 9)
        plt.pcolormesh(x_test, y_test, error_t, shading='auto', cmap='jet')
        plt.colorbar()
        plt.title('Error_T')

        plt.savefig('result_plot.png')
        plt.show()
        plt.close()

    def test(self, x, y, u, v, t, num_epoch, loop=None):
        """ testing all points in the domain """
        x_test = x.reshape(-1,1)
        y_test = y.reshape(-1,1)
        u_test = u.reshape(-1,1)
        v_test = v.reshape(-1,1)
        t_test = t.reshape(-1,1)
        # Prediction
        x_test = torch.tensor(x_test).float().to(device)
        y_test = torch.tensor(y_test).float().to(device)
        u_pred, v_pred, p_pred, t_pred, e1_pred, e2_pred= self.neural_net_u(x_test, y_test)
        u_pred = u_pred.detach().cpu().numpy().reshape(-1,1)
        v_pred = v_pred.detach().cpu().numpy().reshape(-1,1)
        p_pred = p_pred.detach().cpu().numpy().reshape(-1,1)
        t_pred = t_pred.detach().cpu().numpy().reshape(-1,1)
        e1_pred = e1_pred.detach().cpu().numpy().reshape(-1,1)
        e2_pred = e2_pred.detach().cpu().numpy().reshape(-1,1)
        # Error
        error_u = np.linalg.norm(u_test-u_pred,2)/np.linalg.norm(u_test,2)
        error_v = np.linalg.norm(v_test-v_pred,2)/np.linalg.norm(v_test,2)
        error_t = np.linalg.norm(t_test-t_pred,2)/np.linalg.norm(t_test,2)
        print('------------------------')
        print('Error u: %e' % (error_u))
        print('Error v: %e' % (error_v))
        print('Error t: %e' % (error_t))
        print('------------------------')

        u_pred = u_pred.reshape(257,257)
        v_pred = v_pred.reshape(257,257)
        p_pred = p_pred.reshape(257,257)
        t_pred = t_pred.reshape(257,257)
        e1_pred = e1_pred.reshape(257,257)
        e2_pred = e2_pred.reshape(257,257)
        result_folder = 'result_uvtdata'
        os.makedirs(result_folder, exist_ok=True)
        
        # 淇濆瓨.mat鏂囦欢鍒版柊寤烘枃浠跺す涓?
        mat_file_path = os.path.join(result_folder, 'cavity_result_loop_%d_epoch%d.mat' % (loop, num_epoch))
        scipy.io.savemat(mat_file_path,
                    {'U_pred':u_pred,
                     'V_pred':v_pred,
                     'P_pred':p_pred,
                     'T_pred':t_pred,
                     'Error_u':error_u,
                     'Error_v':error_v,
                     'Error_t':error_t,
                     'E1_pred':e1_pred,
                     'E2_pred':e2_pred,
                     'loss_bcs_all':self.loss_bcs_all,
                     'loss_equ_all':self.loss_equ_all,
                     'loss_sum_all':self.loss_sum_all,
                     'lam_bcs':self.alpha_b,
                     'lam_equ':self.alpha_e})



