import os
import torch
import scipy.io
import numpy as np
from net import FCNet
from typing import Dict, List, Set, Optional, Union, Callable
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PysicsInformedNeuralNetwork:
    # Initialize the class
    # training_type:  'unsupervised' | 'half-supervised'
    def __init__(self,
                 opt=None,
                 Re = 2000,
                 Pr = 0.71,
                 layers=6,
                 hidden_size=80,
                 N_f = 4000,
                 learning_rate=0.001,
                 outlet_weight=1,
                 bc_weight=1,
                 eq_weight=1,
                 ic_weight=1,
                 num_ins=2, #输入x，y
                 num_outs=4,#输出u、v、p、t
                 supervised_data_weight=1,
                 training_type='unsupervised',#无监督学习
                 net_params=None,
                 checkpoint_freq=2000,
                 checkpoint_path='./checkpoint/'):
        
        self.Re = Re #雷诺数
        self.Pr = Pr #普朗特数
        self.Pe = self.Re * self.Pr #贝克莱数
        
        self.layers = layers
        self.hidden_size = hidden_size
        self.N_f = N_f
        
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_path = checkpoint_path

        self.training_type = training_type
        self.alpha_b = bc_weight
        self.alpha_e = eq_weight
        self.alpha_i = ic_weight
        self.alpha_o = outlet_weight
        self.alpha_s = supervised_data_weight
        self.loss_i = self.loss_o = self.loss_b = self.loss_e = self.loss_s = 0.0
        self.loss_bcs_all = []   #用于记录loss
        self.loss_equ_all = []
        self.loss_sum_all = []
        self.loss_e1_all = []
        self.loss_e2_all = []
        self.loss_e3_all = []
        self.loss_e4_all = []
        self.loss_e5_all = []
        self.loss_e6_all = []
        self.all_loss = 0
        self.eq_loss = 0
        self.bc_loss = 0
        self.e1_loss = 0

        # initialize NN
        self.net = self.initialize_NN(
                num_ins=num_ins, num_outs=num_outs, num_layers=layers, hidden_size=hidden_size).to(device)
       
        # if net_params:
        #     load_params = torch.load(net_params)
        #     self.net.load_state_dict(load_params)

        if net_params:
            try:
                # 尝试直接使用 torch.load 加载模型参数，映射到当前可用的设备
                load_params = torch.load(net_params,
                                         map_location=torch.device('cuda:0'))  # 或者使用 map_location='cuda'，根据需要选择合适的方式
            except AttributeError as e:
                if "'collections.OrderedDict' object has no attribute 'seek'" in str(e):
                    # 如果 net_params 是一个 OrderedDict 而不是文件路径，需要重新加载
                    buffer = io.BytesIO()
                    torch.save(net_params, buffer)
                    buffer.seek(0)
                    load_params = torch.load(buffer, map_location=torch.device('cuda:0'))  # 或者使用 map_location='cuda'
            except RuntimeError as e:
                if "Attempting to deserialize object on CUDA device" in str(e):
                    # 如果提示尝试在 CUDA 设备上反序列化但设备数量不匹配的错误，手动映射到当前设备
                    load_params = torch.load(net_params,
                                             map_location=torch.device('cuda:0'))  # 或者使用 map_location='cuda'

            # 将加载的参数应用到网络模型的状态字典中
            self.net.load_state_dict(load_params)

        self.opt = torch.optim.Adam(
            list(self.net.parameters()),
            lr=learning_rate,
            weight_decay=0.0) if not opt else opt    #使用adam优化器

    def set_boundary_data(self, X=None):
        # boundary training data | u, v, t, x, y
        """
        设置边界数据。

        参数:
        - X: 一个包含边界数据的列表或数组。应包含以下五个元素：
          [x_b, y_b, u_b, v_b, t_b]，分别对应边界处的x坐标、y坐标、u值、v值和t值。
        """
        requires_grad = False #不需要计算梯度
        # 将边界数据转换为 PyTorch 张量并移动到设备上
        self.x_b = torch.tensor(X[0], requires_grad=requires_grad).float().to(device)
        self.y_b = torch.tensor(X[1], requires_grad=requires_grad).float().to(device)
        self.u_b = torch.tensor(X[2], requires_grad=requires_grad).float().to(device)
        self.v_b = torch.tensor(X[3], requires_grad=requires_grad).float().to(device)
        self.t_b = torch.tensor(X[4], requires_grad=requires_grad).float().to(device)

    def set_eq_training_data(self,X=None):
        """
        设置用于方程训练的数据。

        参数:
        - X: 一个包含方程训练数据的列表或数组。应包含以下两个元素：
          [x_f, y_f]，分别对应方程训练数据的x坐标和y坐标。
        """
        # inferior training data | u, v, t, x,
        requires_grad = True #需要计算梯度
        # 将方程数据转换为 PyTorch 张量并移动到设备上
        self.x_f = torch.tensor(X[0], requires_grad=requires_grad).float().to(device)
        self.y_f = torch.tensor(X[1], requires_grad=requires_grad).float().to(device)

    def set_optimizers(self, opt):
        self.opt = opt #优化器
        
    def initialize_NN(self,
                      num_ins=2,
                      num_outs=3,
                      num_layers=10,
                      hidden_size=50):
        return FCNet(num_ins=num_ins,
                     num_outs=num_outs,
                     num_layers=num_layers,
                     hidden_size=hidden_size,
                     activation=torch.nn.Tanh)
        #初始化NN

    def neural_net_u(self, x, y):
        """
        神经网络输入 x、y 坐标，输出 u、v、p、t。

        参数:
        - x: 输入的 x 坐标张量。
        - y: 输入的 y 坐标张量。

        返回:
        - u: 预测的 u 值。
        - v: 预测的 v 值。
        - p: 预测的 p 值。
        - t: 预测的 t 值。
        """
        X = torch.cat((x, y), dim=1)
        uvp = self.net(X)
        u = uvp[:, 0:1]
        v = uvp[:, 1:2]
        p = uvp[:, 2:3]
        t = uvp[:, 3:4]
        return u, v, p, t

    def neural_net_equations(self, x, y):
        """
        输出内部训练点方程的结果。

        参数:
        - x: 输入的 x 坐标张量。
        - y: 输入的 y 坐标张量。

        返回:
        - eq1, eq2, eq3, eq4: 方程结果。
        """
        X = torch.cat((x, y), dim=1)
        uvp = self.net(X)
        u = uvp[:, 0:1]
        v = uvp[:, 1:2]
        p = uvp[:, 2:3]
        t = uvp[:, 3:4] #神经网络输入x、y；输出u、v、p、t

        u_x, u_y = self.autograd(u, [x,y])
        u_xx = self.autograd(u_x, [x])[0]
        u_yy = self.autograd(u_y, [y])[0]

        v_x, v_y = self.autograd(v, [x,y])
        v_xx = self.autograd(v_x, [x])[0]
        v_yy = self.autograd(v_y, [y])[0]

        t_x, t_y = self.autograd(t, [x,y])
        t_xx = self.autograd(t_x, [x])[0]
        t_yy = self.autograd(t_y, [y])[0]
        
        p_x, p_y = self.autograd(p, [x,y]) #计算u，v，p，t的梯度

        #构建NS方程和能量方程
        eq1 = (u*u_x + v*u_y) + p_x - (1.0/self.Re)*(u_xx + u_yy)
        eq2 = (u*v_x + v*v_y) + p_y - (1.0/self.Re)*(v_xx + v_yy)
        eq3 = (u*t_x + v*t_y) - (1.0/self.Pe)*(t_xx + t_yy)
        eq4 = u_x + v_y 
        return eq1, eq2, eq3, eq4

    @torch.jit.script  #计算梯度
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
        # shuffle_numpy = np.random.shuffle(tensor_to_numpy)
        return torch.tensor(tensor_to_numpy, requires_grad=True).float()

    def fwd_computing_loss_2d(self, loss_mode='MSE'):
        """
        构建损失函数，包括边界的损失和方程的损失。

        参数:
        - loss_mode (str): 损失计算方式，可以是'L2'或'MSE'。默认值为'MSE'。

        返回:
        - self.loss: 总损失
        - [self.loss_e, self.loss_b]: 包含方程损失和边界损失的列表
        """
        # boundary data
        (self.u_pred_b, self.v_pred_b, _, self.t_pred_b) = self.neural_net_u(self.x_b, self.y_b)

        # BC loss 边界的损失
        if loss_mode == 'L2':
            self.loss_b = torch.norm((self.u_b.reshape([-1]) - self.u_pred_b.reshape([-1])), p=2) + \
                          torch.norm((self.v_b.reshape([-1]) - self.v_pred_b.reshape([-1])), p=2) + \
                          torch.norm((self.t_b.reshape([-1]) - self.t_pred_b.reshape([-1])), p=2)
        if loss_mode == 'MSE':
            self.loss_b = torch.mean(torch.square(self.u_b.reshape([-1]) - self.u_pred_b.reshape([-1]))) + \
                          torch.mean(torch.square(self.v_b.reshape([-1]) - self.v_pred_b.reshape([-1]))) + \
                          torch.mean(torch.square(self.t_b.reshape([-1]) - self.t_pred_b.reshape([-1])))

        # equation 方程的损失
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
            self.loss_e = self.loss_eq1 + self.loss_eq2 + self.loss_eq3 + self.loss_eq4 
        # 总损失
        self.loss = self.alpha_b * self.loss_b + self.alpha_e * self.loss_e

        return self.loss, [self.loss_e, self.loss_b]

    def train(self,
              num_epoch=1,
              lr=1e-4,
              optimizer=None,
              scheduler=None,
              batchsize=None):
        """
        训练模型。

        参数:
        - num_epoch (int): 训练的轮数。默认值为1。
        - lr (float): 优化器的学习率。默认值为1e-4。
        - optimizer: 用于训练的优化器。如果为None，则使用self.opt。默认值为None。
        - scheduler: 学习率调度器。默认值为None。
        - batchsize: 训练的批量大小。默认值为None。

        返回:
        - solve_Adam方法的结果。
        """
        if self.opt is not None:
            self.opt.param_groups[0]['lr'] = lr
        return self.solve_Adam(self.fwd_computing_loss_2d, num_epoch, batchsize, scheduler)

    def solve_Adam(self,
                   loss_func,
                   num_epoch=1000,
                   batchsize=None,
                   scheduler=None):
        """
        使用Adam优化器进行训练。

        参数:
        - loss_func: 损失函数。
        - num_epoch (int): 训练的轮数。默认值为1000。
        - batchsize: 训练的批量大小。默认值为None。
        - scheduler: 学习率调度器。默认值为None。
        """
        for epoch_id in range(num_epoch):
            loss, losses = loss_func()  # 计算损失
            loss.backward()  # 反向传播计算梯度
            self.opt.step()  # 更新模型参数
            self.opt.zero_grad()  # 梯度清零

            # 提取并记录各种损失值
            e_loss = losses[0].detach().cpu().item()
            all_loss = loss.detach().cpu().item()
            bc_loss = losses[1].detach().cpu().item()
            e1_loss = self.loss_eq1.detach().cpu().item()
            e2_loss = self.loss_eq2.detach().cpu().item()
            e3_loss = self.loss_eq3.detach().cpu().item()
            e4_loss = self.loss_eq4.detach().cpu().item()

            self.loss_bcs_all.append([bc_loss])
            self.loss_equ_all.append([e_loss])
            self.loss_sum_all.append([all_loss])
            self.loss_e1_all.append([e1_loss])
            self.loss_e2_all.append([e2_loss])
            self.loss_e3_all.append([e3_loss])
            self.loss_e4_all.append([e4_loss])

            if scheduler:
                scheduler.step()  # 更新学习率

            # 打印日志
            if epoch_id == 0 or (epoch_id + 1) % 100 == 0:
                self.print_log(loss, losses, epoch_id, num_epoch)

    def print_log(self, loss, losses, epoch_id, num_epoch):
        """
        打印训练日志。

        参数:
        - loss: 总损失。
        - losses: 一个包含方程损失和边界损失的列表。
        - epoch_id: 当前的训练轮数。
        - num_epoch: 总的训练轮数。
        """
        # 获取当前学习率
        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']

        # 打印当前学习率
        print("current lr is {}".format(get_lr(self.opt)))
        if isinstance(losses[0], int):
            eq_loss = losses[0]
        else:
            eq_loss = losses[0].detach().cpu().item()
        # 打印当前轮次信息和损失信息
        print("epoch/num_epoch: ", epoch_id + 1, "/", num_epoch,
              "loss[Adam]: %.3e"%(loss.detach().cpu().item()),
              "eq_loss: %.3e" %(losses[0].detach().cpu().item()),
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
        # 打印结果
        # print('x_test的元素个数:', num_elements)
        # print('元素个数的平方根:', sqrt_num)
        # Prediction
        x_test = torch.tensor(x_test).float().to(device)
        y_test = torch.tensor(y_test).float().to(device)
        u_pred, v_pred, p_pred, t_pred = self.neural_net_u(x_test, y_test)
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

    def test(self, x, y, u, v, p, t, num_epoch, loop=None):
        """
        在整个域中测试模型的性能，并保存结果。

        参数:
        - x: 输入的 x 坐标数据。
        - y: 输入的 y 坐标数据。
        - u: 真实的 u 值。
        - v: 真实的 v 值。
        - p: 真实的 p 值。
        - t: 真实的 t 值。
        - num_epoch: 当前的训练轮次。
        - loop: 当前的训练循环（如果有）。
        """
        # 将输入数据重新形状为二维张量
        x_test = x.reshape(-1,1)
        y_test = y.reshape(-1,1)
        u_test = u.reshape(-1,1)
        v_test = v.reshape(-1,1)
        p_test = p.reshape(-1,1)
        t_test = t.reshape(-1,1)
        # 将测试数据转换为张量并移动到设备上
        x_test = torch.tensor(x_test).float().to(device)
        y_test = torch.tensor(y_test).float().to(device)
        # 使用神经网络进行预测
        u_pred, v_pred, p_pred, t_pred = self.neural_net_u(x_test, y_test)
        # 将预测结果转换为 NumPy 数组
        u_pred = u_pred.detach().cpu().numpy().reshape(-1,1)
        v_pred = v_pred.detach().cpu().numpy().reshape(-1,1)
        p_pred = p_pred.detach().cpu().numpy().reshape(-1,1)
        t_pred = t_pred.detach().cpu().numpy().reshape(-1,1)
        # Error
        # 计算误差
        error_u = np.linalg.norm(u_test-u_pred,2)/np.linalg.norm(u_test,2)
        error_v = np.linalg.norm(v_test-v_pred,2)/np.linalg.norm(v_test,2)
        error_p = np.linalg.norm(p_test-p_pred,2)/np.linalg.norm(p_test,2)
        error_t = np.linalg.norm(t_test-t_pred,2)/np.linalg.norm(t_test,2)
        # 打印误差信息
        print('------------------------')
        print('Error u: %e' % (error_u))
        print('Error v: %e' % (error_v))
        print('Error p: %e' % (error_p))
        print('Error t: %e' % (error_t))
        print('------------------------')
        # 将预测结果重新形状为二维数组
        u_pred = u_pred.reshape(257,257)
        v_pred = v_pred.reshape(257,257)
        p_pred = p_pred.reshape(257,257)
        t_pred = t_pred.reshape(257,257)
        
        result_folder = 'result_data'
        os.makedirs(result_folder, exist_ok=True)
        
        # 保存.mat文件到新建文件夹中
        mat_file_path = os.path.join(result_folder, 'cavity_result_loop_%d_epoch%d.mat' % (loop, num_epoch))
        scipy.io.savemat(mat_file_path,
                    {'Error_u':error_u,
                     'Error_v':error_v,
                     'Error_p':error_p,
                     'Error_t':error_t,
                     'U_pred':u_pred,
                     'V_pred':v_pred,
                     'P_pred':p_pred,
                     'T_pred':t_pred,
                     'lam_bcs':self.alpha_b,
                     'lam_equ':self.alpha_e,
                     'loss_bcs_all':self.loss_bcs_all,
                     'loss_equ_all':self.loss_equ_all,
                     'loss_sum_all':self.loss_sum_all})
    
