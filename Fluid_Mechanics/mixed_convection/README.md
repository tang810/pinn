# 人工粘性项增强的混合对流传热求解

## 快速使用

您可跳转[使用示例](#使用示例)查看如何使用秋月白平台快速执行该类问题，以下是实现逻辑详解：

## 背景介绍

对流传热是指热量通过流体（液体或者气体）传递的过程。在这个过程中，热量是通过流体进行传递而不是直接通过导体或辐射的方式。对流传热可以通过两种主要方式发生：强迫对流和混合对流。这里介绍的混合对流传热问题（Mixed Convective Heat Transfer Problem）是指在传热过程中同时存在自然对流和强迫对流两种机制的情况。这种传热方式在许多工程应用中都很常见，例如在建筑物的通风系统、电子设备的冷却以及环境控制等方面。流动与传热问题中的偏微分方程主要是Navier-Stokes和能量守恒方程。人工粘性是一种常见于流体力学与计算流体动力学中用于增强数值解稳定性的技术，本案例将借助PINN计算二维方腔流动换热问题展示人工粘性技术对求解准确性的影响。


## 方腔混合对流传热问题
本案例求解的问题为二维方腔混合对流传热问题，主要关注其边界条件，左右两侧边界流速与温度均为0；上边界有一沿$x$轴正方向的速度场，为避免边界条件不连续处的拐角处出现强奇点，我们上边界流动传热的边界条件表述为图中的形式；下边界有一随横坐标变化的温度场。这样设定边界条件可以使得方腔的上半部分区域处于强迫对流状态，下半部分处于自然对流状态。

**计算空间示意图**

![caseMarkdown/case_d749f814/viz/c6517c86ba096749595047997899dc37.png](https://www.science42.tech/cases/caseMarkdown/case_d749f814/viz/c6517c86ba096749595047997899dc37.png)

## PINN实现
### 控制方程
本案例采用全连接神经网络来求解传热问题：网络以 $(x,y)$ 为输入，输出四个物理量 $(u,v,p,\theta)$，其中 $\theta$ 为温度，$(u,v)$ 为流动速度分量，$p$为压力。所要计算的二维稳态Navier-Stokes方程和传热方程如下所示

- 二维方腔混合对流动量控制方程:
$$
(\mathbf{u}\cdot\nabla)\mathbf{u}=-\nabla p+\frac1{\mathrm{Re}}\nabla^2\mathbf{u}+\mathrm{Ri}\theta,\quad\mathrm{~in~}\Omega 
$$
等式左侧$(\mathbf{u}\cdot\nabla)\mathbf{u}$表示惯性对流项，其代表流体自身运动带来的对流惯性力。右侧第一项为压力梯度$-\nabla p$，第二项为粘性扩散项，第三项为浮力源项，该项温度差产生的浮升力，是自然对流来源。


- 能量输运方程:
$$
(\mathbf{u}\cdot\nabla)\theta=\frac1{\mathrm{Pe}}\nabla^2\theta,\quad\mathrm{~in~}\Omega 
$$
等式左侧为热对流项，右侧为热扩散项

- 流动的连续性方程:
$$
\nabla\cdot\mathbf{u}=0,\quad\text{in}\Omega 
$$

- 边界条件：
$$
\theta=\theta_{\Gamma},\quad\mathrm{on}\partial\Omega_{\Gamma}
$$
$$
u=u_{\Gamma},\quad\mathrm{on}\partial\Omega_{\Gamma}
$$

式中，$\boldsymbol{u}$ 为速度矢量，$\theta$ 是温度标量，$p$ 是压强；
$\Omega$ 和 $\Gamma_D$ 分别表示计算的内部区域和狄利克雷边界。

$Re$ 为流动的**雷诺数**：
$$Re=\frac{U L}{\nu}$$
其中 $U$ 和 $L$ 分别为特征速度与特征长度，$\nu$ 为运动粘度。

$Pe$ 为**贝克来数**，$Pr$ 为**普朗特数**：
$$Pe=Re\cdot Pr,\quad Pr=\frac{\mu c_p}{\lambda}$$
其中 $\mu$ 为动力粘滞系数，$c_p$ 为定压比热，$\lambda$ 为热传导系数。$Ri$ 为**理查森数**,$Ri$ 取值适中时表示混合对流传热。

### 误差形式

1. X 方向动量方程残差
$$
e_1 = U \frac{\partial U}{\partial X} + V \frac{\partial U}{\partial Y} + \frac{\partial P}{\partial X} - \left( \frac{1}{Re} + \nu_{E1} \right) \left( \frac{\partial^2 U}{\partial X^2} + \frac{\partial^2 U}{\partial Y^2} \right)
$$

2. Y 方向动量方程残差
$$
e_2 = U \frac{\partial V}{\partial X} + V \frac{\partial V}{\partial Y} + \frac{\partial P}{\partial Y} - \left( \frac{1}{Re} + \nu_{E1} \right) \left( \frac{\partial^2 V}{\partial X^2} + \frac{\partial^2 V}{\partial Y^2} \right) - Ri\theta
$$

 3. 能量方程 + 连续性方程残差
$$
e_3 = U \frac{\partial \theta}{\partial X} + V \frac{\partial \theta}{\partial Y} - \left( \frac{1}{Pe} + \nu_{E2} \right) \left( \frac{\partial^2 \theta}{\partial X^2} + \frac{\partial^2 \theta}{\partial Y^2} \right)
$$
 4. 连续性方程残差
$$
e_4 = \frac{\partial U}{\partial X} + \frac{\partial V}{\partial Y}
$$

其中，上面式中 $\nu_{E1}$ 和 $\nu_{E2}$ 是在训练过程中确定的人工粘度。注意，参数 $\nu_{E1}$ 和 $\nu_{E2}$ 均为标量，它的构造是来自于熵粘性方法，在传统的数值模拟用于实现高 $Re$ 和 $Pr$ 的数值模拟稳定。在本案例中使用了人工粘性参数化模型方法来获得 $\nu_{E1}$ 和 $\nu_{E2}$，该方法不使用任何的标签数据。具体来说，在人工粘性参数化模型中，参数 $\nu_{E1}$ 和 $\nu_{E2}$ 可以通过下面计算得到：
$$
\nu_{E1} = \min\left(\beta_1 \nu, \alpha \frac{|r_1| L^2}{U_\infty^2}\right),
$$

$$
\nu_{E2} = \min\left(\frac{\beta_2 \nu}{\text{Pr}}, \alpha \frac{|r_2| L^2}{U_\infty^2}\right).
$$

其中$r_{1}$和$r_{2}$为神经网络预测的熵残差，$r_{1}$和$r_{2}$是可调参数，计算为如下所示

$$
\boldsymbol{r}_1 = (u - u_m)\boldsymbol{e}_1 + (v - v_m)\boldsymbol{e}_2
$$

$$
\boldsymbol{r}_2 = (\theta - \theta_m)\boldsymbol{e}_3
$$
我们将上面两个式子插入到神经网络的损失中，得到如下的残差形式：
$$
\boldsymbol{e}_5 \equiv (u - u_m)\boldsymbol{e}_1 + (v - v_m)\boldsymbol{e}_2 - \boldsymbol{r}_1
$$

$$
\boldsymbol{e}_6 \equiv (\theta - \theta_m)\boldsymbol{e}_3 - \boldsymbol{r}_2
$$

另外，在粘性参数公式中的$\beta_1$、$\beta_2$和$\alpha$为三个可调的超参数，这些取值会对PINNs训练的结果产生很大的影响;在整个训练过程中，我们设置$\beta_1$和$\beta_2$为恒定的超参数，但是$\alpha$的值是随着训练过程逐渐递减的。其中，$u_m$、$v_m$和$\theta_m$是三个常量，分别表示$u$、$v$和$\theta$的全局平均值，在本文中均采用$u_m = 0.5U_\infty$、$v_m = 0.5V_\infty$ 和 $\theta_m = 0.5\theta_\infty$。下面定义全部的损失函数，设置如下。
$$
\arg \min L = L_b + L_e + L_s
$$

$$
L_b = \lambda_b \left( \sum_{n=1}^{N_\Gamma} \left| \mathbf{u}(\mathbf{x}_n) - \mathbf{u}_\Gamma(\mathbf{x}_n) \right|^2 \right) + \left( \sum_{n=1}^{N_\Gamma} \left| \theta(\mathbf{x}_n) - \theta_\Gamma(\mathbf{x}_n) \right|^2 \right)
$$

$$
L_e = \sum_{i=1}^{4} \left( \lambda_{ei} \sum_{n=1}^{N_e} \left| eq_i(\mathbf{x}_n) \right|^2 \right)
$$

$$
L_s = \sum_{n=1}^{N_e} \left| \lambda_{s1} eq_5(\mathbf{x}_n) + \lambda_{s2} eq_6(\mathbf{x}_n) \right|^2
$$

其中，$L_b$、$L_e$ 和 $L_s$ 分别表示在边界上构建的损失函数、控制方程的损失函数、人工粘性的损失函数。此外，在损失函数中的权重 $\lambda_b = 10, \lambda_{e1} = \lambda_{e4} = 1, \lambda_{s1} = 0.1$ 设置为固定值，$\lambda_{e2}, \lambda_{e3}, \lambda_{s2}$ 的值需要根据问题进行调整。$N_e$ 是控制方程在内部区域 $\Omega$ 取样点的个数，$N_\Gamma$ 是在边界 $\partial\Omega_\Gamma$ 上样本点的数目，如下图所示，其中整个模型需要构建三个独立神经网络，其中第一个神经网络输出 $\mathbf{u}, p, \theta$，第二个神经网络输出 $r_1$，第三个神经网络输出 $r_2$。

**带有人工粘性的物理信息神经网络示意图**

![caseMarkdown/case_d749f814/viz/带有人工粘性的PINN.png](https://www.science42.tech/cases/caseMarkdown/case_d749f814/viz/带有人工粘性的PINN.png)
### 代码实现
本案例共包含4个文件，其中train.py是模型主训练程序，cavity_data.py是是训练数据和验证数据提供程序，pinn_solver.py是设置设置损失和模型架构的程序，tools.py是拉丁超立方体采样程序。
#### cavity_data.py
```
import os
import numpy as np
import scipy.io
from tools import LHSample
from tools import sort_pts

class DataLoader:
    def __init__(self, path=None, N_f=20000, N_b=1000):
        self.N_b = N_b
        self.x_min = 0.0
        self.x_max = 1.0
        self.y_min = 0.0
        self.y_max = 1.0
        self.N_f = N_f # equation points
        self.pts_bc = None

    def loading_boundary_data(self):  #边界条件设定
        Nx = 2500
        Ny = 2500
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

    def loading_training_data(self):  #训练点设定
        xye = LHSample(2, [[self.x_min, self.x_max], [self.y_min, self.y_max]], self.N_f)
        if self.pts_bc is not None:
            xye_sorted, _ = sort_pts(xye, self.pts_bc)
        else:
            print("need to load boundary data first!")
            raise 
        x_train_f = xye_sorted[:, 0:1]
        y_train_f = xye_sorted[:, 1:2]
        return x_train_f, y_train_f

    def loading_evaluate_data(self, filename):  #验证数据点设定
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
```
#### tools.py
```
import math
import numpy as np
import matplotlib.pyplot as plt



def LHSample(D, bounds, N):
    # """
    # :param D: Number of parameters
    # :param bounds:  [[min_1, max_1],[min_2, max_2],[min_3, max_3]](list)
    # :param N: Number of samples
    # :return: Samples
    # """
    result = np.empty([N, D])
    temp = np.empty([N])
    d = 1.0 / N
    for i in range(D):
        for j in range(N):
            temp[j] = np.random.uniform(low=j * d, high=(j + 1) * d, size=1)[0]
        np.random.shuffle(temp)
        for j in range(N):
            result[j, i] = temp[j]
    # Stretching the sampling
    b = np.array(bounds)
    lower_bounds = b[:, 0]
    upper_bounds = b[:, 1]
    if np.any(lower_bounds > upper_bounds):
        print('Wrong value bound')
        return None
    #   sample * (upper_bound - lower_bound) + lower_bound
    np.add(np.multiply(result, (upper_bounds - lower_bounds), out=result),
           lower_bounds,
           out=result)
    return result

def distance(p1, p2):
    "return the distance between two points"
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def minDistance(pt, pts2):
    "return the min distance between one point and a set of points"
    dists = [distance(pt, i) for i in pts2]
    return min(dists)

def sort_pts(pts1, pts2, flag_reverse=False):
    "sort a set of points based on their distances to another set of points"
    minDists = []
    for pt in pts1:
        minDists.append( minDistance(pt, pts2) )
    minDists = np.array(minDists).reshape(1,-1)
    
    dists_sorted = np.sort(minDists).reshape(-1,1)
    sort_index = np.argsort(minDists)
    if flag_reverse:
        sort_index = sort_index.reshape(-1,1)
        sort_index = sort_index[::-1].reshape(1,-1)
        dists_sorted = dists_sorted[::-1]
    pts1_sorted = pts1[sort_index,:]
    pts1_sorted = np.squeeze(pts1_sorted)
    return pts1_sorted, dists_sorted
```
#### pinn_solver.py
```
import os
import torch
import scipy.io
import numpy as np
from net import FCNet
from typing import Dict, List, Set, Optional, Union, Callable
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PysicsInformedNeuralNetwork:  #定义模型类
    # Initialize the class
    def __init__(self,     #初始化模型参数
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

        # initialize NN  神经网络初始化，共需要三个神经网络
        self.net = self.initialize_NN(
                num_ins=num_ins, num_outs=num_outs, num_layers=layers, hidden_size=hidden_size).to(device)
        self.net_1 = self.initialize_NN(
                num_ins=num_ins, num_outs=num_outs_1, num_layers=layers_1, hidden_size=hidden_size_1).to(device)
        self.net_2 = self.initialize_NN(
                num_ins=num_ins, num_outs=num_outs_2, num_layers=layers_2, hidden_size=hidden_size_2).to(device)
        
        self.opt = torch.optim.Adam(
            list(self.net.parameters())+list(self.net_1.parameters())+list(self.net_2.parameters()),
            lr=learning_rate,
            weight_decay=0.0) if not opt else opt

    def init_vis_t(self):  #初始化人工粘性系数
        (_, _, _, _, e0, e1) = self.neural_net_u(self.x_f, self.y_f)
        self.vis_t0_minus  = self.alpha_evm*torch.abs(e0).detach().cpu().numpy()
        self.vis_t1_minus  = self.alpha_evm*torch.abs(e1).detach().cpu().numpy()

    def set_boundary_data(self, X=None, time=False):  #设置边界条件
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
        # 打印结果
        # print('x_test的元素个数:', num_elements)
        # print('元素个数的平方根:', sqrt_num)
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
        
        # 保存.mat文件到新建文件夹中
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


```
#### train.py
```
import torch
from tools import *
import os
import cavity_data as cavity
import pinn_solver as psolver

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def train(net_params=None,loop=None):
    Re = 2000   # Reynolds number
    Pr = 0.71
    Ri = 0.1
    N_neu = 120
    N_neu_1 = 40
    N_neu_2 = 40
    lam_bcs = 10
    lam_equ = 1
    N_f = 400
    N_b = 25
    alpha_evm = 0.03
    N_HLayer = 4
    N_HLayer_1 = 4
    N_HLayer_2 = 4

    PINN = psolver.PysicsInformedNeuralNetwork(
        Re=Re,
        Ri=Ri,
        Pr=Pr,
        layers=N_HLayer,
        layers_1=N_HLayer_1,
        layers_2=N_HLayer_2,
        hidden_size = N_neu,
        hidden_size_1 = N_neu_1,
        hidden_size_2 = N_neu_2,
        N_f = N_f,
        alpha_evm=alpha_evm,
        bc_weight=lam_bcs,
        eq_weight=lam_equ,
        net_params=net_params,
        checkpoint_path='./checkpoint/')

    path = os.path.join(BASE_DIR, "datasets/")
    dataloader = cavity.DataLoader(path=path, N_f=N_f, N_b=N_b)

    # Set boundary data, | u, v, x, y
    boundary_data = dataloader.loading_boundary_data()
    PINN.set_boundary_data(X=boundary_data)

    # Set training data, | x, y
    training_data = dataloader.loading_training_data()
    PINN.set_eq_training_data(X=training_data)

    filename = './data/cavity_Re'+str(Re)+'Pr0.71Ri0.1_256.mat'
    x_star, y_star, u_star, v_star, t_star = dataloader.loading_evaluate_data(filename)

    # Training
    epoch=300000
    PINN.set_alpha_evm(0.05)
    PINN.train(num_epoch=epoch, lr=1e-3)
    PINN.test(x_star, y_star, u_star, v_star, t_star, 300000, loop)

    PINN.set_alpha_evm(0.03)
    PINN.train(num_epoch=epoch, lr=2e-4)
    PINN.test(x_star, y_star, u_star, v_star, t_star, 600000, loop)

    PINN.set_alpha_evm(0.02)
    PINN.train(num_epoch=epoch, lr=5e-5)
    PINN.test(x_star, y_star, u_star, v_star, t_star, 900000, loop)

    PINN.set_alpha_evm(0.01)
    PINN.train(num_epoch=epoch, lr=5e-5)
    PINN.test(x_star, y_star, u_star, v_star, t_star, 1200000,loop)

    PINN.evaluate(x_star, y_star, u_star, v_star, t_star)

if __name__ == "__main__":
    for loop in range(0,1):
        train(loop=loop)

```

## 计算结果
### 人工粘性PINNs与PINNs计算结果对比
两种PINN的计算结果如下图所示，左、中、右三列分别代表CFD计算参考结果、PINN求解结果和带有人工粘性的PINN求解结果。结果表明，对于速度场和温度场，人工粘性PINNs的求解结果与CFD结果十分吻合，而传统PINNs求解的速度场和温度场结果均向右上角发了偏移，传统PINNs模拟强迫对流传热问题也得到相同的现象。

**人工粘性的物理信息神经网络计算结果对比图**

![caseMarkdown/case_d749f814/viz/PINN与人工粘性PINN对比.png](https://www.science42.tech/cases/caseMarkdown/case_d749f814/viz/PINN与人工粘性PINN对比.png)

进一步对比模拟结果，人工粘性的PINNs能准确模拟出流场左右下角的二阶涡，左上角形成涡的趋势，特别的能准确模拟出右下角的三阶涡，传统PINNs模拟的流场结果呈现无序性。

**流线计算结果对比图**

![caseMarkdown/case_d749f814/viz/4211f9a4393a119c6d614020e157609b.png](https://www.science42.tech/cases/caseMarkdown/case_d749f814/viz/4211f9a4393a119c6d614020e157609b.png)


## 常见问题
- **多方程耦合导致的约束冲突**：
二维混合对流需同时嵌入连续性方程、动量方程（x/y 方向）、能量方程，PINN 需同时满足多方程约束，易出现 “满足某一方程却违背另一方程” 的情况（如满足能量守恒但动量不守恒），尤其在对流项占比高（强制对流主导）时，约束平衡难度显著增加。

- **对流项离散与数值耗散矛盾**：对流项离散与数值耗散矛盾：二维混合对流中对流项（如 ρu∂T/∂x）非线性强，PINN 通过自动微分求解梯度时，易产生数值耗散或振荡，导致温度场、速度场预测失真，尤其在高雷诺数（强制对流主导）场景下，该问题更突出。

- **样本分布不合理导致的拟合偏差**：若训练样本（collocation points）在物理域内分布不均（如边界层区域样本过少），PINN 难以捕捉局部物理特性（如边界层内的温度梯度、速度梯度），导致预测结果在关键区域误差过大；反之，样本过多会大幅增加计算成本，降低训练效率。

## 工业价值
### 优点
1. 降低研发成本，缩短产品迭代周期
传统二维混合对流换热仿真（如 CFD）需大量网格划分、长时间迭代计算，且依赖专业工程师操作；实验测试则需搭建物理样机、消耗大量能源与物料。PINN 无需复杂网格离散，可直接嵌入 Navier-Stokes 方程、能量方程等物理约束，结合少量实验数据即可完成仿真，大幅减少计算时间（尤其小尺度、复杂边界场景，计算效率提升 50% 以上），同时避免反复搭建实验样机，显著降低研发成本，加快产品迭代速度。
2. 提升换热系统设计精度，优化产品性能
二维混合对流广泛存在于工业换热设备（如换热器、散热器、管道）中，PINN 可精准捕捉流场、温度场的局部特性（如边界层梯度、涡流区域、温度突变点），解决传统 CFD 在高雷诺数、复杂边界下数值耗散大、精度不足的问题，助力工程师优化换热结构（如通道形状、流道布局、散热片设计），提升换热效率、降低能耗，同时避免局部过热导致的设备损坏，延长产品使用寿命。
3. 适配复杂工况，拓展工业应用场景
工业中的二维混合对流常面临变物性（流体参数随温度 / 速度变化）、复杂边界（非均匀壁面、不规则流道）、瞬态工况（如启停过程中的温度波动）等难题，传统方法适配性差、计算难度高。PINN 具备强非线性拟合能力，可灵活嵌入复杂物理约束，轻松适配上述工况，覆盖更多工业场景，尤其适用于传统方法难以求解的小众、特殊工况（如航空航天设备中的局部换热、电子设备的微型散热）。

### 典型工业应用场景
1. 电子设备工业：微型换热器、芯片散热器的二维混合对流仿真，优化散热结构，解决电子设备高温卡顿、寿命缩短问题（如新能源汽车芯片、服务器散热器）。
2. 化工与能源工业：管道换热、塔式反应器的二维混合对流分析，优化流道设计，提升换热效率，降低能源消耗（如化工原料预热、余热回收系统）。
3. 航空航天工业：飞行器机舱、发动机舱的局部二维混合对流仿真，优化散热布局，保障设备在极端工况下的热稳定性（如倾转旋翼飞行器的电机散热）。
4. 暖通与制冷工业：空调换热器、地暖管道的二维混合对流优化，提升换热效率，降低空调、地暖的能耗，实现节能降耗。
核工业：核反应堆冷却管道的二维混合对流监测，实时预判换热异常，保障反应堆运行安全。


