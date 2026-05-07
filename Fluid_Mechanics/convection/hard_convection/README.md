# PINN和PIKAN用于求解强迫对流传热问题

该目录下一共有四个文件夹，分别是01HCFNets；01ev_HCFNets；02ev_HCKANets；02HCKANets.

01HCFNets，是原始的PINNs用于求解方腔流强迫对流传热问题

01ev_HCFNets，是人工粘性的PINNs用于求解方腔流强迫对流传热问题

02HCFNets，使用KAN神经网络来构成PIKANs用于求解方腔流强迫对流传热问题

02ev_HCFNets，使用人工粘性的PIKANs用于求解方腔流强迫对流传热问题

## 介绍

本教程将引导您完成如何使用PINNs来实现对强迫对流传热问题的求解，了解对流传热问题的相关问题，在本教程中，您将学习到以下的内容
1、如何编写自己的偏微分如何编写自己的偏微分方程的条件
2、如何设置自己的求解参数
3、如何将自己的求解结果进行输出用于结果的后处理

## PINN和PIKAN

### PINN求解示意图

### PIKAN求解示意图

![PIKAN.png](./viz/PIKAN.png)

PINN和PIKAN的区别在于PINN使用的MLP全连接神经网络;而PIKAN使用的是KAN网络

## 问题描述

### 物理问题的描述

对流传热是指热量通过流体（液体或者气体）传递的过程。在这个过程中， 
热量是通过流体进行传递而不是直接通过导体或辐射的方式。对流传热可以通过两种主要方式发生：强迫对流和混合对流。这里介绍强迫对流传热问题，
强迫对流动传热问题（Forced Convective Heat Transfer Problem）是指通过外力驱动流体流动，
从而实现热量传递的过程。与自然对流不同，强迫对流的流体运动不是由于温度差引起的浮力作用，
而是由外部机械手段（如泵、风扇或压差）引起的。这种传热方式在工业和工程应用中非常普遍，特别是在冷却系统、加热器、空调和换热器中。
流动与传热问题中的偏微分方程主要是Navier-Stokes和能量守恒方程。

动量守恒方程

$$
(\mathbf{u} \cdot \nabla) \mathbf{u} = -\nabla p + \frac{1}{\mathrm{Re}} \nabla^2 \mathbf{u} \text{, in } \Omega
$$

能量守恒方程

$$
(\mathbf{u}\cdot\nabla)\theta=\frac{1}{Pe}\nabla^2\theta,\quad\mathrm{in}\Omega
$$

质量守恒方程

$$
\nabla\cdot\mathbf{u}=0,\quad\mathrm{in~}\Omega
$$


动量守恒方程和质量守恒方程共同组成NS方程，（2）是传热方程。其中有一些主要的参数，雷诺数Re，普朗特数Pr，Pe是贝克来数。Pe
=Re*Pr。

### 方腔强迫对流传热问题
![dingyi.png](./viz/dingyi.png)
## 导入所需要的包

本文件夹中的py文件是缺一不可得，所有的py文件的编写是基于pytorch框架。一共有五个py文件。
其中运行的主文件是train.py;
cavity_data.py文件是训练数据和验证数据提供程序；
net.py是网络设置程序，使用全连接神经网络MLP；
pinn_solver.py是设置设置物理损失和边界损失的程序；
tools.py是拉丁超立方体采样方式；
需要的环境是python>=3.9,pytorch>=2.0;
需要安装的Python包详见requirements.txt文件，运行
```python
pip install -r requirements.txt 
```

## train.py

```python
import cavity_data as cavity
import pinn_solver as psolver #导包
```

```python
if __name__ == "__main__":
    for loop in range(0, 1): #设置循环次数
        train(loop=loop)
```

使用人工粘性的设置

```python
epoch=300000 #训练步数
    PINN.set_alpha_evm(0.05) #人工粘性参数设置
    PINN.train(num_epoch=epoch, lr=1e-3)
    PINN.test(x_star, y_star, u_star, v_star, t_star, 300000, loop)
```

不使用人工粘性的设置

```python
epoch=300000
    PINN.train(num_epoch=epoch, lr=1e-3)
    PINN.test(x_star, y_star, u_star, v_star, t_star, 300000, loop)
```

其他注释见train.py文件

## cavity_data.py

导包

```python
import numpy as np
    import scipy.io
    from tools import LHSample
    from tools import sort_pts
```

边界/初始条件

```python
def loading_boundary_data(self):
        ...
        return x_b, y_b, u_b, v_b, t_b 
```
内部采点
```python
    def loading_training_data(self):
        ...
        return x_train_f, y_train_f
```

测试数据

```python
def loading_evaluate_data(self, filename):
        ...
        return x_star, y_star, u_star, v_star, p_star, t_star
```

其他注释见cavity_data.py文件

## net.py

导包

```python
import torch
    from collections import OrderedDict
```

```python
from kan import KAN   #使用KAN来求解
```

构建MLP网络模型

```python
class FCNet(torch.nn.Module): 
        def __init__(self, num_ins=3,
                     num_outs=3,
                     num_layers=10,
                     hidden_size=50,
                     activation=torch.nn.Tanh):
            super(FCNet, self).__init__()
            ```
        def forward(self, x):
            out = self.layers(x)
            return out
```

构建KAN网络模型

```python
class ChebyKANLayer(nn.Module):
       def __init__(self, input_dim, output_dim, degree):
          ...
       def forward(self, x):
          ...
    class ChebyKAN(nn.Module):
       def __init__(self):
          ...
       def forward(self, x):
          ...
```

根据自己的需求构建KAN网络，可以是多个

## pinn_solver.py

导包

```python
import os
    import torch
    import scipy.io
    import numpy as np
    from net import FCNet
    from from net import ChebyKAN
    from net import ChebyKAN_1
    from net import ChebyKAN_2
    from typing import Dict, List, Set, Optional, Union, Callable
```

设置设备

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**PINNs 中增加人工粘性需要增添的部分**

```python
self.vis_t0 = 10.0/self.Re
    self.vis_t1 = 7.1/(self.Re * self.Pr)
```

```python
def init_vis_t(self):
        """
        初始化人工粘性相关参数
        """
        (_, _, _, _, e0, e1) = self.neural_net_u(self.x_f, self.y_f)
        self.vis_t0_minus  = self.alpha_evm*torch.abs(e0).detach().cpu().numpy()
        self.vis_t1_minus  = self.alpha_evm*torch.abs(e1).detach().cpu().numpy()
```

```python
def set_alpha_evm(self, alpha):
        """
        设置人工粘性相关参数。
        """
        self.alpha_evm = alpha
```

```python
def neural_net_equations(self, x, y):
        ...

        # Get the minum between (vis_t0, vis_t_mius(calculated with last step e))
        self.vis_e0 = torch.tensor(np.minimum(self.vis_t0, self.vis_t0_minus)).float().to(device)
        # Save vis_t_minus for computing vis_t in the next step
        self.vis_t0_minus  = self.alpha_evm*torch.abs(e0).detach().cpu().numpy()
        
        self.vis_e1 = torch.tensor(np.minimum(self.vis_t1, self.vis_t1_minus)).float().to(device)
        # Save vis_t_minus for computing vis_t in the next step
        self.vis_t1_minus  = self.alpha_evm*torch.abs(e1).detach().cpu().numpy()

        # NS
        eq1 = (u*u_x + v*u_y) + p_x - (1.0/self.Re + self.vis_e0)*(u_xx + u_yy)
        eq2 = (u*v_x + v*v_y) + p_y - (1.0/self.Re + self.vis_e0)*(v_xx + v_yy)
        eq3 = (u*t_x + v*t_y) - ((1.0/self.Re)*(1.0/self.Pr) + self.vis_e1)*(t_xx + t_yy)
        eq4 = u_x + v_y
        #粘性方程
        residual_1 = (eq1*(u-0.5) + eq2*(v-0.5)) - e0
        residual_2 = (eq3*(t-0.5)) - e1 
        return eq1, eq2, eq3, eq4, residual_1, residual_2
```

```python
def solve_Adam(self,
                   loss_func,
                   num_epoch=1000,
                   batchsize=None,
                   scheduler=None):
        """
        修改优化过程

        """
        self.freeze_evm_net(0)
        for epoch_id in range(num_epoch):
            # train evm net every 10000 step
            if epoch_id !=0 and epoch_id % 10000 == 0:
                self.defreeze_evm_net(epoch_id)
            if (epoch_id - 1) % 10000 == 0:
                self.freeze_evm_net(epoch_id)
            ···
```

```python
def freeze_evm_net(self, epoch_id):
        """
        设置对神经网络的冻结
        """
        for para in self.net_1.parameters():
            para.requires_grad = False
        self.opt.param_groups[0]['params'] = list(self.net.parameters())


    def defreeze_evm_net(self, epoch_id):
        """
        设置对神经网络的解冻
        """
        for para in self.net_1.parameters():
            para.requires_grad = True
        self.opt.param_groups[0]['params'] = list(self.net.parameters()) + list(self.net_1.parameters()) + list(self.net_2.parameters())
```

其他注释见pinns_solver.py文件

## tool.py

用于实现二维的拉丁超立方采样

```python
import math
    import numpy as np
```

## PINN功能介绍

### 参数的设置

PINN的参数设置主要是通过train.py文件进行实现，这其中包括雷诺数的设置，普朗特数的设置，神经网络的层数等参数

```python
def train(net_params=None, loop=None):
        Re = 2000 # Reynolds number
        Pr = 0.71 # Prandtl number
        layers=4  # 神经网络层数
        hidden_size=120 #神经元个数
        lam_bcs = 10 #边界loss的权重
        lam_equ = 1  #方程loss的权重
        N_f = 40000 #内部采样点个数
        N_b = 2500 #边界采样点个数
        loop=loop #遍历的次数
        ...
        filename = './data/cavity_Re'+str(Re)+'_Pr0.71_256.mat'
        # 验证数据集的读取
        x_star, y_star, u_star, v_star, p_star, t_star = dataloader.loading_evaluate_data(filename)
        # Training
        start_time = time.time()
        epoch=300000   #训练的步数
        PINN.train(num_epoch=epoch, lr=1e-3) #设置训练的次数和学习率   
        PINN.test(x_star, y_star, u_star, v_star, p_star, t_star, 300000, loop)
```

PINN.train()   #调用pinn_slover.py中的def train():实现对PINN的训练
PINN.test()    #调用pinn_slover.py中的def test():实现对PINN输出结果的评估并保存计算结果

### 损失的构建

损失包括边界条件的损失和方程的损失，损失的构建全部是在pinn_solver.py中实现
边界条件的损失,可以使用的构建方式有两种，一种是L2，一种是MSE

```python
def fwd_computing_loss_2d(self, loss_mode='MSE'):
        # boundary data
        (self.u_pred_b, self.v_pred_b, _, self.t_pred_b) = self.neural_net_u(self.x_b, self.y_b)
        # 使用self.neural_net_u(),得到边界点的输出
        # BC loss 边界的损失
        if loss_mode == 'L2':
            self.loss_b = torch.norm((self.u_b.reshape([-1]) - self.u_pred_b.reshape([-1])), p=2) + \
                          torch.norm((self.v_b.reshape([-1]) - self.v_pred_b.reshape([-1])), p=2) + \
                          torch.norm((self.t_b.reshape([-1]) - self.t_pred_b.reshape([-1])), p=2)
        if loss_mode == 'MSE':
            self.loss_b = torch.mean(torch.square(self.u_b.reshape([-1]) - self.u_pred_b.reshape([-1]))) + \
                          torch.mean(torch.square(self.v_b.reshape([-1]) - self.v_pred_b.reshape([-1]))) + \
                          torch.mean(torch.square(self.t_b.reshape([-1]) - self.t_pred_b.reshape([-1])))
```

方程的构建,可以使用的构建方式有两种，一种是L2，一种是MSE

```python
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
```

上面的损失对应的是每一个方程

```python
# 总损失
        self.loss = self.alpha_b * self.loss_b + self.alpha_e * self.loss_e
        return self.loss, [self.loss_e, self.loss_b]
```

### 创建输入和节点

输入部分是使用cavity_data.py,这里包括边界条件的输入，内部残差点的输入，测试点和真实数据的输入

```python
def __init__(self, path=None, N_f=20000, N_b=1000):
        ...
```

N_f: 残差点的数量; N_b: 边界点的数量

```python
def loading_boundary_data(self):
        ...
        return x_b, y_b, u_b, v_b, t_b
```

定义边界的约束，分别是边界的X,Y坐标值，对应下的边界上的速度u,v,t

```python
def loading_training_data(self):
        ...
        return x_train_f, y_train_f
```

输入残差点的坐标，分别输入的是X和Y的值

```python
def loading_evaluate_data(self, filename):
        ...
        return x_star, y_star, u_star, v_star, p_star, t_star
```

添加验证数据，输入数值方式的得到的真实解，用于对PINNs计算的结果进行评价。

### 输出并保存数据结果文件

此功能实现是依赖pinn_solver.py文件

```python
def test(self, x, y, u, v, p, t, num_epoch, loop=None):
        ...
```

上面的功能是实现对PINN的输出结果与真实结果之间计算L2误差，同时保存计算结果到.mat用于后处理

## PIKAN的功能介绍

### 参数设置

PIKAN和PINN参数设计不同之处是神经网络参数层数和神经元节点个数的设计

```python
class ChebyKAN(nn.Module):
    def __init__(self):
        super(ChebyKAN_1, self).__init__()
        self.chebykan1 = ChebyKANLayer(2, 32, 8)
        self.chebykan2 = ChebyKANLayer(32, 32, 8)
        self.chebykan3 = ChebyKANLayer(32, 3, 8)
    def forward(self, x):
        x = self.chebykan1(x)
        x = self.chebykan2(x)
        x = self.chebykan3(x)
        return x
```

PIKAN的设置是在net.py文件中实现的。在上面的设置中输入是二维，输出是三维的，网络设置三层，神经元的个数是32个。

## 结果展示

### PINN结果展示

针对Re=2000，Pr=0.71的参数得出求解结果
![pinn.png](./viz/pinn.png)
图中最左边CFD的是直接数值模拟提供的参考结果
图中间的是原始PINNs提供的结果
图最右边的是误差结果
![pinn_ev.png](./viz/pinn_ev.png)
图中最左边CFD的是直接数值模拟提供的参考结果
图中间的是人工粘性PINNs提供的结果
图最右边的是误差结果

对比计算的L2误差结果
![PINN_L2.png](./viz/PINN_L2.png)

### PIKAN结果展示

针对Re=2000，Pr=0.71的参数得出求解结果
![result_plot_kan.png](./viz/result_plot_kan.png)
图中最左边CFD的是直接数值模拟提供的参考结果
图中间的是原始PIKAN提供的结果
图最右边的是误差结果
![result_plot_evkan.png](./viz/result_plot_evkan.png)
图中最左边CFD的是直接数值模拟提供的参考结果
图中间的是人工粘性PIKAN提供的结果
图最右边的是误差结果

对比计算的L2误差结果
![PIKAN_l2.png](./viz/PIKAN_l2.png)

