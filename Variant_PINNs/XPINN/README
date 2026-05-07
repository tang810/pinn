# Extended Physics-InformedNeural Networks (XPINNs)

## 背景介绍：
准确地求解复杂的方程组，特别是在高维情况下，已经成为科学计算中最大的挑战之一，XPINNS 的优势使其成为进行高维复杂模拟的合适选择，而这类模拟通常需要很高的训练成本。

## XPINNS原理：
将物理知识融入神经网络中，以解决非线性偏微分方程。它结合了物理方程和神经网络的优势，通过将空间和时间域分解为子域，并在每个子域中训练神经网络来近似解决方案。这种方法可以提高对复杂非线性问题的建模能力，并在求解过程中保持物理一致性。XPINNs的关键思想是利用物理方程的结构信息来指导神经网络的训练，从而提高模型的准确性和泛化能力。


![caseMarkdown/XPINN/viz/XPINN.png](https://www.science42.tech/cases/caseMarkdown/XPINN/viz/XPINN.png)

## 创新点：

广义时空域分解：
XPINN 公式提供了具有规则边界或高度不规则、凸/非凸时空域分解。 这种域分解在许多应用中都很有用，例如多物理/多尺度计算、涉及裂缝、冲击波等非平滑特征的模拟。由于这种分解，XPINN 方法很容易适用于时空并行化， 从而更有效地降低培训成本。

扩展到任何微分方程：
与 cPINN 方法不同，基于 XPINN 的域分解方法可以扩展到任何类型的偏微分方程，无论其物理性质如何。 这样，任何微分方程都可以有效求解，这使得 XPINN 方法成为真正的基于广义域分解的 PINN 方法，并且可以轻松并行化。

简单的界面条件：
由于不规则的域分解，界面形成高度不规则的形状，尤其是在更高维度中。 在 XPINN 中，对于任意形状的界面，界面条件都非常简单，不需要法线方向，因此，所提出的方法可以轻松扩展到任何复杂的几何形状，甚至在更高的维度上。 而且，这种简单的界面条件在界面移动的动态界面问题中非常有用。



具体可见原论文链接：
https://ceur-ws.org/Vol-2964/article_60.pdf
      
## 1、控制方程
   2D Poisson方程：
$$
{-}\Delta u = f, \quad (x,y) \in [-1, 1]^2,
$$

  特定解：
 $$
u(x,y) = 0.1\sin(4\pi x)\sin(4\pi y) + \tanh(5x)\tanh(5y),
$$

  方程求解：
$$
f(x,y) = 3.2\pi^2 \sin(4\pi x)\sin(4\pi y) + 50\operatorname{sech}^2(5x)\tanh(5y) + 50\operatorname{sech}^2(5y)\tanh(5x).
$$

## 2、参数、属性定义：

| 参数/属性                  | 说明                        | 参数/属性                   | 说明                        |
| :---:                      | :---:                       | :---:                       | :---:                       |
| X_u                        | 边界点坐标                  | u                           | 边界条件                    |
| X_f1, X_f2, X_f3           | 三个子域的残差点            | X_fi1, X_fi2                | 界面点坐标                  |
| layers                     | 每个子网络的层配置          | device                      | 边界点的时间空间坐标        |
| x, t                       | 边界坐标                    | x1, t1, x2, t2, x3, t3       | 子域的时间空间残差点        |
| xi1, ti1, xi2, ti2         | 界面时间空间坐标            | weights, biases             | 网络权重和偏置              |
| loss_list, error_list      | 损失和误差列表              |                             |                             |

软件版本要求：TensorFlow 1.14, Python 3.6

### 所需要导入的库：
```python
import torch
import numpy as np
from pyDOE import lhs
import matplotlib.pyplot as plt
import scipy
```


## 3、XPINNs 类方法：

因为数据众多，且有着多个网络同时训练，因此我们将从0开始详细编写网络。\
在这个类下面一共有如下的方法。
```python
class XPINNs:
    # 初始化
    def __init__(self, X_u, u, X_f1, X_f2, X_f3, X_fi1, X_fi2, layers):
        ···
    # 初始化网络权重， 这里使用的是Xavier初始化
    def initialize_NN(self, layer):
        ···
        return weights, biases
    # 对矩阵进行初始化 (Xavier初始化)
    def xavier_init(self, size):
        ···
        return weight
    # 前向传播过程
    def neural_net(self, X, weights, biases):
        ···
        return ouput
    # 前向传播, 分开变量的形式
    def net_u(self, x, t, weights, biases):
        ···
        return output
    # Loss PDE
    def net_f(self, x1, t1, x2, t2, x3, t3, xi1, ti1, xi2, ti2):
        ···
        return loss， interface_condition
    # 总误差 Loss
    def Loss(self, x, t, u, x1, t1, x2, t2, x3, t3, xi1, ti1, xi2, ti2):
        ···
        return loss
    # lbfgs的闭包方法
    def closure(self):
        ···
        return loss
    def train(self, epoch):
        ···
    def predict(self, x, t):
        ···
        return u
```
### 3.1  init 方法及代码介绍

```python
   def __init__(self, X_u, u, X_f1, X_f2, X_f3, X_fi1, X_fi2, layers):

        # CUDA support 
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # boundary
        self.x = torch.tensor(X_u[:, 0:1], requires_grad=True).float().to(self.device)
        self.t = torch.tensor(X_u[:, 1:2], requires_grad=True).float().to(self.device)
        self.u = torch.tensor(u, requires_grad=True).float().to(self.device)

        # train 这里有多个网络，因此有多个数据
        self.x1 = torch.tensor(X_f1[:, 0:1], requires_grad=True).float().to(self.device)
        self.t1 = torch.tensor(X_f1[:, 1:2], requires_grad=True).float().to(self.device)

        # interface     多个网络之间的边界
        self.xi1 = torch.tensor(X_fi1[:, 0:1], requires_grad=True).float().to(self.device)
        self.ti1 = torch.tensor(X_fi1[:, 1:2], requires_grad=True).float().to(self.device)

        self.loss_list, self.error_list = [], []
        self.activation = torch.tanh

        ## 初始化各个网络
        self.weights1 ,self.biases1 = self.initialize_NN(layers[0])
```

### 3.2 initialize_NN 方法及代码介绍
初始化神经网络的权重和偏置：

```python
    # 获得网络参数
    def initialize_NN(self, layer):
        weights, biases = [], []

        for i in range(len(layer) - 1):
            w = self.xavier_init(size=[layer[i], layer[i+1]])
            b = torch.zeros((1, layer[i+1]), requires_grad=True)

            weights.append(w)
            biases.append(b)
        return weights, biases
```

### 3.3 xavier_init方法及代码介绍
使用Xavier初始化方法初始化权重：

```python
# 对矩阵进行初始化 (μ=0 的正态分布)
    def xavier_init(self, size):
        in_dim, out_dim = size
        xavier_stddev = (2/(in_dim + out_dim))**(1/2)
        x = torch.normal(mean=0, std=xavier_stddev*torch.ones(in_dim, out_dim))
        x.requires_grad = True
        return x
```

### 3.4 neural_net 方法及代码介绍
执行网络的前向传播：

```python
    # 前向传播过程
    def neural_net(self, X, weights, biases):
         H = X
         for i in range(len(weights) - 1):
             w, b = weights[i].float().to(self.device), biases[i].float().to(self.device)
             H = self.activation(torch.mm(H, w) + b)
         w, b = weights[-1].float().to(self.device), biases[-1].float().to(self.device)
         return torch.mm(H, w) + b 
```

### 3.5 net_u 方法及代码介绍
为给定的时间空间坐标执行子网络的前向传播：

```python
# 前向传播
    def net_u(self, x, t, weights, biases):
        return self.neural_net(torch.cat([x,t], 1), weights, biases)
```

### 3.6 net_f 方法及代码介绍
计算所有子网络的残差，包括界面连续性条件：
这里的公式为：

![caseMarkdown/XPINN/viz/XPINN_loss.jpg](https://www.science42.tech/cases/caseMarkdown/XPINN/viz/XPINN_loss.jpg)

```python
    # Loss PDE
    def net_f(self, x1, t1, x2, t2, x3, t3, xi1, ti1, xi2, ti2):

        # Sub-net 1
        u1 = self.net_u(x1, t1, self.weights1, self.biases1)
        u1_t = torch.autograd.grad(u1, t1, grad_outputs=torch.ones_like(u1), retain_graph=True, create_graph=True)[0]
        u1_x = torch.autograd.grad(u1, x1, grad_outputs=torch.ones_like(u1), retain_graph=True, create_graph=True)[0]
        u1_tt = torch.autograd.grad(u1_t, t1, grad_outputs=torch.ones_like(u1_t), retain_graph=True, create_graph=True)[0]
        u1_xx = torch.autograd.grad(u1_x, x1, grad_outputs=torch.ones_like(u1_x), retain_graph=True, create_graph=True)[0]
    
```
### 3.7 Loss 方法及代码介绍
训练模型，使用Adam优化器进行初始训练，然后使用LBFGS优化器进行精细调整：
```python
# Funcrion Loss
    def Loss(self, x, t, u, x1, t1, x2, t2, x3, t3, xi1, ti1, xi2, ti2):

        f1, f2, f3, fi1, fi2, uavgi1, uavgi2, u1i1, u1i2, u2i1, u3i2 = \
            self.net_f(x1, t1, x2, t2, x3, t3, xi1, ti1, xi2, ti2)
        
        loss1 = 20*torch.mean((self.net_u(x, t, self.weights1, self.biases1) - u)**2) + \
            torch.mean(f1**2) + torch.mean(fi1**2) + torch.mean(fi2**2) \
                + 20*torch.mean((u1i1 - uavgi1)**2) + 20*torch.mean((u1i2 - uavgi2)**2)
        
        loss2 = torch.mean(f2**2) + torch.mean(fi1**2) + 20*torch.mean((u2i1 - uavgi1)**2)
        
        loss3 = torch.mean(f3**2) + torch.mean(fi2**2) + 20*torch.mean((u3i2 - uavgi2)**2)
        
        return loss1, loss2, loss3
```
### 3.8 closure 方法及代码介绍
优化器闭包函数，用于LBFGS优化器。
```python
 def closure(self):

        self.optimizer.zero_grad()
        self.loss1, self.loss2, self.loss3 = self.Loss(self.x, self.t, self.u, self.x1, self.t1, self.x2, self.t2, self.x3, self.t3, \
                         self.xi1, self.ti1, self.xi2, self.ti2)
            
        loss = self.loss1 + self.loss2 + self.loss3

        self.loss_list.append(loss.item())

        if self.iter % 100 == 0:
            print('loss: %.3e' %(loss.item()))
        loss.backward()
        self.iter += 1
        return loss    
```
### 3.9 train 方法及代码介绍
训练模型，使用Adam优化器进行初始训练，然后使用LBFGS优化器进行精细调整：
```python
def train(self, epoch):

        self.iter = 0        
        self.optimizer_Adam = torch.optim.Adam(self.weights1 + self.biases1 + self.weights2 + self.biases2 + self.weights3 + self.biases3, lr=8e-4)

        for i in range(epoch):
            self.loss1, self.loss2, self.loss3 = self.Loss(self.x, self.t, self.u, self.x1, self.t1, self.x2, self.t2, self.x3, self.t3, \
                         self.xi1, self.ti1, self.xi2, self.ti2)
            
            loss = self.loss1 + self.loss2 + self.loss3
            
            self.optimizer_Adam.zero_grad()
            loss.backward()
            self.optimizer_Adam.step()

            if self.iter % 1 == 0:
                print('Iter:%d, loss1:%.3e, loss2:%.3e, loss3:%.3e'%(i, self.loss1.item(), self.loss2.item(), self.loss2.item()))
        self.optimizer = torch.optim.LBFGS(self.weights1 + self.biases1 + self.weights2 + self.biases2 + self.weights3 + self.biases3, 1, max_iter = 10000, max_eval = None, 
                        tolerance_grad = 1e-11, tolerance_change = 1e-11, history_size = 100, line_search_fn = 'strong_wolfe')
        self.optimizer.step(self.closure)
```
###3.10 predict 方法及代码介绍
进行预测，返回三个子域的解：
```python
def predict(self, x, t):
        x = torch.tensor(x).float().to(self.device)
        t = torch.tensor(t).float().to(self.device)

        u1 = self.net_u(x, t, self.weights1, self.biases1).cpu().detach().numpy()
        u2 = self.net_u(x, t, self.weights2, self.biases2).cpu().detach().numpy()
        u3 = self.net_u(x, t, self.weights3, self.biases3).cpu().detach().numpy()
        return u1, u2, u3
```



##  4、数据加载和处理
根据已给的数据，加载训练数据（边界点、残差点和界面点），随机选择残差点和边界点：
```python
# Boundary points from subdomian 1
N_ub   = 200

# Residual points in three subdomains
N_f1   = 5000
N_f2   = 1800
N_f3   = 1200

# Interface points along the two interfaces
N_I1   = 100
N_I2   = 100

# NN architecture in each subdomain
layers1 = [2, 30, 30, 1]
layers2 = [2, 20, 20, 20, 20, 1]
layers3 = [2, 25, 25, 25, 1]
layers = [layers1, layers2, layers3]

# Load training data (boundary points), residual and interface points from .mat file
# All points are generated in Matlab
data = scipy.io.loadmat('DATA/XPINN_2D_PoissonEqn.mat')



```


## 5、模型训练
实例化 XPINNs 类并训练模型：
```python 
model = XPINNs(X_ub_train ,ub_train,X_f1_train,X_f2_train,X_f3_train,\
                              X_fi1_train, X_fi2_train, layers)
model.train(000)
```

## 6、 预测、绘图及成果展示
对整个域进行预测，计算误差：


结果2D Possoin方程的预测结果，左1为精确解, 中间为预测解，右1为逐点误差图。\
其中横坐标为x, 纵坐标为有, 颜色深浅为u。

![caseMarkdown/XPINN/viz/10.png](https://www.science42.tech/cases/caseMarkdown/XPINN/viz/10.png)



## 7、常见问题（FAQ）

### 7.1 加载保存的模型：
由于这里的模型的权重与偏差都是手写的，因此会相对复杂。后续可以采用torch的官方写法，保存模型且保存模型会方便很多，首先在2D_Possion.py 中加入如下代码，可以保存模型已经训练好的权重与偏差。后续通过load_model.py即可读取模型并进行预测。这里对load_model.py进行简单介绍。

### 7.2模型的定义：
因为我们只知道权重与偏差，因此前向传播与预测的方法需要自编。
参数：
path 模型的权重与偏差存放的位置。他根据上方保存的格式保存，因此读取也需要相同格式处理。
方法：
neural_net： 处理一次前向传播的方法，给予输入X，权重wights, 偏差biases即可进行一次前向传播
net_u: 这个是将输入进行合并后，进行前向传播的方法。
predict： 进行预测。输出为np.array格式。
## 8、总结与扩展
XPINN 的主要优势在于，它可以轻松应用于涉及复杂域的任何复杂模拟，尤其是在高维情况下。总体而言，所提出的 XPINN 方法是 PINN 和 cPINN 方法的推广，无论在适用性还是域分解技术方面，都能有效实现并行计算。

