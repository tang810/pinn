# 船舶磁场建模与仿真教程

本教程演示如何基于磁偶极子与椭球体模型计算船舶在传感器坐标系下的磁场分量。通过分步骤的问题背景、数学物理公式推导与代码使用示例，快速掌握项目核心功能。

---

## 1. 问题背景

在海洋环境中，船舶会产生磁异常信号。通过在水下部署磁传感器，测量不同位置的磁场（Hx, Hy, Hz），结合船体参数（船长 $L$、船宽 $W$、航向角 $\theta$），可逆向推断船舶的磁矩分布。本项目实现正问题：已知磁偶极子阵列与椭球体模型，计算指定传感器位置的磁场。

**场景示例**：
- 传感器坐标 (x, y, z)
- 船舶参数：船长 $L$, 船宽 $W$, 航向角 $\theta$
- 磁偶极子 15 个 + 椭球体模型 1 个，共 16 组磁矩向量

---

## 2. 数学与物理模型

### 2.1 磁偶极子模型

单个磁偶极子在空间点 $\mathbf r = (x,y,z)$ 处的磁场：
$$
\mathbf B(\mathbf r) = \frac{\mu_0}{4\pi}\left[ \frac{3(\mathbf m \cdot \mathbf r)\,\mathbf r}{\|\mathbf r\|^5} - \frac{\mathbf m}{\|\mathbf r\|^3} \right]
$$
- $\mu_0$：真空磁导率（$4\pi\times10^{-7}\,\mathrm{H/m}$）
- $\mathbf m = (m_x,m_y,m_z)$：偶极矩向量

对每个偶极子，分别计算三个分量系数：
$$
\begin{aligned}
a_x &= \frac{\mu_0}{4\pi}\Bigl(\frac{3\xi^2}{r^5}-\frac{1}{r^3}\Bigr),\\
a_y &= \frac{3\mu_0\,\xi\,\eta}{4\pi\,r^5},\\
a_z &= \frac{3\mu_0\,\xi\,\zeta}{4\pi\,r^5},
\end{aligned}
$$
其中 $(\xi,\eta,\zeta)=(x,y,z)$。

### 2.2 椭球体模型

将船体视为磁化椭球体，参数定义：
$$
K = \sqrt{\bigl(\tfrac{L}{2}\bigr)^2 - \bigl(\tfrac{W}{2}\bigr)^2},
$$
并引入：
$$
\begin{aligned}
t &= \sqrt{\bigl(r^2 + K^2\bigr)^2 - 4K^2 x^2},\\
A &= \sqrt{\tfrac12\bigl(r^2 + K^2 + t\bigr)},\quad
B = \sqrt{\tfrac12\bigl(r^2 - K^2 + t\bigr)}.
\end{aligned}
$$
其磁场分量推导可参考经典椭球体磁化理论，计算得到 $a_x, a_y, a_z$ 等系数后，与偶极子部分同理叠加。

### 2.3 坐标系转换

由船体系到传感器系的旋转矩阵：
$$
R(\theta) = \begin{pmatrix}
\cos\theta & \sin\theta & 0 \\
-\sin\theta & \cos\theta & 0 \\
0 & 0 & 1
\end{pmatrix}
$$
结果磁场向量乘以 $R(\theta)$ 并放大 $1\times10^9$ 后输出。

---

## 3. 代码使用示例

### 3.1 数据读取与预处理（ReadData.py）  
```python
from ReadData import ReadData
import numpy as np
import torch
from torch.utils.data import TensorDataset

# 加载原始数据
data = ReadData('/path/to/mat_files/')  # 返回 DataFrame，其中包含 Hx,Hy,Hz,x,y,z,L,W,Theta,m1x...m16z :contentReference[oaicite:0]{index=0}

# 构建 H_data: [Hx,Hy,Hz,z]
Hx, Hy, Hz = data['Hx'].values, data['Hy'].values, data['Hz'].values
z = data['z'].values
H_data = np.hstack([Hx.reshape(-1,1), Hy.reshape(-1,1), Hz.reshape(-1,1), z.reshape(-1,1)])
H_data = torch.tensor(H_data, dtype=torch.float64)

# 构建 ship_data: [x,y,L,W,Theta,m1x,...,m16z]
ship_np = data.iloc[:, 4:].values
ship_data = torch.tensor(ship_np, dtype=torch.float64)

# 数据集封装
dataset = TensorDataset(H_data, ship_data)
````

### 3.2 正问题计算（MagneticCalculate.py）

```python
from MagneticCalculate import forward_problem
# 调用 forward_problem 计算预测磁场: xn,yn,zn,m_inputs,theta,Ln,Wn 来自 ship_data 切片
H_pred = forward_problem(xn, yn, zn, m_inputs, theta=Theta, Ln=L, Wn=W)  # :contentReference[oaicite:1]{index=1}
```

### 3.3 模型定义与训练（net.py + train.py）

```python
import torch
from net import DNN
from train import Loss_pde, Loss_data
from torch.utils.data import DataLoader, random_split

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 网络结构定义
layers = [4, 64, 128, 512, 512, 256, 53]
model = DNN(layers)
model._initialize_weights()  # 权重初始化 :contentReference[oaicite:2]{index=2}
model.to(device)

# 超参数
epochs = 100
batch_size = 2048
weight = [1e-1, 100]  # [λ_pde, λ_data]
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

# 划分训练/测试集
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

# 训练循环（摘自 train.py） :contentReference[oaicite:3]{index=3}
for epoch in range(epochs):
    for H_batch, ship_batch in train_loader:
        H_batch, ship_batch = H_batch.to(device), ship_batch.to(device)
        optimizer.zero_grad()
        loss_pde = Loss_pde(model, H_batch, ship_batch)
        loss_data = Loss_data(model, H_batch, ship_batch)
        loss = weight[0]*loss_pde + weight[1]*loss_data
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch:03d} | total_loss={loss.item():.4e} | PDE={loss_pde.item():.4e} | data={loss_data.item():.4e}")
```

---

## 4. 模块说明

* **ReadData.py**：读取 MATLAB `.mat` 文件，输出 DataFrame，包括测点坐标、船体参数和磁矩矩阵&#x20;
* **MagneticCalculate.py**：实现 `forward_problem`，包含磁偶极子与椭球体模型的正问题计算&#x20;
* **net.py**：定义前馈全连接网络 `DNN`，支持自定义层数和激活&#x20;
* **train.py**：封装 PDE 损失和数据损失函数，提供训练和测试流程&#x20;
