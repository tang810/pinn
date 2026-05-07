import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict
import time
import warnings

class MultiDimHelmholtzPINN(nn.Module):
    """
    任意维度亥姆霍兹方程PINN求解器
    方程: ∇²u(x) + k²u(x) = 0, x ∈ ℝ^d
    """
    
    def __init__(self, input_dim, hidden_layers, neurons_per_layer, k=1.0, 
                 activation='tanh', use_residual=True):
        """
        初始化PINN模型
        
        参数:
            input_dim: 输入维度 (1, 2, 3)
            hidden_layers: 隐藏层数量
            neurons_per_layer: 每层神经元数量
            k: 波数
            activation: 激活函数 ('tanh', 'relu', 'sin')
            use_residual: 是否使用残差连接
        """
        super().__init__()
        self.input_dim = input_dim
        self.k = nn.Parameter(torch.tensor(float(k)))  # 波数可学习
        self.use_residual = use_residual
        
        # 构建网络层
        layers = OrderedDict()
        
        # 输入层
        layers['input'] = nn.Linear(input_dim, neurons_per_layer)
        layers['input_act'] = self._get_activation(activation)
        
        # 隐藏层
        for i in range(hidden_layers):
            if use_residual and i % 2 == 0 and i > 0:
                # 残差块
                layers[f'res_{i}_linear1'] = nn.Linear(neurons_per_layer, neurons_per_layer)
                layers[f'res_{i}_act1'] = self._get_activation(activation)
                layers[f'res_{i}_linear2'] = nn.Linear(neurons_per_layer, neurons_per_layer)
                layers[f'res_{i}_act2'] = self._get_activation(activation)
                layers[f'res_{i}_residual'] = nn.Identity()  # 残差连接
            else:
                layers[f'hidden_{i}'] = nn.Linear(neurons_per_layer, neurons_per_layer)
                layers[f'act_{i}'] = self._get_activation(activation)
        
        # 输出层
        layers['output'] = nn.Linear(neurons_per_layer, 1)
        
        self.net = nn.Sequential(layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _get_activation(self, activation):
        """获取激活函数"""
        if activation == 'tanh':
            return nn.Tanh()
        elif activation == 'relu':
            return nn.ReLU()
        elif activation == 'sin':
            return SinActivation()
        elif activation == 'siren':
            return SirenActivation()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
    
    def _initialize_weights(self):
        """Xavier初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """前向传播"""
        return self.net(x)
    
    def compute_pde_loss(self, x, requires_grad=True):
        """
        计算PDE残差损失
        
        参数:
            x: 输入坐标 [batch_size, input_dim]
            requires_grad: 是否需要计算梯度
            
        返回:
            pde_residual: PDE残差
            u: 网络输出
        """
        if requires_grad:
            x.requires_grad_(True)
        
        # 网络输出
        u = self.forward(x)
        
        # 计算梯度 (一阶)
        grad_u = torch.autograd.grad(
            u, x, 
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # 计算拉普拉斯算子 (∇²u)
        laplacian = torch.zeros_like(u)
        for i in range(self.input_dim):
            # 对每个维度计算二阶导数
            grad_u_i = grad_u[:, i:i+1]
            grad2_u_i = torch.autograd.grad(
                grad_u_i, x,
                grad_outputs=torch.ones_like(grad_u_i),
                create_graph=True,
                retain_graph=True
            )[0][:, i:i+1]
            laplacian += grad2_u_i
        
        # 亥姆霍兹方程残差: ∇²u + k²u
        pde_residual = laplacian + (self.k ** 2) * u
        
        return pde_residual, u
    
    def compute_boundary_loss(self, x_boundary, u_boundary):
        """计算边界损失"""
        u_pred = self.forward(x_boundary)
        return torch.mean((u_pred - u_boundary) ** 2)
    
    def compute_total_loss(self, x_domain, x_boundary=None, u_boundary=None, 
                          lambda_pde=1.0, lambda_bc=1.0):
        """
        计算总损失
        
        参数:
            x_domain: 计算域内的点
            x_boundary: 边界点
            u_boundary: 边界条件值
            lambda_pde: PDE损失权重
            lambda_bc: 边界损失权重
        """
        # PDE损失
        pde_residual, _ = self.compute_pde_loss(x_domain)
        loss_pde = torch.mean(pde_residual ** 2)
        
        # 边界损失
        loss_bc = torch.tensor(0.0)
        if x_boundary is not None and u_boundary is not None:
            loss_bc = self.compute_boundary_loss(x_boundary, u_boundary)
        
        # 总损失
        total_loss = lambda_pde * loss_pde + lambda_bc * loss_bc
        
        return total_loss, loss_pde, loss_bc

class SinActivation(nn.Module):
    """正弦激活函数 (SIREN风格)"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.sin(x)

class SirenActivation(nn.Module):
    """SIREN激活函数 (用于周期性问题的更好选择)"""
    def __init__(self, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
    
    def forward(self, x):
        return torch.sin(self.omega_0 * x)

class HelmholtzSolver:
    """亥姆霍兹方程求解器主类"""
    
    def __init__(self, config):
        """
        初始化求解器
        
        配置示例:
            config = {
                'input_dim': 2,           # 维度 (1, 2, 3)
                'domain': [[-1, 1], [-1, 1]],  # 每个维度的定义域
                'k': 5.0,                 # 波数
                'hidden_layers': 4,       # 隐藏层数
                'neurons_per_layer': 50,  # 每层神经元数
                'activation': 'tanh',     # 激活函数
                'learning_rate': 1e-3,
                'epochs': 10000,
                'batch_size': 1024,
                'use_residual': True,
            }
        """
        self.config = config
        self.input_dim = config['input_dim']
        
        # 创建模型
        self.model = MultiDimHelmholtzPINN(
            input_dim=config['input_dim'],
            hidden_layers=config.get('hidden_layers', 4),
            neurons_per_layer=config.get('neurons_per_layer', 50),
            k=config.get('k', 1.0),
            activation=config.get('activation', 'tanh'),
            use_residual=config.get('use_residual', True)
        )
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-3)
        )
        
        # 学习率调度器 - 修复版本兼容性问题
        try:
            # 尝试使用verbose参数（新版本）
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=500, verbose=True
            )
        except TypeError:
            # 如果不可用，则使用不带verbose参数的版本
            warnings.warn("ReduceLROnPlateau的verbose参数不可用，使用无verbose版本")
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=500
            )
        
        # 训练历史
        self.history = {
            'loss': [], 'loss_pde': [], 'loss_bc': [], 
            'k_values': [], 'learning_rates': []
        }
    
    def generate_training_data(self, n_domain=10000, n_boundary=1000):
        """生成训练数据"""
        domain = self.config['domain']
        
        # 域内点 (均匀采样或随机采样)
        if self.input_dim == 1:
            x_domain = torch.linspace(domain[0][0], domain[0][1], n_domain).reshape(-1, 1)
            x_boundary = torch.tensor([[domain[0][0]], [domain[0][1]]]).float()
            
        elif self.input_dim == 2:
            # 域内随机点
            x1 = torch.rand(n_domain, 1) * (domain[0][1] - domain[0][0]) + domain[0][0]
            x2 = torch.rand(n_domain, 1) * (domain[1][1] - domain[1][0]) + domain[1][0]
            x_domain = torch.cat([x1, x2], dim=1)
            
            # 边界点
            n_per_edge = n_boundary // 4
            edges = []
            
            # 四个边界采样
            t = torch.rand(n_per_edge) * (domain[0][1] - domain[0][0]) + domain[0][0]
            # 下边界
            edges.append(torch.stack([t, torch.full_like(t, domain[1][0])], dim=1))
            # 上边界
            edges.append(torch.stack([t, torch.full_like(t, domain[1][1])], dim=1))
            
            t = torch.rand(n_per_edge) * (domain[1][1] - domain[1][0]) + domain[1][0]
            # 左边界
            edges.append(torch.stack([torch.full_like(t, domain[0][0]), t], dim=1))
            # 右边界
            edges.append(torch.stack([torch.full_like(t, domain[0][1]), t], dim=1))
            
            x_boundary = torch.cat(edges, dim=0)
            
        elif self.input_dim == 3:
            # 3D情况类似，但需要6个面
            x1 = torch.rand(n_domain, 1) * (domain[0][1] - domain[0][0]) + domain[0][0]
            x2 = torch.rand(n_domain, 1) * (domain[1][1] - domain[1][0]) + domain[1][0]
            x3 = torch.rand(n_domain, 1) * (domain[2][1] - domain[2][0]) + domain[2][0]
            x_domain = torch.cat([x1, x2, x3], dim=1)
            
            # 3D边界生成（简化版）
            n_per_face = n_boundary // 6
            x_boundary = torch.rand(n_boundary, 3)
            # 设置每个面在某个维度上是边界值
            for i in range(6):
                start = i * n_per_face
                end = min((i + 1) * n_per_face, n_boundary)
                dim = i // 2
                value_idx = i % 2
                x_boundary[start:end, dim] = domain[dim][value_idx]
        
        # 边界条件 (示例: Dirichlet边界条件 u = 0)
        u_boundary = torch.zeros(x_boundary.shape[0], 1)
        
        return x_domain.float(), x_boundary.float(), u_boundary.float()
    
    def analytical_solution_1d(self, x, k):
        """1D解析解: u(x) = A*sin(k*x) + B*cos(k*x)"""
        # 假设边界条件 u(-1)=0, u(1)=1
        A = 1.0 / (2.0 * torch.sin(torch.tensor(k))) if torch.sin(torch.tensor(k)) != 0 else 1.0
        B = 0.5
        return A * torch.sin(k * x) + B * torch.cos(k * x)
    
    def analytical_solution_2d(self, x, y, k):
        """2D解析解: u(x,y) = sin(πx) * sin(πy) (对于特定边界条件)"""
        return torch.sin(np.pi * x) * torch.sin(np.pi * y)
    
    def train(self, x_domain, x_boundary, u_boundary, epochs=None):
        """训练模型"""
        if epochs is None:
            epochs = self.config.get('epochs', 10000)
        
        batch_size = self.config.get('batch_size', 1024)
        
        # 创建数据加载器
        dataset = TensorDataset(x_domain)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        print(f"开始训练 {self.input_dim}D 亥姆霍兹方程求解器...")
        print(f"波数 k = {self.model.k.item():.4f}")
        print(f"训练数据: {len(x_domain)} 个域内点, {len(x_boundary)} 个边界点")
        
        start_time = time.time()
        best_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_loss_pde = 0.0
            epoch_loss_bc = 0.0
            
            for batch_x in dataloader:
                batch_x = batch_x[0]
                
                # 前向传播和计算损失
                total_loss, loss_pde, loss_bc = self.model.compute_total_loss(
                    x_domain=batch_x,
                    x_boundary=x_boundary,
                    u_boundary=u_boundary,
                    lambda_pde=1.0,
                    lambda_bc=1.0
                )
                
                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                # 记录损失
                epoch_loss += total_loss.item()
                epoch_loss_pde += loss_pde.item()
                epoch_loss_bc += loss_bc.item() if x_boundary is not None else 0.0
            
            # 平均损失
            epoch_loss /= len(dataloader)
            epoch_loss_pde /= len(dataloader)
            epoch_loss_bc /= len(dataloader) if x_boundary is not None else 1.0
            
            # 学习率调整
            self.scheduler.step(epoch_loss)
            
            # 获取当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.history['loss'].append(epoch_loss)
            self.history['loss_pde'].append(epoch_loss_pde)
            self.history['loss_bc'].append(epoch_loss_bc)
            self.history['k_values'].append(self.model.k.item())
            self.history['learning_rates'].append(current_lr)
            
            # 保存最佳模型
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_state = self.model.state_dict().copy()
            
            # 打印进度
            if (epoch + 1) % 1000 == 0 or epoch == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1}/{epochs}, "
                      f"Loss: {epoch_loss:.2e}, "
                      f"PDE: {epoch_loss_pde:.2e}, "
                      f"BC: {epoch_loss_bc:.2e}, "
                      f"k: {self.model.k.item():.4f}, "
                      f"LR: {current_lr:.2e}, "
                      f"Time: {elapsed:.1f}s")
        
        # 加载最佳模型
        self.model.load_state_dict(best_model_state)
        
        total_time = time.time() - start_time
        print(f"\n训练完成! 总时间: {total_time:.1f}秒")
        print(f"最终波数估计: k = {self.model.k.item():.6f}")
        print(f"最佳损失: {best_loss:.2e}")
        
        return self.history
    
    def predict(self, x):
        """预测场值"""
        self.model.eval()
        with torch.no_grad():
            return self.model(x).numpy()
    
    def evaluate(self, test_points, analytical_solution=None):
        """评估模型精度"""
        self.model.eval()
        
        with torch.no_grad():
            predictions = self.model(test_points).numpy()
        
        if analytical_solution is not None:
            true_values = analytical_solution(test_points).numpy()
            mse = np.mean((predictions - true_values) ** 2)
            rmse = np.sqrt(mse)
            max_error = np.max(np.abs(predictions - true_values))
            relative_error = np.mean(np.abs(predictions - true_values) / (np.abs(true_values) + 1e-8))
            
            print(f"评估结果:")
            print(f"  MSE: {mse:.2e}")
            print(f"  RMSE: {rmse:.2e}")
            print(f"  最大绝对误差: {max_error:.2e}")
            print(f"  相对误差: {relative_error:.2%}")
            
            return predictions, true_values, {
                'mse': mse, 'rmse': rmse, 'max_error': max_error, 
                'relative_error': relative_error
            }
        
        return predictions
    
    def plot_results_1d(self, x_test, u_true=None):
        """绘制1D结果"""
        u_pred = self.predict(x_test)
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(x_test.numpy(), u_pred, 'b-', linewidth=2, label='PINN预测')
        
        if u_true is not None:
            plt.plot(x_test.numpy(), u_true.numpy(), 'r--', linewidth=2, 
                    label='解析解', alpha=0.7)
        
        plt.xlabel('x', fontsize=14)
        plt.ylabel('u(x)', fontsize=14)
        plt.title(f'1D亥姆霍兹方程求解结果 (k={self.model.k.item():.3f})', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 误差图
        if u_true is not None:
            plt.subplot(1, 2, 2)
            error = np.abs(u_pred - u_true.numpy())
            plt.plot(x_test.numpy(), error, 'g-', linewidth=2)
            plt.fill_between(x_test.numpy().flatten(), 0, error.flatten(), alpha=0.3, color='green')
            plt.xlabel('x', fontsize=14)
            plt.ylabel('绝对误差', fontsize=14)
            plt.title('预测误差', fontsize=16)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_results_2d(self, grid_x, grid_y):
        """绘制2D结果"""
        # 创建网格
        xx, yy = np.meshgrid(grid_x, grid_y)
        points = np.stack([xx.flatten(), yy.flatten()], axis=1)
        
        # 预测
        with torch.no_grad():
            u_pred = self.model(torch.tensor(points).float()).numpy()
            u_pred_grid = u_pred.reshape(xx.shape)
        
        # 绘图
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 等高线图
        contour = axes[0].contourf(xx, yy, u_pred_grid, levels=50, cmap='RdBu_r')
        axes[0].set_xlabel('x', fontsize=12)
        axes[0].set_ylabel('y', fontsize=12)
        axes[0].set_title(f'2D亥姆霍兹方程解 (k={self.model.k.item():.3f})', fontsize=14)
        plt.colorbar(contour, ax=axes[0])
        
        # 3D表面图
        ax3d = fig.add_subplot(132, projection='3d')
        surf = ax3d.plot_surface(xx, yy, u_pred_grid, cmap='RdBu_r', 
                                antialiased=True, alpha=0.8)
        ax3d.set_xlabel('x', fontsize=12)
        ax3d.set_ylabel('y', fontsize=12)
        ax3d.set_zlabel('u(x,y)', fontsize=12)
        ax3d.set_title('三维表面图', fontsize=14)
        
        # 热力图
        im = axes[2].imshow(u_pred_grid, extent=[grid_x.min(), grid_x.max(), 
                                                grid_y.min(), grid_y.max()], 
                           origin='lower', aspect='auto', cmap='RdBu_r')
        axes[2].set_xlabel('x', fontsize=12)
        axes[2].set_ylabel('y', fontsize=12)
        axes[2].set_title('热力图', fontsize=14)
        plt.colorbar(im, ax=axes[2])
        
        plt.tight_layout()
        plt.show()
    
    def plot_training_history(self):
        """绘制训练历史"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 损失曲线
        epochs = range(1, len(self.history['loss']) + 1)
        
        axes[0, 0].semilogy(epochs, self.history['loss'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('总损失', fontsize=12)
        axes[0, 0].set_title('训练损失', fontsize=14)
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].semilogy(epochs, self.history['loss_pde'], 'r-', linewidth=2, label='PDE损失')
        if any(l > 0 for l in self.history['loss_bc']):
            axes[0, 1].semilogy(epochs, self.history['loss_bc'], 'g-', linewidth=2, label='边界损失')
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('损失', fontsize=12)
        axes[0, 1].set_title('损失分量', fontsize=14)
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(epochs, self.history['k_values'], 'purple', linewidth=2)
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('波数 k', fontsize=12)
        axes[1, 0].set_title('波数估计值', fontsize=14)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 学习率变化
        axes[1, 1].semilogy(epochs, self.history['learning_rates'], 'orange', linewidth=2)
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('学习率', fontsize=12)
        axes[1, 1].set_title('学习率变化', fontsize=14)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def example_1d_simple():
    """1D亥姆霍兹方程简单示例"""
    print("=" * 60)
    print("1D亥姆霍兹方程求解示例")
    print("=" * 60)
    
    # 配置
    config = {
        'input_dim': 1,
        'domain': [[-1, 1]],  # 定义域 [-1, 1]
        'k': 3.0,              # 波数
        'hidden_layers': 3,
        'neurons_per_layer': 30,
        'activation': 'tanh',
        'learning_rate': 5e-3,  # 提高学习率
        'epochs': 3000,        # 减少训练轮数
        'batch_size': 256,
        'use_residual': False,  # 1D问题简单，不需要残差
    }
    
    # 创建求解器
    solver = HelmholtzSolver(config)
    
    # 生成训练数据
    x_domain, x_boundary, u_boundary = solver.generate_training_data(
        n_domain=1000, n_boundary=2
    )
    
    # 设置边界条件 (示例: u(-1)=0, u(1)=1)
    u_boundary[0] = 0.0  # x = -1
    u_boundary[1] = 1.0  # x = 1
    
    # 训练
    history = solver.train(x_domain, x_boundary, u_boundary, epochs=config['epochs'])
    
    # 评估
    x_test = torch.linspace(-1, 1, 200).reshape(-1, 1)
    
    # 创建解析解函数
    def analytical_solution_1d(x, k=config['k']):
        # 简单解析解: u(x) = sin(kx) * sin(k) 满足 u(±1)=0
        return torch.sin(k * x) / torch.sin(torch.tensor(k))
    
    u_true = analytical_solution_1d(x_test, k=solver.model.k.item())
    
    # 预测和评估
    predictions, true_values, metrics = solver.evaluate(
        x_test, 
        lambda x: analytical_solution_1d(x, k=solver.model.k.item())
    )
    
    # 绘制结果
    solver.plot_results_1d(x_test, u_true)
    solver.plot_training_history()
    
    return solver, metrics

def example_2d_simple():
    """2D亥姆霍兹方程简单示例"""
    print("\n" + "=" * 60)
    print("2D亥姆霍兹方程求解示例")
    print("=" * 60)
    
    # 配置
    config = {
        'input_dim': 2,
        'domain': [[-1, 1], [-1, 1]],  # 正方形区域
        'k': 2.0,                      # 较小波数
        'hidden_layers': 4,
        'neurons_per_layer': 50,
        'activation': 'tanh',
        'learning_rate': 1e-3,
        'epochs': 5000,                # 中等训练轮数
        'batch_size': 1024,
        'use_residual': True,
    }
    
    # 创建求解器
    solver = HelmholtzSolver(config)
    
    # 生成训练数据
    x_domain, x_boundary, u_boundary = solver.generate_training_data(
        n_domain=5000, n_boundary=200
    )
    
    # 设置边界条件 (在边界上u=0)
    # u_boundary已经是0，不需要修改
    
    # 训练
    history = solver.train(x_domain, x_boundary, u_boundary, epochs=config['epochs'])
    
    # 创建测试网格
    n_test = 40
    x_test = torch.linspace(-1, 1, n_test)
    y_test = torch.linspace(-1, 1, n_test)
    
    # 绘制结果
    solver.plot_results_2d(x_test.numpy(), y_test.numpy())
    solver.plot_training_history()
    
    return solver

def test_pytorch_version():
    """检查PyTorch版本并给出建议"""
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    
    # 检查ReduceLROnPlateau参数
    import inspect
    sig = inspect.signature(torch.optim.lr_scheduler.ReduceLROnPlateau.__init__)
    params = list(sig.parameters.keys())
    print(f"\nReduceLROnPlateau支持的参数: {params}")
    
    if 'verbose' in params:
        print("✓ 当前PyTorch版本支持verbose参数")
    else:
        print("⚠ 当前PyTorch版本不支持verbose参数")

# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    # 检查PyTorch版本
    test_pytorch_version()
    
    print("\n" + "=" * 60)
    print("开始运行亥姆霍兹方程PINN求解器")
    print("=" * 60)
    
    try:
        # 运行1D示例
        solver_1d, metrics_1d = example_1d_simple()
        
        print("\n" + "=" * 60)
        print("1D示例完成!")
        print("=" * 60)
        
        # 询问是否运行2D示例
        user_input = input("\n是否运行2D示例? (y/n): ").strip().lower()
        if user_input == 'y':
            solver_2d = example_2d_simple()
            print("\n" + "=" * 60)
            print("2D示例完成!")
            print("=" * 60)
        else:
            print("跳过2D示例")
            
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n尝试简化版本...")
        
        # 尝试更简单的1D示例
        print("\n运行简化1D示例...")
        
        # 创建简单模型
        model = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # 简单训练循环
        x = torch.linspace(-1, 1, 100).reshape(-1, 1)
        for epoch in range(1000):
            u = model(x)
            loss = torch.mean(u**2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 200 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        print("简化示例完成!")
    
    print("\n" + "=" * 60)
    print("代码特点总结:")
    print("1. 支持任意维度 (1D/2D/3D)")
    print("2. 自动计算拉普拉斯算子")
    print("3. 可学习波数参数")
    print("4. 残差连接选项")
    print("5. 多种激活函数支持")
    print("6. 完整的训练监控和可视化")
    print("7. 修复了PyTorch版本兼容性问题")
    print("=" * 60)