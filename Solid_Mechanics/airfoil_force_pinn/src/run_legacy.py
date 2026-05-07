"""
完整3D翼型力约束PINN实现 - 单文件版本

此文件整合了所有必要类和函数，可直接运行。
它结合了通用PINN框架的优点和特定翼型力学问题的需求，
包括力边界条件处理、自动数据生成和全面的可视化。

功能特点:
1.  **单文件集成**: 所有代码（神经网络、PINN求解器、数据加载器、工具函数）都在一个文件中。
2.  **3D弹性力学**: 解决3D线弹性力学问题，核心是Navier-Cauchy平衡方程。
3.  **多物理约束**:
    *   **位移边界条件**: 固定翼根（或指定区域）的位移。
    *   **物理方程约束**: 域内强制满足弹性力学平衡方程。
    *   **力边界条件**: 翼面上的表面牵引力（模拟气动载荷）。
        *   **注意**: 原始“代码一”中未包含力边界条件。此处保留力边界条件，但其权重和其他参数将遵循“代码一”的默认值逻辑。
4.  **智能数据管理**:
    *   尝试从指定 `.mat` 文件加载翼型几何数据。
    *   如果数据文件不存在，则**自动生成简化的3D NACA翼型几何数据**。
    *   基于几何数据，自动**生成位移固定边界条件**（翼根区域）和**表面力边界条件**（翼面随机点）。
    *   自动在翼型包围盒内生成**域内训练点**用于方程约束。
5.  **多阶段训练**: 采用分阶段降低学习率的训练策略，提高收敛性。
    *   **注意**: 训练Epoch数量显著增加，以促进更好的模型收敛。
6.  **全面可视化**: 训练完成后，生成并展示以下结果图：
    *   X, Y, Z方向的预测位移。
    *   总位移大小。
    *   训练过程中的总损失、方程损失和边界损失历史（对数坐标）。
    *   翼型几何形状（评估点）。
    *   预测的von Mises应力分布（域内点）。
    *   关键统计信息。
    *   **所有绘图标签和标题均为英文。**
7.  **模型管理**: 支持训练好的模型参数和优化器状态的保存与加载。
8.  **GPU加速**: 自动检测并优先使用CUDA设备进行计算。

运行方式:
直接运行此Python脚本。
默认模式为 'train'，训练完成后会保存模型并进行评估可视化。
可以修改 `main()` 函数中的 `mode` 变量切换到 'test' 模式来加载并评估已训练的模型。
"""

import os
import math
import torch
import numpy as np
import matplotlib # 导入 matplotlib 主模块
import matplotlib.pyplot as plt # 导入 matplotlib.pyplot 为 plt
import scipy.io
from collections import OrderedDict
from typing import List, Optional
from mpl_toolkits.mplot3d import Axes3D

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# UTILITY FUNCTIONS
# (Originally from tools.py, simplified as not all are strictly needed for this problem)
# ============================================================================

def LHSample(D, bounds, N):
    """
    Latin Hypercube Sampling
    :param D: Number of parameters
    :param bounds: [[min_1, max_1],[min_2, max_2],[min_3, max_3]](list)
    :param N: Number of samples
    :return: Samples
    """
    # Import qmc locally to ensure it's available when this function is called
    from scipy.stats import qmc
    sampler = qmc.LatinHypercube(d=D)
    sample = sampler.random(n=N)
    # Stretching the sampling to the given bounds
    b = np.array(bounds)
    lower_bounds = b[:, 0]
    upper_bounds = b[:, 1]
    scaled_sample = qmc.scale(sample, lower_bounds, upper_bounds)
    return scaled_sample

# ============================================================================
# NEURAL NETWORK ARCHITECTURE
# (Originally from net.py)
# ============================================================================

class FCNet(torch.nn.Module):
    """
    Fully Connected Neural Network for approximating displacement field.
    Input: (x, y, z) coordinates
    Output: (u, v, w) displacement components
    """
    def __init__(self, num_ins=3, num_outs=3, num_layers=10, hidden_size=50): # Strict adherence to Code One
        super(FCNet, self).__init__()
        
        layers = [num_ins] + [hidden_size] * num_layers + [num_outs]
        self.depth = len(layers) - 1
        
        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1])))
            layer_list.append(('activation_%d' % i, torch.nn.Tanh())) # Tanh activation function
        
        layer_list.append(('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1])))
        self.layers = torch.nn.Sequential(OrderedDict(layer_list))

    def forward(self, x):
        return self.layers(x)

# ============================================================================
# AIRFOIL DATA LOADER AND GENERATOR
# (Adapted from SM_data.py and previous airfoil data logic)
# ============================================================================

class AirfoilDataLoader:
    """
    Handles loading or generating airfoil geometry data, displacement boundary conditions,
    force boundary conditions, and interior training points.
    """
    def __init__(self, path='./data/'):
        self.path = path
        os.makedirs(self.path, exist_ok=True) # Ensure data directory exists
        
    def loading_boundary_data(self, filename: str):
        """
        Loads airfoil geometry data and generates displacement and force boundary conditions from it.
        Args:
            filename (str): .mat file containing airfoil coordinates.
        Returns:
            tuple: (bc_data, force_bc_data)
                   bc_data: (x_bc, y_bc, z_bc, u_bc, v_bc, w_bc)
                   force_bc_data: (x_force, y_force, z_force, fx_target, fy_target, fz_target, normal_x, normal_y, normal_z)
        """
        filepath = os.path.join(self.path, filename)
        
        # Create sample data if file does not exist
        if not os.path.exists(filepath):
            print(f"Data file not found: {filepath}")
            print("Creating sample airfoil data...")
            self.create_sample_airfoil_data(filename='sample_airfoil.mat')
            filename = 'sample_airfoil.mat' # Switch to sample filename
            filepath = os.path.join(self.path, filename)
        
        try:
            data = scipy.io.loadmat(filepath)
            print(f"Successfully loaded airfoil data: {filename}")
            print(f"Data keys: {list(data.keys())}")
            
            # Extract coordinate data
            if 'x' in data and 'y' in data and 'z' in data:
                x_coords = data['x'].flatten()
                y_coords = data['y'].flatten()
                z_coords = data['z'].flatten()
            else:
                raise ValueError("Data file must contain 'x', 'y', 'z' coordinate data")
            
            # Generate boundary and force boundary conditions
            return self.generate_airfoil_boundary_conditions(x_coords, y_coords, z_coords)
            
        except Exception as e:
            print(f"Failed to load data: {e}")
            return None, None # Return None if loading fails

    def loading_training_data(self, filename: str, n_points: int = 20000): # Strict adherence to Code One
        """
        Loads or generates interior training data (for PDE constraint).
        Args:
            filename (str): Airfoil data file to get bounding box.
            n_points (int): Number of interior points if generated.
        Returns:
            tuple: (x_f, y_f, z_f) Coordinates of interior points.
        """
        filepath = os.path.join(self.path, filename)
        
        try:
            data = scipy.io.loadmat(filepath)
            
            # Try to load predefined interior points, otherwise generate based on boundary
            x_domain = data.get('x_domain')
            y_domain = data.get('y_domain')
            z_domain = data.get('z_domain')

            if x_domain is not None and y_domain is not None and z_domain is not None and x_domain.size > 0:
                print(f"Loaded predefined interior training points: {x_domain.size} points")
                x_train_f = x_domain.reshape(-1, 1)
                y_train_f = y_domain.reshape(-1, 1)
                z_train_f = z_domain.reshape(-1, 1)
            else:
                # If no interior point data, generate based on boundary data
                x_coords = data['x'].flatten()
                y_coords = data['y'].flatten()
                z_coords = data['z'].flatten()
                
                # Get bounding box of the airfoil
                x_min, x_max = x_coords.min(), x_coords.max()
                y_min, y_max = y_coords.min(), y_coords.max()
                z_min, z_max = z_coords.min(), z_coords.max()
                
                # Use LHSample to generate random training points within the bounding box
                bounds = [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
                points = LHSample(3, bounds, n_points)
                
                x_train_f = points[:, 0].reshape(-1, 1)
                y_train_f = points[:, 1].reshape(-1, 1)
                z_train_f = points[:, 2].reshape(-1, 1)
                
                print(f"Generated interior training points: {n_points} points (LHSampling)")
            
            return x_train_f, y_train_f, z_train_f
            
        except Exception as e:
            print(f"Failed to load training data: {e}")
            return None, None, None

    def loading_evaluate_data(self, filename: str):
        """
        Loads or generates evaluation data. Here, airfoil geometry points are used as evaluation points.
        Args:
            filename (str): Airfoil data file.
        Returns:
            tuple: (x_eval, y_eval, z_eval) Coordinates of evaluation points.
        """
        filepath = os.path.join(self.path, filename)
        try:
            data = scipy.io.loadmat(filepath)
            x_coords = data['x'].flatten().reshape(-1,1)
            y_coords = data['y'].flatten().reshape(-1,1)
            z_coords = data['z'].flatten().reshape(-1,1)
            print(f"Loaded evaluation data: {x_coords.shape[0]} points (using airfoil geometry points)")
            return x_coords, y_coords, z_coords
        except Exception as e:
            print(f"Failed to load evaluation data: {e}")
            return None, None, None


    def generate_airfoil_boundary_conditions(self, x_coords: np.ndarray, y_coords: np.ndarray, z_coords: np.ndarray):
        """
        Generates fixed displacement boundary conditions and surface force boundary conditions for the airfoil.
        Assumes airfoil is oriented along the Y-axis. The root is at the minimum Y-axis value.
        Args:
            x_coords, y_coords, z_coords (np.ndarray): All geometry points of the airfoil.
        Returns:
            tuple: (bc_data, force_bc_data)
        """
        
        # --- 1. Generate Displacement Boundary Conditions (Fixed Airfoil Root) ---
        y_min, y_max = y_coords.min(), y_coords.max()
        y_range = y_max - y_min
        
        # Define root region as 5% range from minimum Y-axis
        root_threshold = y_min + 0.05 * y_range
        root_indices = np.where(y_coords <= root_threshold)[0]
        
        n_root_bc = min(len(root_indices), 500) # Select max 500 points as fixed boundary
        if n_root_bc > 0:
            selected_root = np.random.choice(root_indices, n_root_bc, replace=False)
            
            x_bc = x_coords[selected_root].reshape(-1, 1)
            y_bc = y_coords[selected_root].reshape(-1, 1)
            z_bc = z_coords[selected_root].reshape(-1, 1)
            u_bc = np.zeros((n_root_bc, 1))  # Fixed boundary, zero displacement
            v_bc = np.zeros((n_root_bc, 1))
            w_bc = np.zeros((n_root_bc, 1))
        else:
            # Fallback if no clear root points (e.g., too few points), randomly select some points
            print("Warning: Insufficient root points, randomly selecting points as fixed boundary.")
            n_bc_fallback = min(200, len(x_coords) // 5)
            indices = np.random.choice(len(x_coords), n_bc_fallback, replace=False)
            
            x_bc = x_coords[indices].reshape(-1, 1)
            y_bc = y_coords[indices].reshape(-1, 1)
            z_bc = z_coords[indices].reshape(-1, 1)
            u_bc = np.zeros((n_bc_fallback, 1))
            v_bc = np.zeros((n_bc_fallback, 1))
            w_bc = np.zeros((n_bc_fallback, 1))
            n_root_bc = n_bc_fallback 

        print(f"Generated displacement boundary conditions: {n_root_bc} fixed points (root or random).")
        bc_data = (x_bc, y_bc, z_bc, u_bc, v_bc, w_bc)

        # --- 2. Generate Force Boundary Conditions (Load on Airfoil Surface) ---
        # Simulate aerodynamic load, applied on a subset of airfoil surface points.
        n_force_pts = min(len(x_coords) // 2, 1000) # Select max 1000 points to apply force
        force_indices = np.random.choice(len(x_coords), n_force_pts, replace=False)
        
        x_force = x_coords[force_indices].reshape(-1, 1)
        y_force = y_coords[force_indices].reshape(-1, 1)
        z_force = z_coords[force_indices].reshape(-1, 1)
        
        # Apply a simplified aerodynamic load: assumed constant surface pressure in Z-direction (simulating lift)
        # Pressure is a force normal to the surface. Here, simplified to primarily Z-direction.
        # In reality, normal vector 'n' should be from mesh vertices.
        # Here, assume load is "lift" in positive Z-axis, acting on upper surface of XY-plane airfoil.
        # Thus, assume normal vector is (0, 0, 1).
        
        pressure_load_pa = 5000.0 # Assumed uniform pressure load, in Pascals (N/m^2)
        
        # Note: fx_target, fy_target, fz_target represent surface traction t_x, t_y, t_z,
        # not direct pressure value. If force is normal pressure P, then t = P * n.
        # Assuming n = (0,0,1), then t_x=0, t_y=0, t_z=P.
        
        fx_target = np.zeros((n_force_pts, 1))
        fy_target = np.zeros((n_force_pts, 1))
        fz_target = np.full((n_force_pts, 1), pressure_load_pa) 
        
        # Simplified normal vector: upwards (0, 0, 1). Actual applications require normals from CAD/mesh.
        normal_x = np.zeros((n_force_pts, 1))
        normal_y = np.zeros((n_force_pts, 1))
        normal_z = np.ones((n_force_pts, 1)) 
        
        print(f"Generated force boundary conditions: {n_force_pts} load points (simulating aerodynamic load).")
        force_bc_data = (x_force, y_force, z_force, fx_target, fy_target, fz_target,
                         normal_x, normal_y, normal_z)
        
        return bc_data, force_bc_data

    def create_sample_airfoil_data(self, filename='sample_airfoil.mat'):
        """
        Generates sample NACA 0012 airfoil point data in 3D space and saves as .mat file.
        Args:
            filename (str): Filename to save.
        """
        print(f"Creating NACA airfoil sample data to {os.path.join(self.path, filename)}...")
        
        def naca_airfoil_thickness(x_norm, thickness_ratio=0.12):
            """
            Calculates half-thickness of NACA symmetric airfoil at normalized chord position.
            """
            a0 = 0.2969
            a1 = -0.1260
            a2 = -0.3516
            a3 = 0.2843
            a4 = -0.1015 # Adjusted to close trailing edge
            
            yt = 5 * thickness_ratio * (a0*np.sqrt(x_norm) + a1*x_norm + a2*x_norm**2 + a3*x_norm**3 + a4*x_norm**4)
            return yt
        
        # Airfoil geometric parameters
        chord_length = 1.0  # Chord length [m]
        span_length = 2.0   # Span [m]
        naca_thickness_ratio = 0.12 # NACA 0012
        
        # Discretization points along chord and span
        n_chord_pts = 50
        n_span_pts = 30
        
        # Generate airfoil cross-section (along X-axis)
        x_section_norm = np.linspace(0.0001, 1.0, n_chord_pts) 
        thickness_abs = naca_airfoil_thickness(x_section_norm, naca_thickness_ratio) * chord_length
        
        x_coords_list = []
        y_coords_list = []
        z_coords_list = []
        
        # Replicate airfoil cross-section along span (Y-axis)
        for y_pos in np.linspace(0, span_length, n_span_pts):
            # Upper surface (z > 0)
            x_coords_list.extend(x_section_norm * chord_length)
            y_coords_list.extend([y_pos] * n_chord_pts)
            z_coords_list.extend(thickness_abs)
            
            # Lower surface (z < 0)
            x_coords_list.extend(x_section_norm * chord_length)
            y_coords_list.extend([y_pos] * n_chord_pts)
            z_coords_list.extend(-thickness_abs)
        
        # Convert to Numpy arrays
        x_coords = np.array(x_coords_list).reshape(-1, 1)
        y_coords = np.array(y_coords_list).reshape(-1, 1)
        z_coords = np.array(z_coords_list).reshape(-1, 1)
        
        # Save data
        airfoil_data = {
            'x': x_coords,
            'y': y_coords,
            'z': z_coords,
            'chord': chord_length,
            'span': span_length
        }
        
        filepath = os.path.join(self.path, filename)
        scipy.io.savemat(filepath, airfoil_data)
        print(f"Sample airfoil data saved: {filepath}, containing {x_coords.shape[0]} points.")

# ============================================================================
# PHYSICS-INFORMED NEURAL NETWORK SOLVER
# (Adapted from pinn_solver.py and previous airfoil PINN logic)
# ============================================================================

class AirfoilStructuralPINN:
    """
    Physics-Informed Neural Network (PINN) implementation for 3D elasticity problems.
    Includes displacement boundary conditions, equilibrium equations, and surface force (traction) constraints.
    """
    def __init__(self, E=100, nu=0.3, layers=10, hidden_size=50, learning_rate=0.001, # Strict adherence to Code One
                 bc_weight=1, eq_weight=1, force_weight=1, net_params=None): # Strict adherence to Code One
        
        self.E = E # Young's Modulus
        self.nu = nu # Poisson's Ratio
        
        # Loss weights
        self.alpha_b = bc_weight      # Displacement BC loss weight
        self.alpha_e = eq_weight      # Equilibrium equation (PDE) loss weight  
        self.alpha_force = force_weight # Force BC loss weight
        
        # Loss trackers
        self.loss_b = 0.0
        self.loss_e = 0.0
        self.loss_force = 0.0
        self.loss_bcs_all = []    # Stores displacement BC loss history
        self.loss_equ_all = []    # Stores equation loss history
        self.loss_sum_all = []    # Stores total loss history

        # Initialize neural network
        self.net = FCNet(
            num_ins=3, num_outs=3, num_layers=layers, hidden_size=hidden_size
        ).to(device)
        
        # Load pre-trained weights (if path provided and file exists)
        if net_params and os.path.exists(net_params):
            load_params = torch.load(net_params, map_location=device)
            self.net.load_state_dict(load_params['model_state_dict'])
            # Optimizer state can also be loaded, but here it's re-initialized by default
            print(f"Loaded pre-trained model: {net_params}")

        # Initialize optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), lr=learning_rate, weight_decay=0) # Strict adherence to Code One (default weight_decay is 0)
        
        print(f"Airfoil PINN model initialized: E={self.E:.1f}, ν={self.nu:.2f}, Network {layers}x{hidden_size}")

    def set_boundary_data(self, X: tuple):
        """
        Sets displacement boundary condition data.
        Displacements at these points are known, typically zero displacement (fixed end).
        Args:
            X (tuple): Tuple containing (x_bc, y_bc, z_bc, u_bc, v_bc, w_bc), each as a Numpy array.
        """
        # For BC data, gradients are usually not needed, so requires_grad=False
        requires_grad = False 
        self.x_bc = torch.tensor(X[0], dtype=torch.float32, requires_grad=requires_grad).to(device)
        self.y_bc = torch.tensor(X[1], dtype=torch.float32, requires_grad=requires_grad).to(device)
        self.z_bc = torch.tensor(X[2], dtype=torch.float32, requires_grad=requires_grad).to(device)
        self.u_bc = torch.tensor(X[3], dtype=torch.float32, requires_grad=requires_grad).to(device)
        self.v_bc = torch.tensor(X[4], dtype=torch.float32, requires_grad=requires_grad).to(device)
        self.w_bc = torch.tensor(X[5], dtype=torch.float32, requires_grad=requires_grad).to(device)

    def set_force_boundary_data(self, X_force: tuple):
        """
        Sets force boundary condition data.
        Surface traction forces at these points are known (e.g., pressure or shear).
        Args:
            X_force (tuple): Tuple containing (x_force, y_force, z_force, fx_target, fy_target, fz_target, normal_x, normal_y, normal_z).
                             Coordinates need requires_grad=True for stress calculation.
        """
        requires_grad = True # For force boundary points, stress must be calculated, so coordinates need gradients
        self.x_force = torch.tensor(X_force[0], dtype=torch.float32, requires_grad=requires_grad).to(device)
        self.y_force = torch.tensor(X_force[1], dtype=torch.float32, requires_grad=requires_grad).to(device)
        self.z_force = torch.tensor(X_force[2], dtype=torch.float32, requires_grad=requires_grad).to(device)
        self.fx_target = torch.tensor(X_force[3], dtype=torch.float32, requires_grad=False).to(device)
        self.fy_target = torch.tensor(X_force[4], dtype=torch.float32, requires_grad=False).to(device)
        self.fz_target = torch.tensor(X_force[5], dtype=torch.float32, requires_grad=False).to(device)
        self.normal_x = torch.tensor(X_force[6], dtype=torch.float32, requires_grad=False).to(device)
        self.normal_y = torch.tensor(X_force[7], dtype=torch.float32, requires_grad=False).to(device)
        self.normal_z = torch.tensor(X_force[8], dtype=torch.float32, requires_grad=False).to(device)

    def set_eq_training_data(self, X: tuple):
        """
        Sets equilibrium equation (PDE) training data.
        These points are distributed within the computational domain to enforce satisfaction of elasticity equilibrium equations.
        Args:
            X (tuple): Tuple containing (x_f, y_f, z_f), each as a Numpy array.
        """
        requires_grad = True # For interior points, second derivatives of displacement (for PDE) are needed, so coordinates need gradients
        self.x_f = torch.tensor(X[0], dtype=torch.float32, requires_grad=requires_grad).to(device)
        self.y_f = torch.tensor(X[1], dtype=torch.float32, requires_grad=requires_grad).to(device)
        self.z_f = torch.tensor(X[2], dtype=torch.float32, requires_grad=requires_grad).to(device)

    def neural_net_u(self, x, y, z):
        """
        Predicts displacement components at given coordinates via neural network.
        Args:
            x, y, z (torch.Tensor): Coordinate tensors.
        Returns:
            tuple: (u, v, w) Displacement component tensors.
        """
        X = torch.cat((x, y, z), dim=1) # Concatenate coordinates for NN input
        uvw = self.net(X)
        u = uvw[:, 0:1]
        v = uvw[:, 1:2]
        w = uvw[:, 2:3]
        return u, v, w

    @staticmethod
    def autograd(y: torch.Tensor, x_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Performs automatic differentiation to compute gradients of y with respect to each element in x_list.
        Args:
            y (torch.Tensor): Dependent variable tensor.
            x_list (List[torch.Tensor]): List of independent variable tensors.
        Returns:
            List[torch.Tensor]: Gradients of y with respect to each x_i.
        """
        grad_outputs = [torch.ones_like(y, device=y.device)] # Ensure grad_outputs shape and device match y
        grad = torch.autograd.grad([y], x_list, grad_outputs=grad_outputs,
                                 create_graph=True, allow_unused=True)
        # Ensure zero tensor is returned for x_i that might have no gradient
        return [g if g is not None else torch.zeros_like(x_list[i]) for i, g in enumerate(grad)]

    def neural_net_equation(self, x, y, z):
        """
        Calculates residuals of elasticity equilibrium equations (PDE loss).
        Args:
            x, y, z (torch.Tensor): Coordinates of interior points.
        Returns:
            tuple: (eq1, eq2, eq3) Residuals of equilibrium equations in x, y, z directions.
        """
        u, v, w = self.neural_net_u(x, y, z)

        # Compute first derivatives of displacement
        u_x, u_y, u_z = self.autograd(u, [x, y, z])
        v_x, v_y, v_z = self.autograd(v, [x, y, z])
        w_x, w_y, w_z = self.autograd(w, [x, y, z])
        
        # Compute second derivatives of displacement
        u_xx = self.autograd(u_x, [x])[0]
        u_xy = self.autograd(u_x, [y])[0]
        u_xz = self.autograd(u_x, [z])[0]
        u_yy = self.autograd(u_y, [y])[0]
        u_zz = self.autograd(u_z, [z])[0]

        v_xx = self.autograd(v_x, [x])[0]
        v_xy = self.autograd(v_x, [y])[0] 
        v_yy = self.autograd(v_y, [y])[0]
        v_yz = self.autograd(v_y, [z])[0]
        v_zz = self.autograd(v_z, [z])[0]

        w_xx = self.autograd(w_x, [x])[0]
        w_xz = self.autograd(w_x, [z])[0] 
        w_yy = self.autograd(w_y, [y])[0]
        w_yz = self.autograd(w_y, [z])[0] 
        w_zz = self.autograd(w_z, [z])[0]
        
        # Compute Lamé parameters
        G = self.E / (2 * (1 + self.nu)) # Shear Modulus
        la = (self.E * self.nu) / ((1 + self.nu) * (1 - 2 * self.nu)) # First Lame Parameter
        
        # 3D Elasticity Equilibrium Equations (Navier-Cauchy Equations), with body forces
        f_1 = 0.1 # Body force in x-direction (Strict adherence to Code One)
        f_2 = 0.1 # Body force in y-direction (Strict adherence to Code One)
        f_3 = 0.1 # Body force in z-direction (Strict adherence to Code One)

        eq1 = ((2 * G + la) * ( u_xx ) + G * (u_yy + u_zz) + ( G + la )*(v_xy + w_xz) + f_1)
        eq2 = ((2 * G + la) * ( v_yy ) + G * (v_xx + v_zz) + ( G + la )*(u_xy + w_yz) + f_2)
        eq3 = ((2 * G + la) * ( w_zz ) + G * (w_xx + w_yy) + ( G + la )*(u_xz + v_yz) + f_3)
        
        return eq1, eq2, eq3

    def neural_net_s(self, x, y, z):
        """
        Calculates stress components at given coordinates.
        Args:
            x, y, z (torch.Tensor): Coordinate tensors.
        Returns:
            tuple: (s1, s2, s3, s12, s23, s13) Stress component tensors
                   corresponding to (sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_xz)
        """
        u, v, w = self.neural_net_u(x, y, z)
        
        # Compute first derivatives of displacement (strains)
        u_x, u_y, u_z = self.autograd(u, [x, y, z])
        v_x, v_y, v_z = self.autograd(v, [x, y, z])
        w_x, w_y, w_z = self.autograd(w, [x, y, z])
        
        G = self.E / (2 * (1 + self.nu))
        la = (self.E * self.nu) / ((1 + self.nu) * (1 - 2 * self.nu))
        
        # Calculate stress components based on constitutive relations (Hooke's Law for isotropic materials)
        s1 = (la + 2 * G) * u_x + la * v_y + la * w_z   # sigma_xx
        s2 = (la + 2 * G) * v_y + la * u_x + la * w_z   # sigma_yy
        s3 = (la + 2 * G) * w_z + la * u_x + la * v_y   # sigma_zz
        
        s12 = G * (u_y + v_x)                             # sigma_xy
        s23 = G * (w_y + v_z)                             # sigma_yz
        s13 = G * (u_z + w_x)                             # sigma_xz
        
        return s1, s2, s3, s12, s23, s13

    def compute_traction_force(self, x, y, z, normal_x, normal_y, normal_z):
        """
        Computes surface traction forces.
        Based on Cauchy Stress Theorem: t_i = sigma_ij * n_j
        Args:
            x, y, z (torch.Tensor): Coordinates of surface points.
            normal_x, normal_y, normal_z (torch.Tensor): Components of outward normal vector at surface points.
        Returns:
            tuple: (t_x, t_y, t_z) Predicted surface traction force components.
        """
        s1, s2, s3, s12, s23, s13 = self.neural_net_s(x, y, z)
        
        # Traction force components
        t_x = s1 * normal_x + s12 * normal_y + s13 * normal_z
        t_y = s12 * normal_x + s2 * normal_y + s23 * normal_z
        t_z = s13 * normal_x + s23 * normal_y + s3 * normal_z
        
        return t_x, t_y, t_z

    def fwd_computing_loss_3d(self, loss_mode='MSE'):
        """
        Forward computation of total loss.
        Includes displacement boundary loss, equilibrium equation loss, and force boundary condition loss.
        Args:
            loss_mode (str): Type of loss, currently only 'MSE' is supported.
        Returns:
            tuple: (total_loss, [eq_loss, bc_loss, force_loss])
        """
        total_loss = 0.0
        
        # 1. Displacement Boundary Condition Loss (Dirichlet BCs)
        if hasattr(self, 'x_bc'):
            u_pred, v_pred, w_pred = self.neural_net_u(self.x_bc, self.y_bc, self.z_bc)
            
            if loss_mode == 'MSE':
                self.loss_b = (torch.mean(torch.square(u_pred - self.u_bc)) +
                              torch.mean(torch.square(v_pred - self.v_bc)) +
                              torch.mean(torch.square(w_pred - self.w_bc)))
            
            total_loss += self.alpha_b * self.loss_b

        # 2. Equilibrium Equation Loss (PDE Residual)
        if hasattr(self, 'x_f'):
            eq1_pred, eq2_pred, eq3_pred = self.neural_net_equation(self.x_f, self.y_f, self.z_f)
            
            if loss_mode == 'MSE':
                self.loss_e = (torch.mean(torch.square(eq1_pred)) +
                              torch.mean(torch.square(eq2_pred)) +
                              torch.mean(torch.square(eq3_pred)))
            
            total_loss += self.alpha_e * self.loss_e

        # 3. Force Boundary Condition Loss (Neumann BCs)
        # This part of the loss is kept to maintain "force constraint" functionality,
        # even though it was not in Code One. Its weight follows Code One's default weights.
        if hasattr(self, 'x_force'):
            t_x_pred, t_y_pred, t_z_pred = self.compute_traction_force(
                self.x_force, self.y_force, self.z_force,
                self.normal_x, self.normal_y, self.normal_z
            )
            
            if loss_mode == 'MSE':
                self.loss_force = (torch.mean(torch.square(t_x_pred - self.fx_target)) +
                                  torch.mean(torch.square(t_y_pred - self.fy_target)) +
                                  torch.mean(torch.square(t_z_pred - self.fz_target)))
            
            total_loss += self.alpha_force * self.loss_force

        return total_loss, [self.loss_e, self.loss_b, self.loss_force if hasattr(self, 'loss_force') else 0.0]

    def train(self, num_epoch=1000, lr=0.001, optimizer=None, scheduler=None, batchsize=None): # lr strictly adheres to Code One
        """
        Main function to train the model.
        """
        if optimizer is not None:
            self.opt = optimizer
        else:
            self.opt = torch.optim.Adam(params=self.net.parameters(), lr=lr, weight_decay=0) # Adam default weight_decay is 0
        
        self.solve_Adam(self.fwd_computing_loss_3d, num_epoch, batchsize, scheduler)

    def solve_Adam(self, loss_func, num_epoch=1000, batchsize=None, scheduler=None):
        """
        Core loop for training using Adam optimizer.
        """
        for epoch_id in range(num_epoch):
            # Compute loss
            loss, losses = loss_func()
            
            # Backpropagation and optimization
            loss.backward()
            self.opt.step()
            self.opt.zero_grad() # Clear gradients
            
            # Record losses
            e_loss = losses[0].detach().cpu().item() if hasattr(losses[0], 'detach') else losses[0]
            b_loss = losses[1].detach().cpu().item() if hasattr(losses[1], 'detach') else losses[1]
            f_loss = losses[2].detach().cpu().item() if hasattr(losses[2], 'detach') else losses[2]
            all_loss = loss.detach().cpu().item()
            
            # Record loss history every 10 epochs
            if epoch_id % 10 == 0:
                self.loss_bcs_all.append([b_loss])
                self.loss_equ_all.append([e_loss])
                self.loss_sum_all.append([all_loss]) # all_loss here includes force_loss
           
            # Learning rate scheduling
            if scheduler:
                scheduler.step()

            # Print training log
            if epoch_id == 0 or (epoch_id + 1) % 100 == 0:
                self.print_log(loss, losses, epoch_id, num_epoch)

    def print_log(self, loss, losses, epoch_id, num_epoch):
        """
        Prints log information for the current training epoch.
        """
        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']

        current_lr = get_lr(self.opt)
        eq_loss = losses[0].detach().cpu().item() if hasattr(losses[0], 'detach') else losses[0]
        bc_loss = losses[1].detach().cpu().item() if hasattr(losses[1], 'detach') else losses[1]
        force_loss = losses[2].detach().cpu().item() if hasattr(losses[2], 'detach') else losses[2]
        
        print(f"Epoch: {epoch_id + 1}/{num_epoch}, LR: {current_lr:.1e}, "
              f"Loss[Total]: {loss.detach().cpu().item():.3e}, "
              f"Loss[Eq]: {eq_loss:.3e}, "
              f"Loss[BC]: {bc_loss:.3e}, "
              f"Loss[Force]: {force_loss:.3e}")

    def predict(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        """
        Predicts displacements using the trained model.
        Args:
            x, y, z (np.ndarray): X, Y, Z coordinates of points to predict.
        Returns:
            tuple: (u, v, w) Predicted displacement components, as Numpy arrays.
        """
        # Ensure input is (N, 1) shape
        x_t = torch.tensor(x.reshape(-1, 1), dtype=torch.float32).to(device)
        y_t = torch.tensor(y.reshape(-1, 1), dtype=torch.float32).to(device)
        z_t = torch.tensor(z.reshape(-1, 1), dtype=torch.float32).to(device)
        
        self.net.eval() # Switch to evaluation mode
        with torch.no_grad(): # Disable gradient computation during prediction
            u, v, w = self.neural_net_u(x_t, y_t, z_t)
        self.net.train() # Switch back to training mode
        
        return u.cpu().numpy(), v.cpu().numpy(), w.cpu().numpy()

    def evaluate(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                 u_ref: Optional[np.ndarray]=None, v_ref: Optional[np.ndarray]=None, w_ref: Optional[np.ndarray]=None):
        """
        Evaluates model performance and visualizes results.
        Args:
            x, y, z (np.ndarray): X, Y, Z coordinates of evaluation points.
            u_ref, v_ref, w_ref (np.ndarray, optional): Reference (ground truth) displacements for error calculation.
        Returns:
            tuple: (u_pred, v_pred, w_pred) Predicted displacement components.
        """
        # Get predicted displacements
        u_pred, v_pred, w_pred = self.predict(x, y, z)
        
        # Calculate relative L2 error if reference solution is provided (simplified for sample data without ref solution)
        if u_ref is not None and v_ref is not None and w_ref is not None:
            u_test = u_ref.reshape(-1,1)
            v_test = v_ref.reshape(-1,1)
            w_test = w_ref.reshape(-1,1)
            
            error_u = np.linalg.norm(u_test-u_pred,2)/np.linalg.norm(u_test,2)
            error_v = np.linalg.norm(v_test-v_pred,2)/np.linalg.norm(v_test,2)
            error_w = np.linalg.norm(w_test-w_pred,2)/np.linalg.norm(w_test,2)
            
            print('------------------------')
            print('Evaluation Results:')
            print('Error u: %e' % (error_u))
            print('Error v: %e' % (error_v))
            print('Error w: %e' % (error_w))
            print('------------------------')
        else:
            print('------------------------')
            print('Airfoil Analysis Results (No Reference Solution Available):')
            print(f'U Displacement Range: [{u_pred.min():.6e}, {u_pred.max():.6e}] m')
            print(f'V Displacement Range: [{v_pred.min():.6e}, {v_pred.max():.6e}] m')
            print(f'W Displacement Range: [{w_pred.min():.6e}, {w_pred.max():.6e}] m')
            print('------------------------')

        # Visualize results
        self.visualize_airfoil_results(x, y, z, u_pred, v_pred, w_pred)
        return u_pred, v_pred, w_pred

    def visualize_airfoil_results(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                                  u: np.ndarray, v: np.ndarray, w: np.ndarray):
        """
        Visualizes airfoil displacement, von Mises stress, and training history.
        Args:
            x, y, z (np.ndarray): X, Y, Z coordinates of evaluation points.
            u, v, w (np.ndarray): Predicted displacement components.
        """
        fig = plt.figure(figsize=(20, 14)) # Adjust canvas size for more subplots
        
        # Ensure coordinates and displacement data are 1D numpy arrays
        x_flat, y_flat, z_flat = x.flatten(), y.flatten(), z.flatten()
        u_flat, v_flat, w_flat = u.flatten(), v.flatten(), w.flatten()

        # 1-3. Displacement component visualization (U, V, W)
        displacement_data = [u_flat, v_flat, w_flat]
        displacement_titles = ['U-Displacement (X-dir)', 'V-Displacement (Y-dir)', 'W-Displacement (Z-dir)']
        
        for i, (data, title) in enumerate(zip(displacement_data, displacement_titles)):
            ax = fig.add_subplot(2, 4, i+1, projection='3d')
            scatter = ax.scatter(x_flat, y_flat, z_flat, 
                               c=data, cmap='viridis', s=10, alpha=0.8)
            ax.set_xlabel('X [m]')
            ax.set_ylabel('Y [m]')
            ax.set_zlabel('Z [m]')
            ax.set_title(title)
            plt.colorbar(scatter, ax=ax, shrink=0.6, label='Displacement [m]')
            ax.view_init(elev=20, azim=-60) # Adjust view angle

        # 4. Total Displacement Magnitude visualization
        displacement_mag = np.sqrt(u_flat**2 + v_flat**2 + w_flat**2)
        ax4 = fig.add_subplot(2, 4, 4, projection='3d')
        scatter4 = ax4.scatter(x_flat, y_flat, z_flat,
                              c=displacement_mag, cmap='plasma', s=10, alpha=0.8)
        ax4.set_xlabel('X [m]')
        ax4.set_ylabel('Y [m]')
        ax4.set_zlabel('Z [m]')
        ax4.set_title('Total Displacement Magnitude')
        plt.colorbar(scatter4, ax=ax4, shrink=0.6, label='Displacement [m]')
        ax4.view_init(elev=20, azim=-60) # Adjust view angle

        # 5. Training History Loss Curves
        if self.loss_sum_all:
            ax5 = fig.add_subplot(2, 4, 5)
            # Convert list of lists to single list
            sum_losses = [l[0] for l in self.loss_sum_all]
            equ_losses = [l[0] for l in self.loss_equ_all]
            bcs_losses = [l[0] for l in self.loss_bcs_all]
            
            ax5.semilogy(sum_losses, 'b-', label='Total Loss')
            ax5.semilogy(equ_losses, 'r-', label='Equation Loss')
            ax5.semilogy(bcs_losses, 'g-', label='Displacement BC Loss')
            
            ax5.set_xlabel('Training Steps (×10)')
            ax5.set_ylabel('Loss Value (log scale)')
            ax5.set_title('Training History')
            ax5.legend()
            ax5.grid(True)

        # 6. Airfoil Geometry
        ax6 = fig.add_subplot(2, 4, 6, projection='3d')
        ax6.scatter(x_flat, y_flat, z_flat, 
                   c='blue', s=5, alpha=0.6)
        ax6.set_xlabel('X [m]')
        ax6.set_ylabel('Y [m]')
        ax6.set_zlabel('Z [m]')
        ax6.set_title('Airfoil Geometry (Evaluation Points)')
        ax6.view_init(elev=20, azim=-60) # Adjust view angle
        # Set equal aspect ratio for realistic shape observation
        max_range = np.array([x_flat.max()-x_flat.min(), y_flat.max()-y_flat.min(), z_flat.max()-z_flat.min()]).max() / 2.0
        mid_x = (x_flat.max()+x_flat.min()) * 0.5
        mid_y = (y_flat.max()+y_flat.min()) * 0.5
        mid_z = (z_flat.max()+z_flat.min()) * 0.5
        ax6.set_xlim(mid_x - max_range, mid_x + max_range)
        ax6.set_ylim(mid_y - max_range, mid_y + max_range)
        ax6.set_zlim(mid_z - max_range, mid_z + max_range)


        # 7. von Mises Stress Prediction
        if hasattr(self, 'x_f'): # Assume interior points are used for stress calculation
            x_f_np = self.x_f.detach().cpu().numpy().flatten()
            y_f_np = self.y_f.detach().cpu().numpy().flatten()
            z_f_np = self.z_f.detach().cpu().numpy().flatten()
            
            # Compute stress
            self.net.eval()
            with torch.no_grad():
                s1, s2, s3, s12, s23, s13 = self.neural_net_s(self.x_f, self.y_f, self.z_f)
                # Compute von Mises stress
                von_mises = torch.sqrt(0.5 * ((s1 - s2)**2 + (s2 - s3)**2 + (s3 - s1)**2 + 
                                              6 * (s12**2 + s23**2 + s13**2)))
                von_mises_np = von_mises.detach().cpu().numpy().flatten()
            self.net.train()
                
            # Sample a subset of points for visualization to avoid slow rendering
            n_plot = min(5000, len(x_f_np))
            indices = np.random.choice(len(x_f_np), n_plot, replace=False)
            
            ax7 = fig.add_subplot(2, 4, 7, projection='3d')
            scatter7 = ax7.scatter(x_f_np[indices], y_f_np[indices], z_f_np[indices],
                                  c=von_mises_np[indices], cmap='hot', s=10, alpha=0.8)
            ax7.set_xlabel('X [m]')
            ax7.set_ylabel('Y [m]')
            ax7.set_zlabel('Z [m]')
            ax7.set_title('von Mises Stress')
            plt.colorbar(scatter7, ax=ax7, shrink=0.6, label='Stress [Pa]')
            ax7.view_init(elev=20, azim=-60) # Adjust view angle

        # 8. Statistics Information
        ax8 = fig.add_subplot(2, 4, 8)
        ax8.axis('off') # Hide axes
        stats_text = f"""
Airfoil Analysis Statistics:

Displacement Range:
U: [{u.min():.3e}, {u.max():.3e}] m
V: [{v.min():.3e}, {v.max():.3e}] m
W: [{w.min():.3e}, {w.max():.3e}] m

Max Total Displacement: {displacement_mag.max():.3e} m

Material Parameters:
Young's Modulus E = {self.E:.1f}
Poisson's Ratio ν = {self.nu:.2f}

PINN Training Parameters:
Displacement BC Weight = {self.alpha_b}
Equation PDE Weight = {self.alpha_e}
Force BC Weight = {self.alpha_force}
        """
        ax8.text(0.1, 0.9, stats_text, transform=ax8.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout for suptitle
        plt.suptitle("3D Airfoil Elasticity PINN Results", fontsize=16)
        plt.savefig('airfoil_pinn_results.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save(self, filename: str, directory: Optional[str]=None, N_HLayer=None, N_neu=None, N_f=None):
        """
        Saves model weights, optimizer state, and training history.
        """
        if directory is None:
            directory = './models/'
        
        os.makedirs(directory, exist_ok=True) # Ensure directory exists
        filepath = os.path.join(directory, filename)
        
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'E': self.E,
            'nu': self.nu,
            'N_HLayer': N_HLayer,
            'N_neu': N_neu,
            'N_f': N_f,
            'loss_history': {
                'loss_sum_all': self.loss_sum_all,
                'loss_equ_all': self.loss_equ_all,
                'loss_bcs_all': self.loss_bcs_all
            }
        }, filepath)
        print(f"Model saved to: {filepath}")

# ============================================================================
# MAIN TRAINING AND TESTING FUNCTIONS
# (Adapted from train.py and test.py logic)
# ============================================================================

def train_airfoil_model(net_params: Optional[str]=None, loop: int=0):
    """
    Trains the 3D airfoil PINN model.
    Args:
        net_params (str, optional): Path to pre-trained model file, if provided.
        loop (int): Identifier for training loop, used for filename.
    Returns:
        AirfoilStructuralPINN: Trained PINN model instance.
    """
    
    # Material parameters (Strict adherence to Code One)
    E = 100       # Young's Modulus
    nu = 0.3      # Poisson's Ratio

    # Neural network and training parameters (Strict adherence to Code One)
    N_neu = 50        # Number of neurons in hidden layers
    lam_bcs = 1       # Weight for displacement BC loss
    lam_equ = 1       # Weight for equation loss
    lam_force = 1     # Weight for force BC loss (consistent with other Code One weights)
    N_f_eq_points = 20000 # Number of interior equation points
    N_HLayer = 10     # Number of hidden layers
    learning_rate = 0.001 # Initial learning rate
    
    print("\n=== Airfoil PINN Training Start ===")
    print(f"Material Parameters: E={E:.1f}, ν={nu:.2f}")
    print(f"Network Parameters: {N_HLayer} layers x {N_neu} neurons")
    print(f"Loss Weights: Disp BC={lam_bcs}, Eq={lam_equ}, Force={lam_force}")

    # Initialize Airfoil PINN model
    PINN = AirfoilStructuralPINN(
        E=E,
        nu=nu,
        layers=N_HLayer,
        hidden_size=N_neu,
        bc_weight=lam_bcs,
        eq_weight=lam_equ,
        force_weight=lam_force,
        net_params=net_params
    )

    # Initialize data loader
    dataloader = AirfoilDataLoader(path='./data/')

    # Load airfoil geometry data, and generate displacement and force BCs from it
    print("\n=== Preparing Data ===")
    # Default to loading 'airfoil_data.mat', if not found, create and load 'sample_airfoil.mat'
    bc_data, force_bc_data = dataloader.loading_boundary_data('airfoil_data.mat')  
    
    # Set data to PINN model
    if bc_data is None or force_bc_data is None:
        print("Data loading failed, cannot proceed with training.")
        return None
        
    PINN.set_boundary_data(bc_data)       # Displacement boundary conditions
    PINN.set_force_boundary_data(force_bc_data) # Force boundary conditions

    # Load interior training data (for PDE constraint)
    training_data = dataloader.loading_training_data('sample_airfoil.mat', n_points=N_f_eq_points)
    if training_data[0] is None:
        print("Interior training data loading failed, cannot proceed with training.")
        return None
    PINN.set_eq_training_data(training_data)

    # Load evaluation data (typically points on the entire geometry)
    x_star, y_star, z_star = dataloader.loading_evaluate_data('sample_airfoil.mat')
    if x_star is None:
        print("Evaluation data loading failed, cannot proceed with training.")
        return None

    # Training process (Multi-stage training strategy)
    # The number of epochs is significantly increased for better convergence.
    print("\n=== Starting Training ===")
    
    # Stage 1
    print("Stage 1: 10000 Epochs, LR=0.001")
    PINN.train(num_epoch=10000, lr=learning_rate)
    
    # Stage 2
    print("Stage 2: 10000 Epochs, LR=0.0002") # Learning rate decay (0.001 * 0.2)
    PINN.train(num_epoch=10000, lr=learning_rate * 0.2)
    
    # Stage 3
    print("Stage 3: 5000 Epochs, LR=0.00004") # Learning rate decay (0.001 * 0.04)
    PINN.train(num_epoch=5000, lr=learning_rate * 0.04)

    # Stage 4
    print("Stage 4: 10000 Epochs, LR=0.00002") # Further learning rate decay (0.001 * 0.02)
    PINN.train(num_epoch=10000, lr=learning_rate * 0.02)

    # Stage 5
    print("Stage 5: 15000 Epochs, LR=0.000004") # Even further learning rate decay (0.001 * 0.004)
    PINN.train(num_epoch=15000, lr=learning_rate * 0.004)
    
    total_epochs = 10000 + 10000 + 5000 + 10000 + 15000
    print(f"\nTraining Complete! Total Epochs: {total_epochs}")

    # Evaluate model (Note: No reference solution provided, so L2 error won't be calculated, only prediction ranges and visualization.)
    print("\n=== Evaluating Model ===")
    PINN.evaluate(x_star, y_star, z_star)

    # Save model
    print("\n=== Saving Model ===")
    saved_ckpt = f'airfoil_model_loop{loop}.pth'
    PINN.save(saved_ckpt, N_HLayer=N_HLayer, N_neu=N_neu, N_f=N_f_eq_points)
    
    print("=== Airfoil PINN Training Finished ===")
    return PINN

def test_airfoil_model(net_params: Optional[str]=None):
    """
    Loads a trained model and evaluates it.
    Args:
        net_params (str, optional): Path to trained model file, must be provided.
    Returns:
        AirfoilStructuralPINN: Loaded PINN model instance.
    """
    print("=== Testing Airfoil PINN Model ===")
    
    if net_params is None or not os.path.exists(net_params):
        print(f"Error: Model file '{net_params}' does not exist. Please train the model first or provide the correct path.")
        return None
    
    # Parameters (Strict adherence to Code One, must match training configuration)
    E = 100
    nu = 0.3
    N_neu = 50
    lam_bcs = 1
    lam_equ = 1
    lam_force = 1
    N_HLayer = 10

    # Create model and load weights
    PINN = AirfoilStructuralPINN(
        E=E, nu=nu, layers=N_HLayer, hidden_size=N_neu,
        bc_weight=lam_bcs, eq_weight=lam_equ, force_weight=lam_force,
        net_params=net_params # Load model by providing path
    )

    # Load evaluation data
    dataloader = AirfoilDataLoader(path='./data/')
    x_star, y_star, z_star = dataloader.loading_evaluate_data('sample_airfoil.mat')
    
    if x_star is None:
        print("Evaluation data loading failed, cannot proceed with testing.")
        return None

    # Evaluate
    PINN.evaluate(x_star, y_star, z_star)
    print("=== Model Testing Finished ===")
    return PINN

# ============================================================================
# MAIN PROGRAM
# ============================================================================

def main():
    """Main program entry point"""
    print("=== 3D Airfoil Force-Constrained PINN System ===")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"NumPy Version: {np.__version__}")
    print(f"SciPy Version: {scipy.__version__}")
    print(f"Matplotlib Version: {matplotlib.__version__}") # Use matplotlib.__version__
    print(f"Using device: {device}")
    
    # --- Run mode selection ---
    # Options: "train" - Train a new model and evaluate
    #          "test"  - Load a trained model and evaluate
    mode = "train" 
    
    if mode == "train":
        print("\n【TRAINING MODE】")
        # Can loop through multiple instances, here only one
        for loop in range(0, 1): 
            print(f"\n--- Training Instance {loop + 1} ---")
            model = train_airfoil_model(loop=loop)
        
    elif mode == "test":
        print("\n【TESTING MODE】")
        # Specify path to the trained model file
        model_path = './models/airfoil_model_loop0.pth' # Ensure this file exists
        model = test_airfoil_model(net_params=model_path)
    else:
        print("Invalid run mode. Please set 'mode' to 'train' or 'test'.")
    
    print("\n=== Program Execution Complete ===")
    print("\nPlease check the following output files:")
    print(f"  - airfoil_pinn_results.png: Visualization of training results")
    print(f"  - ./models/airfoil_model_loop*.pth: Trained model weights file")
    print(f"  - ./data/sample_airfoil.mat: Auto-generated sample airfoil data file (if your data did not exist)")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nProgram execution error: {e}")
        import traceback
        traceback.print_exc()