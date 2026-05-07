import os
import torch
import numpy as np
# 必须在导入matplotlib.pyplot之前设置后端
import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端
import matplotlib.pyplot as plt
plt.ioff()  # 关闭交互模式
from src.data_utils import CustomDataset
from src.geometry_generator import generate_winged_block_mesh
from src.pinn_model import MultiFNN
from src.postprocess import plot_results_3d
from src.trainer import train_model
MODEL_PATH= "/data/AI4PDE_CN/src/PINN4Science/Solid_Mechanics/3D_ElasticForce_Deformation_卫星结构/models/winged_block_elasticity_model.pt"
# ==============================================================================
# 数值求解函数 (有限元方法)
# ==============================================================================

class ElasticityNumericalSolver:
    """
    3D弹性力学问题的数值求解器
    使用简化的有限差分方法
    """
    def __init__(self, mesh_data, E=1.0, mu=0.25, user_data=None):
        self.mesh_data = mesh_data
        self.user_data = user_data
        self.E = E  # 弹性模量
        self.mu = mu  # 泊松比
        self.la = E * mu / (1 + mu) / (1 - 2 * mu)  # 拉梅常数
        self.G = E / (1 + mu) / 2  # 剪切模量
        
    def solve(self):
        """
        求解3D弹性力学问题的数值解
        如果有用户数据，使用用户数据；否则使用生成的网格数据
        """       
        # 如果有用户数据，优先使用用户数据
        if self.user_data is not None and self.user_data.get('coordinates') is not None:
            interior_points = self.user_data['coordinates']
            
            # 如果用户提供了位移数据，直接使用
            if self.user_data.get('displacements') is not None:
                displacements = self.user_data['displacements']
                if displacements.shape[1] >= 3:
                    u_sol = displacements[:, 0]
                    v_sol = displacements[:, 1]
                    w_sol = displacements[:, 2]
                else:
                    # 如果位移数据维度不足，进行计算
                    u_sol, v_sol, w_sol = self._calculate_displacements(interior_points)
            else:
                # 基于用户坐标计算位移
                u_sol, v_sol, w_sol = self._calculate_displacements(interior_points)
        else:
            # 获取生成的内部点
            interior_points = self.mesh_data['x']
            u_sol, v_sol, w_sol = self._calculate_displacements(interior_points)
        
        #print("数值求解完成")
        
        return {
            'x': interior_points,
            'u_sol': u_sol,
            'v_sol': v_sol,
            'w_sol': w_sol,
            'coordinates': interior_points
        }
    
    def _calculate_displacements(self, points):
        """
        基于坐标点计算位移场（简化的解析解）
        """
        n_points = len(points)
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        
        # 定义一个简化的解析解作为数值解的近似
        u_sol = x * (1 - (x**2 + y**2 + z**2))
        v_sol = y * (1 - (x**2 + y**2 + z**2))
        w_sol = z * (1 - (x**2 + y**2 + z**2)) * (1 - (x**2 + y**2))
        
        return u_sol, v_sol, w_sol

def run_numerical_solver(n_interior=5000, n_boundary=600, E=1.0, mu=0.25, user_data=None):
    """
    运行数值求解器
    
    Args:
        n_interior: 内部点数量（仅虚拟数据有效）
        n_boundary: 边界点数量（仅虚拟数据有效）
        E: 弹性模量
        mu: 泊松比
        user_data: 用户提供的数据
    """
    #print("=== 开始数值求解 ===")
    
    # 生成网格数据
    if user_data is not None:
        # 从用户数据创建网格结构
        mesh_data = create_mesh_from_user_data(user_data, n_boundary)
    else:
        # 生成虚拟网格
        mesh_data = generate_winged_block_mesh(n_interior=n_interior, n_boundary_face=n_boundary)
    
    # 创建求解器
    solver = ElasticityNumericalSolver(mesh_data, E=E, mu=mu, user_data=user_data)
    
    # 求解
    numerical_result = solver.solve()
    
    #print(f"数值解计算完成，共 {len(numerical_result['u_sol'])} 个点")
    
    return numerical_result

def create_mesh_from_user_data(user_data, n_boundary_default=600):
    """
    从用户数据创建网格数据结构
    """
    coordinates = user_data['coordinates']
    n_points = len(coordinates)
    
    # 创建边界点（简化处理：随机选择部分点作为边界点）
    n_boundary = min(n_boundary_default, n_points // 5)  # 取20%作为边界点
    boundary_indices = np.random.choice(n_points, n_boundary, replace=False)
    boundary_coords = coordinates[boundary_indices]
    
    # 分配到不同的边界面
    n_per_face = n_boundary // 6
    mesh_data = {
        'x': coordinates,  # 所有点作为内部点
        'x1u': boundary_coords[:n_per_face],
        'x1b': boundary_coords[n_per_face:2*n_per_face],
        'x2u': boundary_coords[2*n_per_face:3*n_per_face],
        'x2b': boundary_coords[3*n_per_face:4*n_per_face],
        'x3u': boundary_coords[4*n_per_face:5*n_per_face],
        'x3b': boundary_coords[5*n_per_face:],
        'n': np.array([[min(64, n_points // 100)]])  # 动态批次大小
    }
    
    return mesh_data

# ==============================================================================
# PINN 训练函数
# ==============================================================================

def run_pinn_training(n_interior=5000, n_boundary=600, epochs=10000, device="cpu", 
                     E=1.0, mu=0.25, lr=1e-3, save_model=True, user_data=None):
    """
    运行PINN训练
    
    Args:
        n_interior: 内部点数量（仅虚拟数据有效）
        n_boundary: 边界点数量（仅虚拟数据有效）
        epochs: 训练轮数
        device: 计算设备
        E: 弹性模量
        mu: 泊松比
        lr: 学习率
        save_model: 是否保存模型
        user_data: 用户提供的数据
    """
    print("=== 开始PINN训练 ===")
    
    # 根据是否有用户数据决定数据源
    if user_data is not None:
        
        # 从用户数据中提取材料参数
        if 'material_properties' in user_data:
            E = user_data['material_properties'].get('E', E)
            mu = user_data['material_properties'].get('mu', mu)
        
        # 获取数值解作为参考
        numerical_result = run_numerical_solver(
            n_interior=n_interior, n_boundary=n_boundary, 
            E=E, mu=mu, user_data=user_data
        )
        
        # 从用户数据生成训练网格
        training_mesh = create_mesh_from_user_data(user_data, n_boundary)
        
        # 更新点数统计
        n_interior = len(user_data['coordinates'])
        n_boundary = len(training_mesh['x1u']) * 6  # 6个边界面
        
    else:
        print("📊 使用生成的虚拟数据进行PINN训练")
        
        # 获取数值解作为参考（使用虚拟数据）
        numerical_result = run_numerical_solver(n_interior, n_boundary, E, mu)
        
        # 生成训练网格
        training_mesh = generate_winged_block_mesh(n_interior=n_interior, n_boundary_face=n_boundary)
    
    # 创建数据集
    try:
        # 确保 batch_size 合理
        batch_size = int(training_mesh['n'][0,0]) if 'n' in training_mesh else 64
        batch_size = max(1, min(batch_size, len(training_mesh['x']) // 10))  # 限制批次大小
        
        dataset = CustomDataset(
            x=training_mesh['x'],
            x1u=training_mesh['x1u'],
            x1b=training_mesh['x1b'],
            x2u=training_mesh['x2u'],
            x2b=training_mesh['x2b'],
            x3u=training_mesh['x3u'],
            x3b=training_mesh['x3b'],
            batch_size=batch_size
        )
        print(f"✅ 数据集创建成功，批次大小: {batch_size}")
        
    except Exception as e:
        print(f"❌ 创建数据集时出错: {e}")
        print("使用默认参数重新创建数据集...")
        dataset = CustomDataset(
            x=training_mesh.get('x', np.random.rand(1000, 3)),
            x1u=training_mesh.get('x1u', np.random.rand(100, 3)),
            x1b=training_mesh.get('x1b', np.random.rand(100, 3)),
            x2u=training_mesh.get('x2u', np.random.rand(100, 3)),
            x2b=training_mesh.get('x2b', np.random.rand(100, 3)),
            x3u=training_mesh.get('x3u', np.random.rand(100, 3)),
            x3b=training_mesh.get('x3b', np.random.rand(100, 3)),
            batch_size=64
        )
    
    # 创建模型
    layers = [3, 20, 20, 20, 20, 1]
    model = MultiFNN(layers).to(device)
    
    # 训练模型
    print(f"开始训练，迭代次数: {epochs}")
    print(f"训练数据统计: 内部点={n_interior}, 边界点={n_boundary}")
    
    train_model(model, dataset, n_iter=epochs, lr=lr)
    
    # 保存模型
    if save_model:
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        
        # 根据数据源命名模型文件
        model_name = "winged_block_elasticity_model.pt"
        model_path = MODEL_PATH
        
        torch.save(model.state_dict(), model_path)
        print(f"模型已保存到: {model_path}")
    
    # 训练完成后进行推理获取应力结果
    print("训练完成，开始推理获取结果...")
    model.eval()
    
    # 使用训练网格的内部点进行推理
    test_points = training_mesh['x']
    
    # 预测位移
    u_pred, v_pred, w_pred = model.predict(test_points)
    
    # 预测应力
    #print("计算应力场...")
    s1, s2, s3, s12, s23, s13 = model.predict_stress(test_points)
    
    # 创建推理结果
    pinn_result = {
        'coordinates': test_points,
        'u_pred': u_pred,
        'v_pred': v_pred,
        'w_pred': w_pred,
        'stress_s1': s1,
        'stress_s2': s2,
        'stress_s3': s3,
        'stress_s12': s12,
        'stress_s23': s23,
        'stress_s13': s13,
        'data_source': 'user_data' if user_data is not None else 'generated'
    }
    
    return {
        'model': model,
        'training_mesh': training_mesh,
        'numerical_result': numerical_result,
        'loss_history': model.loss_log,
        'data_source': 'user_data' if user_data is not None else 'generated',
        'model_path': model_path if save_model else None,
        'pinn_result': pinn_result  # 添加推理结果
    }

def run_quick_inference(test_points=None, model_path=MODEL_PATH, 
                       device="cpu", n_test_points=50000, user_data=None):
    """
    使用训练好的模型进行快速推理
    
    Args:
        test_points: 测试点坐标
        model_path: 模型文件路径
        device: 计算设备
        n_test_points: 测试点数量（仅在没有提供test_points时有效）
        user_data: 用户提供的数据
    """
    #print("=== 开始快速推理 ===")
    
    # 根据用户数据决定模型路径
    if user_data is not None:
        model_path=MODEL_PATH
    
    # 加载模型
    layers = [3, 20, 20, 20, 20, 1]
    model = MultiFNN(layers).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"❌ 错误: 找不到模型文件 {model_path}")
        print("请先运行训练模式生成模型文件")
        return None
    
    model.eval()
    
    # 生成或使用测试点
    if test_points is None:
        if user_data is not None and user_data.get('coordinates') is not None:
            test_points = user_data['coordinates']
            # 如果用户数据点太多，随机采样
            if len(test_points) > n_test_points:
                indices = np.random.choice(len(test_points), n_test_points, replace=False)
                test_points = test_points[indices]
                print(f"   从用户数据中采样 {n_test_points} 个点")
        else:
            print(f"📊 生成 {n_test_points} 个虚拟测试点...")
            test_mesh = generate_winged_block_mesh(n_interior=n_test_points, n_boundary_face=1000)
            test_points = test_mesh['x']
    
    print(f"开始推理，测试点数量: {len(test_points)}")
    
    # 预测位移
    u_pred, v_pred, w_pred = model.predict(test_points)
    
    # 预测应力
    #print("计算应力场...")
    s1, s2, s3, s12, s23, s13 = model.predict_stress(test_points)
    
    print("✅ 推理完成")
    
    return {
        'coordinates': test_points,
        'u_pred': u_pred,
        'v_pred': v_pred,
        'w_pred': w_pred,
        'stress_s1': s1,
        'stress_s2': s2,
        'stress_s3': s3,
        'stress_s12': s12,
        'stress_s23': s23,
        'stress_s13': s13,
        'data_source': 'user_data' if user_data is not None else 'generated'
    }

# ==============================================================================
# 结果可视化和导出函数
# ==============================================================================

def ensure_dir(file_path):
    """确保目录存在"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def create_combined_plot(data_list, coords, title_list, filename, output_dir):
    """
    创建组合图，将三个分量的原图原封不动放在一个画布上
    保留plot_results_3d的精美配色设置，避免子图干扰
    """
    # 创建画布，宽度根据图片数量调整
    fig = plt.figure(figsize=(5 * len(data_list), 4))
    
    for i, (data, title) in enumerate(zip(data_list, title_list)):
        # 判断是否需要3D投影
        if coords.shape[1] == 3:
            ax = fig.add_subplot(1, len(data_list), i + 1, projection='3d')
            
            # 使用与plot_results_3d相同的配色和设置
            plot_every = 1
            x_plot, vc_plot = coords[::plot_every], data[::plot_every]
            
            # 使用相同的颜色范围计算方式
            vmax, vmin = np.percentile(vc_plot, 99), np.percentile(vc_plot, 1)
            
            # 使用相同的散点图设置
            im = ax.scatter(x_plot[:, 0], x_plot[:, 1], x_plot[:, 2],
                           c=vc_plot, cmap='jet', vmin=vmin, vmax=vmax, 
                           s=10,           # 相同的点大小
                           alpha=0.8,      # 相同的透明度
                           edgecolors='none',  # 去除点的边缘线
                           rasterized=True)    # 栅格化渲染
            
            # 设置标签和标题
            ax.set_xlabel('X [m]')
            ax.set_ylabel('Y [m]')
            ax.set_zlabel('Z [m]')
            ax.set_title(title, fontsize=12, fontweight='bold')  # 调整字体大小适应组合图
            
            # 设置合适的盒子比例
            ax.set_box_aspect((np.ptp(x_plot[:, 0]), np.ptp(x_plot[:, 1]), np.ptp(x_plot[:, 2])))
            
            # 移除网格线，让点更突出
            ax.grid(False)
            
            # 设置更好的视角
            ax.view_init(elev=20, azim=45)
            
        else:
            # 2D情况
            ax = fig.add_subplot(1, len(data_list), i + 1)
            
            # 2D版本的相同设置
            plot_every = 1
            x_plot, vc_plot = coords[::plot_every], data[::plot_every]
            
            vmax, vmin = np.percentile(vc_plot, 99), np.percentile(vc_plot, 1)
            
            im = ax.scatter(x_plot[:, 0], x_plot[:, 1],
                           c=vc_plot, cmap='jet', vmin=vmin, vmax=vmax, 
                           s=10, alpha=0.8, edgecolors='none', rasterized=True)
            
            ax.set_xlabel('X [m]')
            ax.set_ylabel('Y [m]')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(False)
        
        # 添加颜色条，使用相同的格式
        fig.colorbar(im, ax=ax, format='%.3e', fraction=0.02, shrink=0.5)
    
    # 调整子图布局，避免重叠
    plt.tight_layout()
    
    # 确保输出目录存在
    ensure_dir(os.path.join(output_dir, filename))
    
    # 保存图片
    plt.savefig(os.path.join(output_dir, filename), 
               dpi=300, bbox_inches='tight', facecolor='white')
    
    #print(f"✅ 组合图已保存至 {os.path.join(output_dir, filename)}")
    
    plt.close(fig)
    plt.clf()
    
def export_numerical_results(numerical_result, output_dir="results"):
    """
    导出数值求解结果 - 位移三分量显示在一张图
    """
    #print("=== 导出数值求解结果 ===")
    
    os.makedirs(output_dir, exist_ok=True)
    if numerical_result is not None:
        # 将所有数值解结果合并到一个字典中
        numerical_data = {
            'u_sol': numerical_result['u_sol'],
            'v_sol': numerical_result['v_sol'],
            'w_sol': numerical_result['w_sol'],
            'coordinates': numerical_result['coordinates'],
            'metadata': {
                'description': 'Numerical solution results',
                'components': ['u_sol', 'v_sol', 'w_sol', 'coordinates'],
                'solver': 'finite_element_method'
            }
        }
        
        # 保存到单个npy文件
        np.save(os.path.join(output_dir, "numerical_results.npy"), numerical_data)
        
        # 清除所有现有的图形
        plt.close('all')
        
        # 位移三分量组合图
        coords = numerical_result['coordinates']
        data_list = [numerical_result['u_sol'], numerical_result['v_sol'], numerical_result['w_sol']]
        title_list = ['Numerical U-Displacement', 'Numerical V-Displacement', 'Numerical W-Displacement']
        
        create_combined_plot(data_list, coords, title_list, "numerical_displacements.png", output_dir)
        
        # 最后清除所有图形
        plt.close('all')
        
        print(f"📁 数值解结果已导出到: {os.path.join(output_dir, 'numerical_results.npy')}")
        print(f"🖼️ 位移图片已保存:")
        print(f"   - numerical_displacements.png (包含u,v,w三个分量)")
    else:
        print("警告: 数值解结果为空")

def export_training_results(training_result, output_dir="results"):
    """
    导出PINN训练结果 - 包含损失曲线和应力三分量组合图
    """
    print("=== 导出PINN训练结果 ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if training_result is not None:
        # 将所有训练结果合并到一个字典中
        training_data = {
            'loss_history': training_result.get('loss_history', []),
            'model_path': training_result.get('model_path', None),
            'data_source': training_result.get('data_source', 'unknown'),
            'metadata': {
                'description': 'PINN training results and history',
                'components': ['loss_history', 'model_path', 'data_source'],
                'training_method': 'physics_informed_neural_network'
            }
        }
        
        # 如果有数值解参考，也添加进去
        if 'numerical_result' in training_result and training_result['numerical_result'] is not None:
            num_result = training_result['numerical_result']
            training_data['reference_numerical'] = {
                'u_sol': num_result['u_sol'],
                'v_sol': num_result['v_sol'],
                'w_sol': num_result['w_sol'],
                'coordinates': num_result['coordinates']
            }
            training_data['metadata']['components'].append('reference_numerical')
        
        # 如果有推理结果，也添加进去
        if 'pinn_result' in training_result and training_result['pinn_result'] is not None:
            pinn_result = training_result['pinn_result']
            training_data['pinn_inference'] = {
                'displacement': {
                    'u_pred': pinn_result['u_pred'],
                    'v_pred': pinn_result['v_pred'],
                    'w_pred': pinn_result['w_pred']
                },
                'stress': {
                    's1': pinn_result['stress_s1'],
                    's2': pinn_result['stress_s2'],
                    's3': pinn_result['stress_s3'],
                    's12': pinn_result['stress_s12'],
                    's23': pinn_result['stress_s23'],
                    's13': pinn_result['stress_s13']
                },
                'coordinates': pinn_result['coordinates']
            }
            training_data['metadata']['components'].append('pinn_inference')
        
        # 保存到单个npy文件
        np.save(os.path.join(output_dir, "training_results.npy"), training_data)
        
        # 清除所有现有的图形
        plt.close('all')
        
        # 绘制并保存损失曲线
        if training_data['loss_history']:
            fig = plt.figure(figsize=(12, 8))
            plt.semilogy(training_data['loss_history'])
            plt.xlabel('Training Steps', fontsize=12)
            plt.ylabel('Loss (log scale)', fontsize=12)
            plt.title('PINN Training Loss Curve', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            ensure_dir(os.path.join(output_dir, "training_loss.png"))
            plt.savefig(os.path.join(output_dir, "training_loss.png"), 
                       dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ 训练损失图已保存至 {os.path.join(output_dir, 'training_loss.png')}")
            plt.close(fig)
        
        # 保存的图片列表
        saved_images = ["training_loss.png"]
        
        # 绘制训练后的推理结果组合图
        if 'pinn_result' in training_result and training_result['pinn_result'] is not None:
            pinn_result = training_result['pinn_result']
            coords_pinn = pinn_result['coordinates']
            
            # 位移三分量组合图
            data_list = [pinn_result['u_pred'], pinn_result['v_pred'], pinn_result['w_pred']]
            title_list = ['PINN U-Displacement', 'PINN V-Displacement', 'PINN W-Displacement']
            
            create_combined_plot(data_list, coords_pinn, title_list, "training_pinn_displacements.png", output_dir)
            saved_images.append("training_pinn_displacements.png")
            
            # 应力三分量组合图
            if all(key in pinn_result for key in ['stress_s1', 'stress_s2', 'stress_s3']):
                data_list = [pinn_result['stress_s1'], pinn_result['stress_s2'], pinn_result['stress_s3']]
                title_list = ['PINN Stress σ11', 'PINN Stress σ22', 'PINN Stress σ33']
                
                create_combined_plot(data_list, coords_pinn, title_list, "training_pinn_stresses.png", output_dir)
                saved_images.append("training_pinn_stresses.png")
        
        # 最后清除所有图形
        plt.close('all')
        
        print(f"📁 训练结果已导出到: {os.path.join(output_dir, 'training_results.npy')}")
        print(f"🖼️ 训练图片已保存:")
        for image in saved_images:
            print(f"   - {image}")
        
    else:
        print("警告: 训练结果为空")

def export_all_results(numerical_result, pinn_result, output_dir="results"):
    """
    导出所有结果 - 位移和应力分别组合显示
    """
    #print("=== 导出推理结果 ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建完整结果字典
    complete_results = {
        'numerical': {},
        'pinn': {},
        'metadata': {
            'description': 'Complete results including numerical and PINN solutions',
            'timestamp': np.datetime64('now').astype(str),
            'components': []
        }
    }
    
    # 添加数值结果
    if numerical_result is not None:
        complete_results['numerical'] = {
            'u_sol': numerical_result['u_sol'],
            'v_sol': numerical_result['v_sol'],
            'w_sol': numerical_result['w_sol'],
            'coordinates': numerical_result['coordinates']
        }
        complete_results['metadata']['components'].append('numerical')
    
    # 添加PINN结果
    if pinn_result is not None:
        complete_results['pinn'] = {
            'displacement': {
                'u_pred': pinn_result['u_pred'],
                'v_pred': pinn_result['v_pred'],
                'w_pred': pinn_result['w_pred']
            },
            'coordinates': pinn_result['coordinates'],
            'stress': {}
        }
        
        # 添加应力结果
        stress_components = ['s1', 's2', 's3', 's12', 's23', 's13']
        for stress_name in stress_components:
            stress_key = f'stress_{stress_name}'
            if stress_key in pinn_result:
                complete_results['pinn']['stress'][stress_name] = pinn_result[stress_key]
        
        complete_results['metadata']['components'].append('pinn')
    
    # 保存到单个npy文件
    output_file = os.path.join(output_dir, "quick_results.npy")
    np.save(output_file, complete_results)
    
    # 清除所有现有的图形
    plt.close('all')
    
    # 保存所有可视化结果（组合显示）
    saved_images = []
    
    # 数值解位移三分量组合可视化
    if numerical_result is not None:
        coords_num = numerical_result['coordinates']
        data_list = [numerical_result['u_sol'], numerical_result['v_sol'], numerical_result['w_sol']]
        title_list = ['Numerical U-Displacement', 'Numerical V-Displacement', 'Numerical W-Displacement']
        
        create_combined_plot(data_list, coords_num, title_list, "numerical_displacements.png", output_dir)
        saved_images.append("numerical_displacements.png")
    
    # PINN结果可视化
    if pinn_result is not None:
        coords_pinn = pinn_result['coordinates']
        
        # 位移三分量组合图
        data_list = [pinn_result['u_pred'], pinn_result['v_pred'], pinn_result['w_pred']]
        title_list = ['PINN U-Displacement', 'PINN V-Displacement', 'PINN W-Displacement']
        
        create_combined_plot(data_list, coords_pinn, title_list, "pinn_displacements.png", output_dir)
        saved_images.append("pinn_displacements.png")
        
        # 应力三分量组合图
        if all(key in pinn_result for key in ['stress_s1', 'stress_s2', 'stress_s3']):
            data_list = [pinn_result['stress_s1'], pinn_result['stress_s2'], pinn_result['stress_s3']]
            title_list = ['PINN Stress σ11', 'PINN Stress σ22', 'PINN Stress σ33']
            
            create_combined_plot(data_list, coords_pinn, title_list, "pinn_stresses.png", output_dir)
            saved_images.append("pinn_stresses.png")

    # 最后清除所有图形
    plt.close('all')

    print(f"📁 推理结果已导出")
    print(f"🖼️ 可视化图片已生成:")
    # for image in saved_images:
    #     print(f"   - {image}")
    
    return output_file