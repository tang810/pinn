import math 
import numpy as np
import matplotlib.pyplot as plt

def generate_winged_block_mesh(n_interior=5000, n_boundary_face=500):
    """
    生成一个"带翅膀的方块"几何体的网格点。
    该形状由一个中心方块和两个侧翼组成。
    """
    # 1. 定义几何形状尺寸
    main_x = [-0.5, 0.5]  # 长度 1.0
    main_y = [-0.5, 0.5]  # 长度 1.0
    main_z = [-0.5, 0.5]  # 长度 1.0
    
    # 翅膀的定义
    wing_x = [-0.5, 0.5]
    wing_y_left = [-1.0, -0.5]  # 左翼
    wing_y_right = [0.5, 1.0]   # 右翼
    wing_z = [-0.05, 0.05]      # 厚度

    # 辅助函数：检查一个点是否在几何体内
    def is_inside(p):
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        in_main = (x >= main_x[0]) & (x <= main_x[1]) & (y >= main_y[0]) & (y <= main_y[1]) & (z >= main_z[0]) & (z <= main_z[1])
        in_left_wing = (x >= wing_x[0]) & (x <= wing_x[1]) & (y >= wing_y_left[0]) & (y <= wing_y_left[1]) & (z >= wing_z[0]) & (z <= wing_z[1])
        in_right_wing = (x >= wing_x[0]) & (x <= wing_x[1]) & (y >= wing_y_right[0]) & (y <= wing_y_right[1]) & (z >= wing_z[0]) & (z <= wing_z[1])
        return in_main | in_left_wing | in_right_wing

    # 2. 生成内部点 (使用拒绝采样)
    bounding_box = np.array([
        [min(main_x[0], wing_x[0]), max(main_x[1], wing_x[1])],
        [wing_y_left[0], wing_y_right[1]],
        [main_z[0], main_z[1]]
    ], dtype='float64')  # 明确指定数据类型
    
    # 验证 bounding_box 形状
    assert bounding_box.shape == (3, 2), f"bounding_box shape error: {bounding_box.shape}"
    
    interior_points = []
    while len(interior_points) < n_interior:
        n_sample = (n_interior - len(interior_points)) * 2
        bb_min = bounding_box[:, 0].reshape(-1, 1)
        bb_max = bounding_box[:, 1].reshape(-1, 1)
        random_vals = np.random.rand(3, n_sample)
        points = (bb_min + (bb_max - bb_min) * random_vals).T
        valid_points = points[is_inside(points)]
        interior_points.extend(valid_points)
    x_interior = np.array(interior_points[:n_interior])

    def generate_face_points(n_pts, fixed_dim, fixed_val, dim1_range, dim2_range):
        pts = np.random.rand(n_pts, 2)
        pts[:, 0] = dim1_range[0] + (dim1_range[1] - dim1_range[0]) * pts[:, 0]
        pts[:, 1] = dim2_range[0] + (dim2_range[1] - dim2_range[0]) * pts[:, 1]
        
        full_pts = np.full((n_pts, 3), fixed_val)
        dims = [0, 1, 2]
        data_dims = [d for d in dims if d != fixed_dim]
        full_pts[:, data_dims[0]] = pts[:, 0]
        full_pts[:, data_dims[1]] = pts[:, 1]
        return full_pts

    # 生成边界点
    x1u = generate_face_points(n_boundary_face, 0, main_x[1], main_y, main_z)
    x1b = generate_face_points(n_boundary_face, 0, main_x[0], main_y, main_z)
    x2u = generate_face_points(n_boundary_face, 1, wing_y_right[1], wing_x, wing_z)
    x2b = generate_face_points(n_boundary_face, 1, wing_y_left[0], wing_x, wing_z)
    x3u = generate_face_points(n_boundary_face, 2, main_z[1], main_x, main_y)
    x3b = generate_face_points(n_boundary_face, 2, main_z[0], main_x, main_y)

    # 返回 run_pinn_training 期望的格式
    mesh_data = {
        'x': x_interior,
        'x1u': x1u,
        'x1b': x1b,
        'x2u': x2u,
        'x2b': x2b,
        'x3u': x3u,
        'x3b': x3b,
        'n': np.array([[n_interior + 6 * n_boundary_face]])  # 总点数，2D数组格式
    }
    
    return mesh_data

def visualize_geometry_three_views(mesh_data):
    """
    使用 Matplotlib 创建一个包含三个2D投影视图的静态图来检查几何体的连续性。

    参数:
        mesh_data (dict): 从 generate_winged_block_mesh 返回的字典。
    """
    # 将所有点合并，并为每个点分配一个类别标签用于着色
    # 0: 内部, 1: 主体, 2: 左翼, 3: 右翼
    points = []
    colors = []

    # 定义几何尺寸以便分类
    main_y, wing_y_left, wing_y_right = [-0.5, 0.5], [-1.0, -0.5], [0.5, 1.0]

    all_points = np.vstack(list(mesh_data.values())[:-1]) # 合并所有坐标点

    for p in all_points:
        points.append(p)
        y = p[1]
        if main_y[0] <= y <= main_y[1]:
            colors.append(1) # 主体
        elif wing_y_left[0] <= y < main_y[0]:
            colors.append(2) # 左翼
        elif main_y[1] < y <= wing_y_right[1]:
            colors.append(3) # 右翼
        else:
            colors.append(0) # 理论上不应该有其他点
            
    points = np.array(points)
    colors = np.array(colors)

    # 创建一个 2x2 的子图网格
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2)

    # 主视图 (Front View, X-Y plane)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(points[:, 0], points[:, 1], c=colors, s=1, cmap='viridis')
    ax1.set_title('主视图 (Front View / Y-X Plane)')
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y-axis')
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 俯视图 (Top View, X-Z plane)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(points[:, 0], points[:, 2], c=colors, s=1, cmap='viridis')
    ax2.set_title('俯视图 (Top View / Z-X Plane)')
    ax2.set_xlabel('X-axis')
    ax2.set_ylabel('Z-axis')
    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # 左视图 (Side View, Y-Z plane)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(points[:, 1], points[:, 2], c=colors, s=1, cmap='viridis')
    ax3.set_title('左视图 (Side View / Z-Y Plane)')
    ax3.set_xlabel('Y-axis')
    ax3.set_ylabel('Z-axis')
    ax3.set_aspect('equal', adjustable='box')
    ax3.grid(True, linestyle='--', alpha=0.6)
    
    # 3D 视图作为参考
    ax4 = fig.add_subplot(gs[1, 1], projection='3d')
    ax4.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=0.5, cmap='viridis')
    ax4.set_title('3D 参考视图')
    ax4.set_xlabel('X'); ax4.set_ylabel('Y'); ax4.set_zlabel('Z')
    # 调整视角
    ax4.view_init(elev=20, azim=-65)


    fig.suptitle('几何体验证三视图', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()