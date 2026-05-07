import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .utils import material
from .networks import MLP
from .config import device


class MultiFNN(nn.Module):
    """
    物理信息神经网络，包含三个子网络分别用于u, v, w分量
    """
    def __init__(self, layers):
        super(MultiFNN, self).__init__()
        
        # 为每个分量创建子网络
        self.u_net = MLP(layers).to(device)
        self.v_net = MLP(layers).to(device)
        self.w_net = MLP(layers).to(device)
        
        # 日志记录器
        self.loss_log = []
        self.loss_res_log = []
    
    def u_model(self, x, y, z):
        """u分量模型"""
        # 检查输入形状
        inputs = torch.stack([x, y, z], dim=-1)
        # 确保形状匹配避免广播
        output = self.u_net(inputs)
        output = output.squeeze(-1)
        return output * x
    
    def v_model(self, x, y, z):
        """v分量模型"""
        # 检查输入形状
        inputs = torch.stack([x, y, z], dim=-1)
        # 确保形状匹配避免广播
        output = self.v_net(inputs)
        output = output.squeeze(-1)
        return output * y
    
    def w_model(self, x, y, z):
        """w分量模型"""
        # 检查输入形状
        inputs = torch.stack([x, y, z], dim=-1)
        # 确保形状匹配避免广播
        output = self.w_net(inputs)
        output = output.squeeze(-1)
        return output * z  
    
    def compute_derivatives(self, x, y, z):
        """使用自动微分计算所需的导数"""
        # 使坐标需要梯度
        x.requires_grad_(True)
        y.requires_grad_(True)
        z.requires_grad_(True)

        # 前向传播
        u = self.u_model(x, y, z)
        v = self.v_model(x, y, z)
        w = self.w_model(x, y, z)
        
        # 一阶导数
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_z = torch.autograd.grad(v, z, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        
        w_x = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(w), create_graph=True)[0]
        w_y = torch.autograd.grad(w, y, grad_outputs=torch.ones_like(w), create_graph=True)[0]
        w_z = torch.autograd.grad(w, z, grad_outputs=torch.ones_like(w), create_graph=True)[0]
        
        # 二阶导数
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_xy = torch.autograd.grad(u_x, y, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_xz = torch.autograd.grad(u_x, z, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        u_yz = torch.autograd.grad(u_y, z, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        u_zz = torch.autograd.grad(u_z, z, grad_outputs=torch.ones_like(u_z), create_graph=True)[0]
        
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_xy = torch.autograd.grad(v_x, y, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_xz = torch.autograd.grad(v_x, z, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
        v_yz = torch.autograd.grad(v_y, z, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]   
        v_zz = torch.autograd.grad(v_z, z, grad_outputs=torch.ones_like(v_z), create_graph=True)[0]
        
        w_xx = torch.autograd.grad(w_x, x, grad_outputs=torch.ones_like(w_x), create_graph=True)[0]
        w_xy = torch.autograd.grad(w_x, y, grad_outputs=torch.ones_like(w_x), create_graph=True)[0]
        w_xz = torch.autograd.grad(w_x, z, grad_outputs=torch.ones_like(w_x), create_graph=True)[0]
        
        w_yy = torch.autograd.grad(w_y, y, grad_outputs=torch.ones_like(w_y), create_graph=True)[0]
        w_yz = torch.autograd.grad(w_y, z, grad_outputs=torch.ones_like(w_y), create_graph=True)[0]   
        w_zz = torch.autograd.grad(w_z, z, grad_outputs=torch.ones_like(w_z), create_graph=True)[0]

        # 打印梯度

            
        return (u_x, u_y, u_z, u_xx, u_xy, u_xz, u_yy, u_yz, u_zz,
                v_x, v_y, v_z, v_xx, v_xy, v_xz, v_yy, v_yz, v_zz,
                w_x, w_y, w_z, w_xx, w_xy, w_xz, w_yy, w_yz, w_zz)
    
    def forward(self, batch):
        """前向传播计算损失分量"""
        x, x1u, x1b, x2u, x2b, x3u, x3b, y = batch
        # 内部域
        a, b, c = x[ :,0], x[ :,1], x[:,2]
        # 计算内部点的导数

        derivs = self.compute_derivatives(a, b, c)
    
        # 获取材料残差
        _, _, _, _, _, _, _, _, _, _, _, _, Gex, Gey, Gez = material(*derivs)

        
        # 计算边界导数 - x-y-z平面 Up/Bottom
        a1u, b1u, c1u = x1u[:, 0], x1u[ :,1], x1u[:, 2]
        a1b, b1b, c1b = x1b[:, 0], x1b[ :,1], x1b[:, 2]
        a2u, b2u, c2u = x2u[:, 0], x2u[:, 1], x2u[:, 2]
        a2b, b2b, c2b = x2b[:, 0], x2b[:, 1], x2b[:,2]
        a3u, b3u, c3u = x3u[:, 0], x3u[:, 1], x3u[:,2]
        a3b, b3b, c3b = x3b[ :,0], x3b[ :,1], x3b[:,2]
        
        # 计算边界点的导数
        derivs_1u = self.compute_derivatives(a1u, b1u, c1u)
        derivs_1b = self.compute_derivatives(a1b, b1b, c1b)
        derivs_2u = self.compute_derivatives(a2u, b2u, c2u)
        derivs_2b = self.compute_derivatives(a2b, b2b, c2b)
        derivs_3u = self.compute_derivatives(a3u, b3u, c3u)
        derivs_3b = self.compute_derivatives(a3b, b3b, c3b)
        
        # 获取每个边界的材料特性
        _, _, _, _, _, _, s11u, _, _, s121u, _, s131u, _, _, _ = material(*derivs_1u)
        _, _, _, _, _, _, _, _, _, s121b, _, s131b, _, _, _ = material(*derivs_1b)
        _, _, _, _, _, _, _, s22u, _, s122u, s232u, _, _, _, _ = material(*derivs_2u)
        _, _, _, _, _, _, _, _, _, s122b, s232b, _, _, _, _ = material(*derivs_2b)
        _, _, _, _, _, _, _, _, s33u, _, s233u, s133u, _, _, _ = material(*derivs_3u)
        _, _, _, _, _, _, _, _, _, _, s233b, s133b, _, _, _ = material(*derivs_3b)
        # 计算损失
        l1 = torch.sum(Gex**2) + torch.sum(Gey**2) + torch.sum(Gez**2)
        
        lx = (torch.sum(s11u**2) + torch.sum(s121u**2) + torch.sum(s131u**2) + 
              torch.sum(s121b**2) + torch.sum(s131b**2))
        
        ly = (torch.sum(s22u**2) + torch.sum(s122u**2) + torch.sum(s232u**2) + 
              torch.sum(s122b**2) + torch.sum(s232b**2))
        
        lz = (torch.sum((s33u-y)**2) + torch.sum(s233u**2) + torch.sum(s133u**2) + 
              torch.sum(s233b**2) + torch.sum(s133b**2))
        
        l2 = lx + ly + lz
        
        # 总损失
        loss = l1 + l2
        
        return l1, lx, ly, lz, loss
    
    
    def predict(self, x_coords, batch_size=9261):
        n_samples = x_coords.shape[0]
        u_result = np.zeros(n_samples)
        v_result = np.zeros(n_samples)
        w_result = np.zeros(n_samples)
        bs = int(batch_size)
        for i in range(0, n_samples, bs):
            end = min(i + bs, n_samples)
            x_batch = x_coords[i:end]
            x = torch.tensor(x_batch[:, 0], dtype=torch.float32).to(device)
            y = torch.tensor(x_batch[:, 1], dtype=torch.float32).to(device)
            z = torch.tensor(x_batch[:, 2], dtype=torch.float32).to(device)

            with torch.no_grad():
                u = self.u_model(x, y, z)
                v = self.v_model(x, y, z)
                w = self.w_model(x, y, z)


                assert u.shape[0] == (end - i)
                assert v.shape[0] == (end - i)
                assert w.shape[0] == (end - i)

                u_result[i:end] = u.cpu().numpy()
                v_result[i:end] = v.cpu().numpy()
                w_result[i:end] = w.cpu().numpy()

            del u, v, w, x, y, z

        return u_result, v_result, w_result


    def predict_stress(self, x_coords, batch_size=9261):
        """使用批处理方式预测应力，避免内存溢出"""

        num_points = x_coords.shape[0]
        # 计算需要的批次数量
        num_batches = (num_points + batch_size - 1) // batch_size

        # 预先分配存储结果的 NumPy 数组
        # 假设应变和应力是单个浮点数值，形状与输入点数相同
        # 如果它们是张量或有其他维度，请调整 shape
        e1_results = np.empty(num_points, dtype=np.float32)
        e2_results = np.empty(num_points, dtype=np.float32)
        e3_results = np.empty(num_points, dtype=np.float32)
        e12_results = np.empty(num_points, dtype=np.float32)
        e23_results = np.empty(num_points, dtype=np.float32)
        e13_results = np.empty(num_points, dtype=np.float32)
        s1_results = np.empty(num_points, dtype=np.float32)
        s2_results = np.empty(num_points, dtype=np.float32)
        s3_results = np.empty(num_points, dtype=np.float32)
        s12_results = np.empty(num_points, dtype=np.float32)
        s23_results = np.empty(num_points, dtype=np.float32)
        s13_results = np.empty(num_points, dtype=np.float32)
        gx_results = np.empty(num_points, dtype=np.float32)
        gy_results = np.empty(num_points, dtype=np.float32)
        gz_results = np.empty(num_points, dtype=np.float32)


        # 循环处理每个批次
        for i in range(num_batches):
            # 计算当前批次的起始和结束索引
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_points)

            # 提取当前批次的坐标
            x_batch = x_coords[start_idx:end_idx, :]

            # 将当前批次转换为 Tensor 并移动到设备
            # 确保这里只处理 batch_size 个点，而不是整个数据集
            x = torch.tensor(x_batch[:, 0], dtype=torch.float32).to(device)
            y = torch.tensor(x_batch[:, 1], dtype=torch.float32).to(device)
            z = torch.tensor(x_batch[:, 2], dtype=torch.float32).to(device)

            # 计算当前批次的导数
            # derivs 中的张量大小是 [当前批次大小, ...]
            derivs = self.compute_derivatives(x, y, z)
            # 计算当前批次的应变/应力/残差
            # e, s, g 中的张量大小是 [当前批次大小, ...]
            
            e1, e2, e3, e12, e23, e13, s1, s2, s3, s12, s23, s13, gx, gy, gz = material(*derivs)
            # 将当前批次的结果转换为 NumPy 数组并存储到预分配的总结果数组中
            # 这里只处理并存储一个批次的结果
            e1_results[start_idx:end_idx] = e1.detach().cpu().numpy().flatten() # .flatten() 如果结果是单值
            e2_results[start_idx:end_idx] = e2.detach().cpu().numpy().flatten()
            e3_results[start_idx:end_idx] = e3.detach().cpu().numpy().flatten()
            e12_results[start_idx:end_idx] = e12.detach().cpu().numpy().flatten()
            e23_results[start_idx:end_idx] = e23.detach().cpu().numpy().flatten()
            e13_results[start_idx:end_idx] = e13.detach().cpu().numpy().flatten()
            s1_results[start_idx:end_idx] = s1.detach().cpu().numpy().flatten()
            s2_results[start_idx:end_idx] = s2.detach().cpu().numpy().flatten()
            s3_results[start_idx:end_idx] = s3.detach().cpu().numpy().flatten()
            s12_results[start_idx:end_idx] = s12.detach().cpu().numpy().flatten()
            s23_results[start_idx:end_idx] = s23.detach().cpu().numpy().flatten()
            s13_results[start_idx:end_idx] = s13.detach().cpu().numpy().flatten()
            gx_results[start_idx:end_idx] = gx.detach().cpu().numpy().flatten()
            gy_results[start_idx:end_idx] = gy.detach().cpu().numpy().flatten()
            gz_results[start_idx:end_idx] = gz.detach().cpu().numpy().flatten()

            # 释放当前批次不再需要的内存
            # (PyTorch/TensorFlow 的垃圾回收通常会自动处理，但显式删除有时有帮助)
            del x, y, z, derivs, e1, e2, e3, e12, e23, e13, s1, s2, s3, s12, s23, s13, gx, gy, gz
            torch.cuda.empty_cache() # 如果使用 CUDA

        # 返回累积了所有批次结果的 NumPy 数组
        return (e1_results, e2_results, e3_results, e12_results, e23_results, e13_results,
                s1_results, s2_results, s3_results, s12_results, s23_results, s13_results,
                gx_results, gy_results, gz_results)