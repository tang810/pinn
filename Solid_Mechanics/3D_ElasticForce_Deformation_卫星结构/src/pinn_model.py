import torch
import numpy as np
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, layers):
        super(MLP, self).__init__()
        modules = []
        for i in range(len(layers)-2):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            modules.append(nn.Tanh())
        modules.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*modules)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.net(x)

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MultiFNN(nn.Module):
    def __init__(self, layers):
        super(MultiFNN, self).__init__()
        self.u_net = MLP(layers).to(device)
        self.v_net = MLP(layers).to(device)
        self.w_net = MLP(layers).to(device)
        self.loss_log = []

    def u_model(self, x, y, z): return self.u_net(torch.stack([x, y, z], dim=-1)).squeeze(-1) * x
    def v_model(self, x, y, z): return self.v_net(torch.stack([x, y, z], dim=-1)).squeeze(-1) * y
    def w_model(self, x, y, z): return self.w_net(torch.stack([x, y, z], dim=-1)).squeeze(-1) * z

    def compute_derivatives(self, x, y, z):
        x.requires_grad_(True); y.requires_grad_(True); z.requires_grad_(True)
        u, v, w = self.u_model(x, y, z), self.v_model(x, y, z), self.w_model(x, y, z)
        grad_u = torch.autograd.grad(u, (x,y,z), torch.ones_like(u), create_graph=True); u_x, u_y, u_z = grad_u
        grad_v = torch.autograd.grad(v, (x,y,z), torch.ones_like(v), create_graph=True); v_x, v_y, v_z = grad_v
        grad_w = torch.autograd.grad(w, (x,y,z), torch.ones_like(w), create_graph=True); w_x, w_y, w_z = grad_w
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
        u_zz = torch.autograd.grad(u_z, z, torch.ones_like(u_z), create_graph=True)[0]
        u_xy = torch.autograd.grad(u_x, y, torch.ones_like(u_x), create_graph=True)[0]
        u_xz = torch.autograd.grad(u_x, z, torch.ones_like(u_x), create_graph=True)[0]
        u_yz = torch.autograd.grad(u_y, z, torch.ones_like(u_y), create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, torch.ones_like(v_y), create_graph=True)[0]
        v_zz = torch.autograd.grad(v_z, z, torch.ones_like(v_z), create_graph=True)[0]
        v_xy = torch.autograd.grad(v_x, y, torch.ones_like(v_x), create_graph=True)[0]
        v_xz = torch.autograd.grad(v_x, z, torch.ones_like(v_x), create_graph=True)[0]
        v_yz = torch.autograd.grad(v_y, z, torch.ones_like(v_y), create_graph=True)[0]
        w_xx = torch.autograd.grad(w_x, x, torch.ones_like(w_x), create_graph=True)[0]
        w_yy = torch.autograd.grad(w_y, y, torch.ones_like(w_y), create_graph=True)[0]
        w_zz = torch.autograd.grad(w_z, z, torch.ones_like(w_z), create_graph=True)[0]
        w_xy = torch.autograd.grad(w_x, y, torch.ones_like(w_x), create_graph=True)[0]
        w_xz = torch.autograd.grad(w_x, z, torch.ones_like(w_x), create_graph=True)[0]
        w_yz = torch.autograd.grad(w_y, z, torch.ones_like(w_y), create_graph=True)[0]
        return (u_x,u_y,u_z,u_xx,u_xy,u_xz,u_yy,u_yz,u_zz, v_x,v_y,v_z,v_xx,v_xy,v_xz,v_yy,v_yz,v_zz, w_x,w_y,w_z,w_xx,w_xy,w_xz,w_yy,w_yz,w_zz)

    def forward(self, batch):
        x, x1u, x1b, x2u, x2b, x3u, x3b, y = batch
        derivs = self.compute_derivatives(x[:, 0], x[:, 1], x[:, 2])
        *_, Gex, Gey, Gez = material(*derivs)
        s = [material(*self.compute_derivatives(b[:, 0], b[:, 1], b[:, 2])) for b in [x1u, x1b, x2u, x2b, x3u, x3b]]
        l1 = torch.mean(Gex**2) + torch.mean(Gey**2) + torch.mean(Gez**2)
        lx = torch.mean(s[0][6]**2) + torch.mean(s[0][9]**2) + torch.mean(s[0][11]**2) + torch.mean(s[1][9]**2) + torch.mean(s[1][11]**2)
        ly = torch.mean(s[2][7]**2) + torch.mean(s[2][9]**2) + torch.mean(s[2][10]**2) + torch.mean(s[3][9]**2) + torch.mean(s[3][10]**2)
        lz = torch.mean((s[4][8]-y)**2) + torch.mean(s[4][10]**2) + torch.mean(s[4][11]**2) + torch.mean(s[5][10]**2) + torch.mean(s[5][11]**2)
        loss = l1 + lx + ly + lz
        return loss, l1, lx, ly, lz
    
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
        '''只返回应力分量'''
        results = self.predict_full_mechanical_response(x_coords, batch_size)
        # 只返回应力分量（索引6-11）
        return results[6:12]

    def predict_strain(self, x_coords, batch_size=9261):
        """只返回应变分量"""
        results = self.predict_full_mechanical_response(x_coords, batch_size)
        # 只返回应变分量（索引0-5）
        return results[0:6]

    def predict_gradients(self, x_coords, batch_size=9261):
        """只返回梯度分量"""
        results = self.predict_full_mechanical_response(x_coords, batch_size)
        # 只返回梯度分量（索引12-14）
        return results[12:15]

    def predict_full_mechanical_response(self, x_coords, batch_size=9261):
        """返回完整的力学响应（原来的predict_stress函数内容）"""
        num_points = x_coords.shape[0]
        num_batches = (num_points + batch_size - 1) // batch_size

        # 预先分配存储结果的 NumPy 数组
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
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_points)
            x_batch = x_coords[start_idx:end_idx, :]

            x = torch.tensor(x_batch[:, 0], dtype=torch.float32).to(device)
            y = torch.tensor(x_batch[:, 1], dtype=torch.float32).to(device)
            z = torch.tensor(x_batch[:, 2], dtype=torch.float32).to(device)

            derivs = self.compute_derivatives(x, y, z)           
            e1, e2, e3, e12, e23, e13, s1, s2, s3, s12, s23, s13, gx, gy, gz = material(*derivs)
            
            # 存储所有结果
            e1_results[start_idx:end_idx] = e1.detach().cpu().numpy().flatten()
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

            del x, y, z, derivs, e1, e2, e3, e12, e23, e13, s1, s2, s3, s12, s23, s13, gx, gy, gz
            torch.cuda.empty_cache()

        # 返回所有15个分量
        return (e1_results, e2_results, e3_results, e12_results, e23_results, e13_results,
                s1_results, s2_results, s3_results, s12_results, s23_results, s13_results,
                gx_results, gy_results, gz_results)
    
def material(*derivs, E=1.0, mu=0.25):
    u_x, u_y, u_z, u_xx, u_xy, u_xz, u_yy, u_yz, u_zz, \
    v_x, v_y, v_z, v_xx, v_xy, v_xz, v_yy, v_yz, v_zz, \
    w_x, w_y, w_z, w_xx, w_xy, w_xz, w_yy, w_yz, w_zz = derivs
    la = E * mu / (1 + mu) / (1 - 2 * mu); G = E / (1 + mu) / 2
    e1,e2,e3 = u_x, v_y, w_z; e12,e23,e13 = 0.5*(u_y+v_x), 0.5*(w_y+v_z), 0.5*(u_z+w_x)
    s1,s2,s3 = (2*G+la)*e1+la*e2+la*e3, (2*G+la)*e2+la*e1+la*e3, (2*G+la)*e3+la*e1+la*e2
    s12,s23,s13 = 2*G*e12, 2*G*e23, 2*G*e13
    Gex = (G+la)*(u_xx+v_xy+w_xz) + G*(u_xx+u_yy+u_zz)
    Gey = (G+la)*(v_yy+u_xy+w_yz) + G*(v_xx+v_yy+v_zz)
    Gez = (G+la)*(w_zz+u_xz+v_yz) + G*(w_xx+w_yy+w_zz)
    return e1, e2, e3, e12, e23, e13, s1, s2, s3, s12, s23, s13, Gex, Gey, Gez
