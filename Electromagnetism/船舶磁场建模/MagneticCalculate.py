import numpy as np
import torch
import math
from ReadData import ReadData_normal,ReadData
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") # Run on CPU

# 设置默认数据类型为 float64
torch.set_default_dtype(torch.float64)

# 定义矩阵方程的正问题
def forward_problem(xn,yn,zn,m_inputs,theta,Ln,Wn):
    ## inputs：磁矩输入形状为（n,3+48），theta：航向/夹角（1，），L,W:船长和船宽
    # print(f'Ln.shape:{Ln.shape}')
    # print(f'Wn.shape:{Wn.shape}')
    # print(f'xn.shape:{xn.shape}')
    # 给定的物理常数和矩阵参数
    mu_0 = torch.tensor(4 * math.pi * 1e-7, dtype=torch.float64).to(device)  # 真空磁导率
    # 提取输入变量 x, y, z, m_x, m_y, m_z 形状均为为[16，1]
    Hxyz = torch.zeros(m_inputs.shape[0],3)
    for n in range(m_inputs.shape[0]):
        ## 坐标系转换矩阵（{b}到{s}系转换矩阵）
        tran_matrix = torch.tensor([[math.cos(theta[n]), math.sin(theta[n]), 0.0],
                                    [-math.sin(theta[n]), math.cos(theta[n]), 0.0],
                                    [0.0, 0.0, 1.0]], dtype=torch.float64).to(device)
        # 尝试计算逆矩阵
        try:
            tran_matrix_inverse = torch.linalg.inv(tran_matrix)
        except torch.linalg.LinAlgError:
            tran_matrix_inverse = torch.zeros_like(tran_matrix)
            return "该矩阵不可逆"
        # print(matrix_inverse)
        # x, y, z = xyz[n, 0:1].to(torch.float64), xyz[n, 1:2].to(torch.float64),\
        #         xyz[n, 2:3].to(torch.float64)
        # x, y, z = xn[n].to(torch.float64), yn[n].to(torch.float64), zn[n].to(torch.float64)
        # print("xn.shape:",xn.shape)
        # print("yn.shape:",yn.shape)
        # print("zn.shape:",zn.shape)
        x, y, z = xn[n], yn[n], zn[n]
        # print("x:",x)
        # print("x.shape:",x.shape)
        # print("y.shape:",y.shape)
        # print("z.shape:",z.shape)
        # m = inputs[n, 3:].to(torch.float64)  ## 形状为（48，1）
        m = m_inputs[n]
        # print(f'm.shape:{m.shape}')
        m_xyz = m.reshape(-1,3)   ## 形状为（16，3）
        # print(f'mxyz.shape:{m_xyz.shape}')
        m_x, m_y, m_z = m_xyz[:, 0:1], m_xyz[:, 1:2], m_xyz[:, 2:3]
        # 给定的物理常数和矩阵参数
        L = Ln[n]
        W = Wn[n]
        # print("船长：",L)
        # print("船宽：",W)
        # print("航向（夹角）：",theta)
        # 磁偶极子的数量 M
        M = m_x.shape[0]-1
        dL = L/(M-1)
        # print("磁偶极子的数量：",M)
        ## 磁偶极子的坐标 形状:(15,3)  + 一个椭球体
        r_i = torch.zeros(M,3).to(torch.float64).to(device)
        r_i_norm = torch.zeros(M,1).to(torch.float64).to(device)
        # print("r_i_norm.shape:",r_i_norm.shape)
        # print("x.shape:",x.shape)
        # print("y.shape:",y.shape)
        # print("z.shape:",z.shape)
        # print("L.shape:",L.shape)
        for i in range(M):
            r_i[i] = torch.tensor([x-(-0.5*L + (i)* dL),y,z], dtype=torch.float64).to(device)
            r_i_norm[i] = torch.norm(r_i[i],2)
        # print("磁传感器在{b}系中相对第i个磁偶极子的坐标:",r_i)
        # 计算磁偶极子的系数矩阵
        xi = r_i[:,0].unsqueeze(1)
        # print(xi)
        yi = r_i[:,1].unsqueeze(1)
        zi = r_i[:,2].unsqueeze(1)
        # print("xi_shape:",xi.shape)
        a_xi = (mu_0 / (4*torch.pi)) * ((3/(r_i_norm**5) *(xi**2))-(1/(r_i_norm**3)))
        a_yi = (3 * mu_0 * xi * yi) / (4 * torch.pi * (r_i_norm**5))
        a_zi = (3 * mu_0 * xi * zi) / (4 * torch.pi * (r_i_norm**5))

        b_yi = (mu_0/(4 * torch.pi)) * ((3 / (r_i_norm**5))* (yi**2) - (1/(r_i_norm**3)))
        b_zi = (3 * mu_0 * yi * zi) / (4 * torch.pi * (r_i_norm**5))

        c_zi = (mu_0/(4 * torch.pi)) * ((3 / (r_i_norm**5))* (zi**2) - (1/(r_i_norm**3)))
        b_xi, c_xi, c_yi = a_yi, a_zi, b_zi

        # print("a_xi.shape",a_xi.shape)
        
        # 计算 B_x, B_y, B_z b系 i=1...M
        B_x = a_xi * m_x[:-1] + a_yi * m_y[:-1] + a_zi * m_z[:-1]
        B_y = b_xi * m_x[:-1] + b_yi * m_y[:-1] + b_zi * m_z[:-1]
        B_z = c_xi * m_x[:-1] + c_yi * m_y[:-1] + c_zi * m_z[:-1]
        # print("B_x.shape:",B_x.shape)

        # i = M+1 时，为椭球体模型
        # print("传感器在b系中的坐标:", (x,y,z))
        # 计算向量 r 的范数 ||r||
        # r_norm = torch.sqrt(x[-1] ** 2 + y[-1] ** 2 + z[-1] ** 2)
        r_norm = torch.norm(torch.tensor([x,y,z])).to(device)
        # 计算参数 K, A, B, t
        K = torch.sqrt((L/2) ** 2 - (W/2) ** 2)  
        t = torch.sqrt((r_norm ** 2 + K ** 2) ** 2 - 4 * (K ** 2) * (x ** 2))
        A = torch.sqrt(0.5 * (r_norm ** 2 + K ** 2 + t))
        B = torch.sqrt(0.5 * (r_norm ** 2 - K ** 2 + t))

        # 计算矩阵元素 a_x, a_y, a_z
        a_x = (3 * mu_0 / (4 * torch.pi)) * (A / (K ** 2 * t) + (1 / (2 * K ** 3)) * torch.log((A - K) / (A + K)))
        a_y = (3 * mu_0 / (4 * torch.pi)) * (x * y / (A * (B ** 2) * t))
        a_z = (3 * mu_0 / (4 * torch.pi)) * (x * z / (A * (B ** 2) * t))

        b_y = (3 * mu_0 / (8 * torch.pi)) * ((2 * A * (y ** 2)) / (B ** 4 * t) - A / (B ** 2 * K ** 2) - (1 / (2 * K ** 3)) * torch.log((A - K) / (A + K)))
        b_z = (3 * mu_0 / (4 * torch.pi)) * (A * y * z / (B ** 4 * t))
        c_z = (3 * mu_0 / (8 * torch.pi)) * ((2 * A * (z ** 2)) / (B ** 4 * t) - A / (B ** 2 * K ** 2) - (1 / (2 * K ** 3)) * torch.log((A - K) / (A + K)))

        b_x, c_x, c_y = a_y, a_z, b_z
        # print("a_x.shape",a_x.shape)
        # 计算 B_x, B_y, B_z b系 i=M+1时
        B_x_t = a_x * m_x[-1] + a_y * m_y[-1] + a_z * m_z[-1]
        B_y_t = b_x * m_x[-1] + b_y * m_y[-1] + b_z * m_z[-1]
        B_z_t = c_x * m_x[-1] + c_y * m_y[-1] + c_z * m_z[-1]
        # print("B_x_t.shape",B_x_t.unsqueeze(1).shape)
        ## 磁偶极子和椭球体
        B_x = torch.cat([B_x,B_x_t.unsqueeze(1)],dim=0)
        B_y = torch.cat([B_y,B_y_t.unsqueeze(1)],dim=0)
        B_z = torch.cat([B_z,B_z_t.unsqueeze(1)],dim=0)
        # print("B_x.shape:",B_x.shape)
        B_x_fin = torch.sum(B_x)
        B_y_fin = torch.sum(B_y)
        B_z_fin = torch.sum(B_z)
        B_xyz = torch.stack([B_x_fin,B_y_fin,B_z_fin])
        # print("B_xyz.shape:",B_xyz.shape)
        B_sxyz = (B_xyz @ tran_matrix) *1e+9
        # print("B_sxyz.shape:",B_sxyz.shape)
        # s系
        # B_sx = B_x_fin* math.cos(theta[n]) - B_y_fin * math.sin(theta[n])
        # B_sy = B_x_fin* math.sin(theta[n]) + B_y_fin * math.cos(theta[n])
        # B_sz = B_z_fin
        # B_sxyz = torch.stack([B_sx,B_sy,B_sz])*1e+9
        Hxyz[n,:] = B_sxyz
        # print("B_sx:",B_sx)
        # print("B_sx.shape:",B_sx.shape)
    return Hxyz.to(device)


if __name__ == "__main__":
    ## test
    file_path = '/save/magnetic/Data/Dataset/Ship_SNR=0dB/'
    data = ReadData(file_path)
    print(data.head())
    print("Hx describe:",data["Hx"].describe())  # 数值列的统计信息
    print("Hy describe:",data["Hy"].describe())  # 数值列的统计信息
    print("Hz describe:",data["Hz"].describe())  # 数值列的统计信息
    Hx,Hy,Hz = data["Hx"].values,data["Hy"].values,data["Hz"].values
    z = data["z"].values
    H_data = np.hstack([Hx.reshape(-1, 1),Hy.reshape(-1, 1),Hz.reshape(-1, 1),z.reshape(-1, 1)])
    H_data = torch.tensor(H_data,dtype=torch.float64)
    print(f"H_data 形状: {H_data.shape}")

    columns_ship_data = data.iloc[:, 4:]
    ship_data = columns_ship_data.values
    print("ship data columns:",columns_ship_data.columns)  # 数值列的统计信息
    # print("ship data describe:",columns_ship_data.describe())  # 数值列的统计信息
    # print(f'data:{data.head()}')
    ship_data = torch.tensor(ship_data,dtype=torch.float64)
    L,W,Theta = ship_data[0:3,2],ship_data[0:3,3],ship_data[0:3,4]
    xn,yn = ship_data[0:3,0],ship_data[0:3,1]
    zn = H_data[0:3,3]
    m_inputs = ship_data[0:3,5:]
    print(f'L:{L}')
    print(f'W:{W}')
    print(f'Theta:{Theta}')
    print(f'xn:{xn}')
    print(f'yn:{yn}')
    print(f'zn:{zn}')
    print(f'm_inputs:{m_inputs}')
    print(f'xn.shape:{xn.shape}')
    print(f'yn.shape:{yn.shape}')
    print(f'zn.shape:{zn.shape}')
    print(f'm_inputs.shape:{m_inputs.shape}')
    # outputs = test_problem(xn,yn,zn,m_inputs,theta=Theta,L=L,W=W)
    outputs = forward_problem(xn,yn,zn,m_inputs,theta=Theta,Ln=L,Wn=W)
    # print("inputs.shape:",inputs.shape)
    
    # print(f"计算-传感器坐标系的磁场B_x * 1e+09: {outputs[:,0:1]}")
    print(f"计算-传感器坐标系的磁场B_x * 1e+09: {outputs[:,0:1]}")
    print(f"计算-传感器坐标系的磁场B_y * 1e+09: {outputs[:,1:2]}")
    print(f"计算-传感器坐标系的磁场B_z * 1e+09: {outputs[:,2:3]}")

    print(f"原数据-传感器坐标系的磁场B_x * 1e+09: {Hx[0:3]}")
    print(f"原数据-传感器坐标系的磁场B_y * 1e+09: {Hy[0:3]}")
    print(f"原数据-传感器坐标系的磁场B_z * 1e+09: {Hz[0:3]}")

    print(f'error:{((outputs[:,0:1]-H_data[0:3,0:1])**2).mean()}')
    # print(f"原归一化数据-传感器坐标系的磁场B_x * 1e+09: {H_data[0:3,0:1]}")
    # print(f"原归一化数据-传感器坐标系的磁场B_y * 1e+09: {H_data[0:3,1:2]}")
    # print(f"原归一化数据-传感器坐标系的磁场B_z * 1e+09: {H_data[0:3,2:3]}")
