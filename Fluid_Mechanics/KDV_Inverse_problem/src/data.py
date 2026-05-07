import numpy as np
import os
os.environ["DDE_BACKEND"] = "pytorch"   # 小写且要放最前
import deepxde as dde
# import numpy as np
a = -0.77
g = 9.81
h = 5.13

b = dde.Variable(0.5)
rho1 = dde.Variable(3.0)
rho2 = dde.Variable(0.5)

c_t = 0.5549706945768627
lamda_t = 3.1031120097882274

def kdv_eq_exact_solution(x, t):
    return a * (1.0 / np.cosh((x - c_t * t) / lamda_t)) ** 2

def gen_exact_solution():
    x_dim, t_dim = (256, 201)
    x_min, t_min = (-20, 0.0)
    x_max, t_max = (20, 4.0)

    t = np.linspace(t_min, t_max, num=t_dim).reshape(t_dim, 1)
    x = np.linspace(x_min, x_max, num=x_dim).reshape(x_dim, 1)
    usol = np.zeros((x_dim, t_dim)).reshape(x_dim, t_dim)

    for i in range(x_dim):
        for j in range(t_dim):
            usol[i][j] = kdv_eq_exact_solution(x[i], t[j])

    np.savez("heat_eq_data", x=x, t=t, usol=usol)

def gen_testdata():
    current_dir = os.path.dirname(__file__)

    # 定位到上一级目录的 data 文件夹
    data_path = os.path.join(current_dir, "..", "data", "heat_eq_data.npz")

    # 转换成绝对路径
    data_path = os.path.abspath(data_path)

    # 读取 npz 文件
    data = np.load(data_path)
    # data = np.load("heat_eq_data.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X, y