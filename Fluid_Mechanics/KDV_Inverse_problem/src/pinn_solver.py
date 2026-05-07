
import os
os.environ["DDE_BACKEND"] = "pytorch"   # 小写且要放最前
import deepxde as dde
import numpy as np
from .data import b, rho1, rho2, h, g, a, lamda_t

def pde(x, y):
    h1 = h * b
    h2 = h * (1 - b)
    c0 = (g * h1 * h2 * (rho1 - rho2) / (rho1 * h2 + rho2 * h1)) ** (1/2)
    c1 = -(3 / 2) * c0 * ((rho2 * h1**2 - rho1 * h2**2) / (rho2 * h2 * h1**2 + rho1 * h1 * h2**2))
    c2 = (1 / 6) * c0 * ((rho2 * h2**2 * h1 + rho1 * h1**2 * h2) / (rho2 * h1 + rho1 * h2))
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_xxx = dde.grad.jacobian(dy_xx, x, i=0, j=0)
    return dy_t + c0 * dy_x + c1 * y * dy_x + c2 * dy_xxx

def get_bc_ic(geomtime, observe_x, y):
    bc = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary)
    ic = dde.icbc.IC(geomtime, lambda x: a * (1.0 / np.cosh(x[:, 0:1] / lamda_t)) ** 2, lambda _, on_initial: on_initial)
    observe_y1 = dde.icbc.PointSetBC(observe_x, y)
    return [bc, ic, observe_y1]
def true_solution(x, t):
    # x, t 是 1D np.ndarray
    # 这里写你的解析解，例如：
    return np.sin(np.pi * x) * np.exp(-t)
    # 如果没有解析解，返回 None
