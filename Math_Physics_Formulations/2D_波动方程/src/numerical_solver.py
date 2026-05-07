import numpy as np
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
from typing import Tuple


class WaveEquation:
    """
    2D Wave Equation Solver using Chebyshev Spectral Method
    """

    def __init__(self, N, T, x0=-1.0, xf=1.0, y0=-1.0, yf=1.0):
        """
        初始化求解器
        参数:
        - N: 网格数量 (Chebyshev 点数)
        - T: 模拟结束时间
        - x0, xf: x方向区间
        - y0, yf: y方向区间
        """
        self.N = N
        self.T = T
        self.x0 = x0
        self.xf = xf
        self.y0 = y0
        self.yf = yf

        self._initialization()
        self._init_condition()

    def _initialization(self):
        k = np.arange(self.N + 1)
        self.x = np.cos(k * np.pi / self.N)
        self.y = self.x.copy()
        self.xx, self.yy = np.meshgrid(self.x, self.y)

        self.dt = 6 / self.N ** 2
        self.plotgap = round((1 / 3) / self.dt)
        self.dt = (1 / 3) / self.plotgap

    def _init_condition(self):
        self.vv = np.exp(-40 * ((self.xx - 0.4) ** 2 + self.yy ** 2))
        self.vvold = self.vv.copy()

    def solve(self):
        """
        执行数值求解，返回每个时间步的插值网格解
        输出:
        - u_list: list of ndarray, shape = (time_steps, x_grid, y_grid)
        """
        u_list = []

        tc = 0
        nstep = round(self.T / self.dt) + 1

        # 创建目标插值网格
        xxx = np.arange(self.x0, self.xf + 1 / 16, 1 / 16)
        yyy = np.arange(self.y0, self.yf + 1 / 16, 1 / 16)
        xxf, yyf = np.meshgrid(xxx, yyy, indexing='ij')
        interp_points = np.stack([xxf.flatten(), yyf.flatten()], axis=-1)

        ii = np.arange(1, self.N)

        while tc < nstep:
            # 空间插值
            interp_func = RegularGridInterpolator((self.x, self.y), self.vv, method='linear')
            Z = interp_func(interp_points).reshape(xxf.shape)

            uxx = np.zeros((self.N + 1, self.N + 1))
            uyy = np.zeros((self.N + 1, self.N + 1))

            for i in range(1, self.N):
                v = self.vv[i, :]
                V = np.hstack((v, np.flipud(v[ii])))
                U = np.fft.fft(V).real

                r1 = np.arange(self.N)
                r2 = 1j * np.hstack((r1, 0, -r1[:0:-1])) * U
                W1 = np.fft.ifft(r2).real

                s1 = np.arange(self.N + 1)
                s2 = np.hstack((s1, -s1[self.N - 1:0:-1]))
                s3 = -s2 ** 2 * U
                W2 = np.fft.ifft(s3).real

                uxx[i, ii] = W2[ii] / (1 - self.x[ii] ** 2) - self.x[ii] * W1[ii] / (1 - self.x[ii] ** 2) ** (3 / 2)

            for j in range(1, self.N):
                v = self.vv[:, j]
                V = np.hstack((v, np.flipud(v[ii])))
                U = np.fft.fft(V).real

                r1 = np.arange(self.N)
                r2 = 1j * np.hstack((r1, 0, -r1[:0:-1])) * U
                W1 = np.fft.ifft(r2).real

                s1 = np.arange(self.N + 1)
                s2 = np.hstack((s1, -s1[self.N - 1:0:-1]))
                s3 = -s2 ** 2 * U
                W2 = np.fft.ifft(s3).real

                uyy[ii, j] = W2[ii] / (1 - self.y[ii] ** 2) - self.y[ii] * W1[ii] / (1 - self.y[ii] ** 2) ** (3 / 2)

            vvnew = 2 * self.vv - self.vvold + self.dt ** 2 * (uxx + uyy)
            self.vvold = self.vv.copy()
            self.vv = vvnew.copy()

            tc += 1
            u_list.append(Z)

        return np.asarray(u_list)

def save_wave_equation_to_csv(u_list, x, y, dt, save_path):
    t_values = np.arange(len(u_list)) * dt
    records = [
        [x[i], y[j], t, u[i, j]]
        for t_idx, t in enumerate(t_values)
        for i in range(len(x))
        for j in range(len(y))
        for u in [u_list[t_idx]]
    ]
    df = pd.DataFrame(records, columns=["x", "y", "t", "u"])
    df.to_csv(save_path, index=False)
    return save_path

def load_wave_equation_from_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    从CSV文件中加载二维波动方程模拟数据

    参数:
    - csv_path: 文件路径，必须包含列 ['x', 'y', 't', 'u']

    返回:
    - u_array: ndarray, shape = (T, X, Y)
    - x_vals: ndarray, shape = (X,)
    - y_vals: ndarray, shape = (Y,)
    - t_vals: ndarray, shape = (T,)
    """
    df = pd.read_csv(csv_path)

    # 验证字段完整
    required_cols = {"x", "y", "t", "u"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV 文件缺少必须的列: {required_cols - set(df.columns)}")

    # 去重排序
    x_vals = np.sort(df["x"].unique())
    y_vals = np.sort(df["y"].unique())
    t_vals = np.sort(df["t"].unique())

    nx, ny, nt = len(x_vals), len(y_vals), len(t_vals)
    u_array = np.zeros((nt, nx, ny))

    # 建立索引映射
    x_map = {v: i for i, v in enumerate(x_vals)}
    y_map = {v: i for i, v in enumerate(y_vals)}
    t_map = {v: i for i, v in enumerate(t_vals)}

    for _, row in df.iterrows():
        xi, yi, ti = x_map[row["x"]], y_map[row["y"]], t_map[row["t"]]
        u_array[ti, xi, yi] = row["u"]

    return u_array, x_vals, y_vals, t_vals
