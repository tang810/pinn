import torch
import numpy as np
from src.physics import lippmann_schwinger_sc, frequencies, angles


def generate_observations(N_obs, x_src, eps, r, device):
    """
    生成观测数据（正向计算）
    """
    xy_obs_list, Eobs_re_list, Eobs_im_list = [], [], []

    for k in frequencies:
        for theta in angles:
            t = torch.linspace(0, 2*np.pi, N_obs)
            xy_obs = torch.stack([
                0.7 * torch.cos(t),
                0.7 * torch.sin(t)
            ], dim=1).to(device)

            # 使用Lippmann-Schwinger方程进行正向计算
            E_re, E_im = lippmann_schwinger_sc(
                xy_obs, x_src, eps, r, k, theta
            )

            xy_obs_list.append(xy_obs)
            Eobs_re_list.append(E_re.detach())
            Eobs_im_list.append(E_im.detach())

    return xy_obs_list, Eobs_re_list, Eobs_im_list