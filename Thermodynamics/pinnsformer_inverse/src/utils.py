import numpy as np
import torch
import copy

def get_clones(module, N):
    """克隆 N 个相同的 ModuleList"""
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def get_n_params(model):
    """返回模型可训练参数总数"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def make_time_sequence(src_tensor, num_step, step):
    """
    将输入张量扩展为时间序列（用于时间步进）
    src_tensor: [N, 4] (x,y,z,t)
    num_step: 时间步数
    step: 时间步长
    返回: [N, num_step, 4] 其中每个样本的时间依次增加 step
    """
    src_expanded = src_tensor.unsqueeze(1).repeat(1, num_step, 1)
    time_offset = torch.arange(num_step, device=src_tensor.device, dtype=torch.float32) * step
    src_expanded[..., 3] += time_offset
    return src_expanded

def LHSample(D, bounds, N):
    """
    拉丁超立方采样
    :param D: 参数维度
    :param bounds: [[min1,max1], [min2,max2], ...]
    :param N: 样本数
    :return: [N, D] 样本数组
    """
    result = np.empty([N, D])
    temp = np.empty([N])
    d = 1.0 / N
    for i in range(D):
        for j in range(N):
            temp[j] = np.random.uniform(low=j * d, high=(j + 1) * d, size=1)[0]
        np.random.shuffle(temp)
        for j in range(N):
            result[j, i] = temp[j]
    # 缩放到实际范围
    b = np.array(bounds)
    lower_bounds = b[:, 0]
    upper_bounds = b[:, 1]
    if np.any(lower_bounds > upper_bounds):
        print('Wrong value bound')
        return None
    #   sample * (upper_bound - lower_bound) + lower_bound
    np.add(np.multiply(result, (upper_bounds - lower_bounds), out=result),
           lower_bounds,
           out=result)
    return result