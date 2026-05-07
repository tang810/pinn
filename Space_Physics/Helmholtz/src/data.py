import torch


def generate_1d_points(n_domain=1024):
    x_domain = (torch.rand(n_domain, 1) * 2.0 - 1.0).float()
    x_boundary = torch.tensor([[-1.0], [1.0]], dtype=torch.float32)
    u_boundary = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    return x_domain, x_boundary, u_boundary
