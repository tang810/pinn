import numpy as np
import torch


def generate_domain_points(num_domain, length, dpml, device):
    outer_min = -length / 2.0 - dpml
    outer_max = length / 2.0 + dpml
    points = torch.rand(num_domain, 2, device=device)
    points[:, 0] = (outer_max - outer_min) * points[:, 0] + outer_min
    points[:, 1] = (outer_max - outer_min) * points[:, 1] + outer_min
    points.requires_grad_(True)
    return points


def generate_inverse_points(num_domain, num_boundary, length, dpml, radius, device):
    outer_min = -length / 2.0 - dpml
    outer_max = length / 2.0 + dpml
    domain_points = []

    while len(domain_points) < num_domain:
        x = np.random.uniform(outer_min, outer_max)
        y = np.random.uniform(outer_min, outer_max)
        if np.hypot(x, y) > radius:
            domain_points.append([x, y])

    boundary_points = []
    while len(boundary_points) < num_boundary:
        side = np.random.choice(4)
        if side == 0:
            x = outer_min
            y = np.random.uniform(outer_min, outer_max)
        elif side == 1:
            x = outer_max
            y = np.random.uniform(outer_min, outer_max)
        elif side == 2:
            x = np.random.uniform(outer_min, outer_max)
            y = outer_min
        else:
            x = np.random.uniform(outer_min, outer_max)
            y = outer_max
        if np.hypot(x, y) > radius:
            boundary_points.append([x, y])

    domain_points = torch.tensor(domain_points, dtype=torch.float32, device=device)
    boundary_points = torch.tensor(boundary_points, dtype=torch.float32, device=device)
    domain_points.requires_grad_(True)
    boundary_points.requires_grad_(True)
    return domain_points, boundary_points


def create_visualization_grid(length, dpml, resolution, device):
    x = np.linspace(-length / 2.0 - dpml, length / 2.0 + dpml, resolution)
    y = np.linspace(-length / 2.0 - dpml, length / 2.0 + dpml, resolution)
    xx, yy = np.meshgrid(x, y)
    points = np.vstack([xx.ravel(), yy.ravel()]).T
    pts = torch.tensor(points, dtype=torch.float32, device=device)
    return pts, xx, yy
