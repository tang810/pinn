import os

import numpy as np
import torch


def generate_points(num_points: int, device: torch.device) -> torch.Tensor:
    x = torch.linspace(-1.0, 1.0, num_points, device=device)
    y = torch.linspace(-1.0, 1.0, num_points, device=device)
    try:
        xx, yy = torch.meshgrid(x, y, indexing="ij")
    except TypeError:
        xx, yy = torch.meshgrid(x, y)
    return torch.stack([xx.flatten(), yy.flatten()], dim=1)


def generate_points_from_source(
    data_source: str,
    num_points: int,
    device: torch.device,
    data_path: str = "",
) -> torch.Tensor:
    if data_source == "simul":
        return generate_points(num_points, device)

    if data_path and os.path.exists(data_path):
        arr = np.load(data_path)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("real data must be an N x 2 numpy array.")
        return torch.tensor(arr, dtype=torch.float32, device=device)

    print(f"[warn] real data not found at {data_path}. Falling back to simulated grid.")
    return generate_points(num_points, device)


def boundary_condition(points: torch.Tensor) -> torch.Tensor:
    return torch.cos(points[:, 0]).unsqueeze(1)


def exact_solution(points: torch.Tensor) -> torch.Tensor:
    return torch.cos(points[:, 0]).unsqueeze(1)


def select_boundary_points(points: torch.Tensor) -> torch.Tensor:
    mask = (
        (points[:, 0] == -1)
        | (points[:, 0] == 1)
        | (points[:, 1] == -1)
        | (points[:, 1] == 1)
    )
    return points[mask]
