import scipy.io as sio
import torch


def _build_simulated_dataset(device):
    n = 96
    x = torch.linspace(-80.0, 80.0, n, device=device).view(-1, 1)
    y = 20.0 * torch.sin(x / 30.0)
    z = 10.0 + 2.0 * torch.cos(x / 25.0)
    theta_values = torch.full_like(x, 0.05)

    bx = 1e-4 * torch.sin(x / 20.0)
    by = 1e-4 * torch.cos(x / 23.0)
    bz = 1e-4 * torch.sin(x / 17.0 + 0.3)

    inputs_xyz = torch.cat([x, y, z], dim=1)
    target_b = torch.cat([bx, by, bz], dim=1)
    model_inputs = torch.cat([inputs_xyz, theta_values, target_b], dim=1)
    gt_moments = torch.zeros((1, 48), dtype=torch.float32, device=device)

    return {
        "inputs_xyz": inputs_xyz,
        "theta": theta_values,
        "target_b": target_b,
        "model_inputs": model_inputs,
        "gt_moments": gt_moments,
    }


def load_dataset(mat_path, device):
    if mat_path == "simul":
        return _build_simulated_dataset(device)

    mat_data = sio.loadmat(mat_path)
    matrix = mat_data["Data"]
    measure = matrix[0, 0]["measurepoint"]
    data_m = matrix[0, 0]["M"]
    mag_s0 = matrix[0, 0]["MagS0"]
    theta = matrix[0, 0]["theta"]

    x = torch.tensor(measure[:, 0], dtype=torch.float32, device=device).view(-1, 1)
    y = torch.tensor(measure[:, 1], dtype=torch.float32, device=device).view(-1, 1)
    z = torch.tensor(measure[:, 2], dtype=torch.float32, device=device).view(-1, 1)

    theta_values = torch.full_like(x, float(theta))
    bx = torch.tensor(mag_s0[0, :], dtype=torch.float32, device=device).view(-1, 1)
    by = torch.tensor(mag_s0[1, :], dtype=torch.float32, device=device).view(-1, 1)
    bz = torch.tensor(mag_s0[2, :], dtype=torch.float32, device=device).view(-1, 1)

    inputs_xyz = torch.cat([x, y, z], dim=1)
    target_b = torch.cat([bx, by, bz], dim=1)
    model_inputs = torch.cat([inputs_xyz, theta_values, target_b], dim=1)

    # Ground truth moments from first column (same as original script).
    rod_moments = torch.tensor(data_m[:45, 0], dtype=torch.float32, device=device).view(1, -1)
    ellipsoid_moments = torch.tensor(data_m[45:48, 0], dtype=torch.float32, device=device).view(1, -1)
    gt_moments = torch.cat([rod_moments, ellipsoid_moments], dim=1)

    return {
        "inputs_xyz": inputs_xyz,
        "theta": theta_values,
        "target_b": target_b,
        "model_inputs": model_inputs,
        "gt_moments": gt_moments,
    }
