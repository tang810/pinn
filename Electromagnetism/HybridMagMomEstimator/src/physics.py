import numpy as np
import torch


def magnetic_field_loss(predicted_magnetic_moments, inputs):
    m_count = 15
    length = 152.96
    width = 17.28
    mu_0 = 4 * np.pi * 1e-7
    eps = 1e-12

    x, y, z = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3]
    theta = inputs[:, 3:4]
    magnetic_moments = predicted_magnetic_moments.view(-1, m_count + 1, 3)

    delta_l = length / (m_count - 1)
    positions = torch.tensor(
        [[(i - (m_count - 1) / 2) * delta_l, 0.0, 0.0] for i in range(m_count)],
        dtype=torch.float32,
        device=inputs.device,
    )

    b_x_total = torch.zeros_like(x)
    b_y_total = torch.zeros_like(y)
    b_z_total = torch.zeros_like(z)

    for i in range(m_count):
        r_d = positions[i]
        r = torch.sqrt((x - r_d[0]) ** 2 + (y - r_d[1]) ** 2 + (z - r_d[2]) ** 2 + eps)

        a_x = (mu_0 / (4 * np.pi)) * (3 * (x - r_d[0]) ** 2 / r**5 - 1 / r**3)
        a_y = (3 * mu_0 / (4 * np.pi)) * ((x - r_d[0]) * (y - r_d[1]) / r**5)
        a_z = (3 * mu_0 / (4 * np.pi)) * ((x - r_d[0]) * (z - r_d[2]) / r**5)
        b_y = (mu_0 / (4 * np.pi)) * (3 * (y - r_d[1]) ** 2 / r**5 - 1 / r**3)
        b_z = (3 * mu_0 / (4 * np.pi)) * ((y - r_d[1]) * (z - r_d[2]) / r**5)
        c_z = (mu_0 / (4 * np.pi)) * (3 * (z - r_d[2]) ** 2 / r**5 - 1 / r**3)

        b_x, c_x, c_y = a_y, a_z, b_z
        m_x = magnetic_moments[:, i, 0].unsqueeze(1)
        m_y = magnetic_moments[:, i, 1].unsqueeze(1)
        m_z = magnetic_moments[:, i, 2].unsqueeze(1)

        b_x_total += a_x * m_x + a_y * m_y + a_z * m_z
        b_y_total += b_x * m_x + b_y * m_y + b_z * m_z
        b_z_total += c_x * m_x + c_y * m_y + c_z * m_z

    ell_mx = magnetic_moments[:, m_count, 0].unsqueeze(1)
    ell_my = magnetic_moments[:, m_count, 1].unsqueeze(1)
    ell_mz = magnetic_moments[:, m_count, 2].unsqueeze(1)
    length_t = torch.tensor(length, dtype=torch.float32, device=inputs.device)
    width_t = torch.tensor(width, dtype=torch.float32, device=inputs.device)

    r_norm = torch.sqrt(x**2 + y**2 + z**2 + eps)
    k = torch.sqrt((length_t / 2) ** 2 - (width_t / 2) ** 2 + eps)
    t = torch.sqrt((r_norm**2 + k**2) ** 2 - 4 * k**2 * x**2 + eps)
    a = torch.sqrt(0.5 * (r_norm**2 + k**2 + t) + eps)
    b = torch.sqrt(0.5 * (r_norm**2 - k**2 + t) + eps)

    log_term = torch.log(torch.clamp((a - k) / (a + k + eps), min=eps))
    a_x = (3 * mu_0 / (4 * np.pi)) * (a / (k**2 * t + eps) + (1 / (2 * k**3 + eps)) * log_term)
    a_y = (3 * mu_0 / (4 * np.pi)) * (x * y / (a * b**2 * t + eps))
    a_z = (3 * mu_0 / (4 * np.pi)) * (x * z / (a * b**2 * t + eps))
    b_y = (3 * mu_0 / (8 * np.pi)) * (
        (2 * a * y**2) / (b**4 * t + eps) - a / (b**2 * k**2 + eps) - (1 / (2 * k**3 + eps)) * log_term
    )
    b_z = (3 * mu_0 / (4 * np.pi)) * (a * y * z / (b**4 * t + eps))
    c_z = (3 * mu_0 / (8 * np.pi)) * (
        (2 * a * z**2) / (b**4 * t + eps) - a / (b**2 * k**2 + eps) - (1 / (2 * k**3 + eps)) * log_term
    )
    b_x, c_x, c_y = a_y, a_z, b_z

    b_x_total += a_x * ell_mx + a_y * ell_my + a_z * ell_mz
    b_y_total += b_x * ell_mx + b_y * ell_my + b_z * ell_mz
    b_z_total += c_x * ell_mx + c_y * ell_my + c_z * ell_mz

    b_x_prime = b_x_total * torch.cos(theta) - b_y_total * torch.sin(theta)
    b_y_prime = b_x_total * torch.sin(theta) + b_y_total * torch.cos(theta)
    b_z_prime = b_z_total

    loss = torch.mean(
        (b_x_prime - inputs[:, 4:5]) ** 2
        + (b_y_prime - inputs[:, 5:6]) ** 2
        + (b_z_prime - inputs[:, 6:7]) ** 2
    ) / 10**2

    return loss
