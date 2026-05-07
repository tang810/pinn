import torch
import numpy as np

# -----------------------------
# Physical configuration
# -----------------------------
frequencies = [3.0, 4.0]
angles = [0.0, np.pi / 3, 2 * np.pi / 3]

L = 1.0
BETA = 60.0

TRUE_EPS = 1.5
TRUE_R = 0.15


# -----------------------------
# Incident field
# -----------------------------
def Ez_inc(xy, theta, k):
    if not torch.is_tensor(theta):
        theta = torch.tensor(theta, device=xy.device)

    phase = k * (
        xy[:, 0:1] * torch.cos(theta) +
        xy[:, 1:2] * torch.sin(theta)
    )
    return torch.cos(phase), torch.sin(phase)


# -----------------------------
# Contrast χ(x)
# -----------------------------
def contrast_chi(xy, eps, r):
    rr = torch.sqrt(xy[:, 0:1] ** 2 + xy[:, 1:2] ** 2)
    mask = torch.sigmoid(BETA * (r - rr))
    return (eps - 1.0) * mask


# -----------------------------
# 2D Green function (far-field stabilized)
# -----------------------------
def green_2d(x_obs, x_src, k):
    diff = x_obs[:, None, :] - x_src[None, :, :]
    r = torch.norm(diff, dim=2) + 1e-6

    amp = 1.0 / torch.sqrt(8 * np.pi * k * r)
    phase = k * r - np.pi / 4

    return amp * torch.cos(phase), amp * torch.sin(phase)


# -----------------------------
# Lippmann–Schwinger (self-consistent single-iteration)
# -----------------------------
def lippmann_schwinger_sc(x_obs, x_src, eps, r, k, theta):
    Ei_re, Ei_im = Ez_inc(x_src, theta, k)
    chi = contrast_chi(x_src, eps, r)

    G_re, G_im = green_2d(x_obs, x_src, k)

    Ei_re, Ei_im, chi = Ei_re.T, Ei_im.T, chi.T

    # First Born approximation
    Es1_re = k**2 * torch.mean(
        G_re * chi * Ei_re - G_im * chi * Ei_im, dim=1, keepdim=True
    )
    Es1_im = k**2 * torch.mean(
        G_re * chi * Ei_im + G_im * chi * Ei_re, dim=1, keepdim=True
    )

    # Self-consistent correction
    Et_re = Ei_re + Es1_re
    Et_im = Ei_im + Es1_im

    Es_re = k**2 * torch.mean(
        G_re * chi * Et_re - G_im * chi * Et_im, dim=1, keepdim=True
    )
    Es_im = k**2 * torch.mean(
        G_re * chi * Et_im + G_im * chi * Et_re, dim=1, keepdim=True
    )

    return Es_re, Es_im