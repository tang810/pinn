from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import torch

from .config import MASS, IX, IY, IZ, IXZ, S_REF, B_REF, C_REF, RHO, G, T_CONST

@dataclass
class PhysicsLossOutput:
    total: torch.Tensor
    loss_trans: torch.Tensor
    loss_rot: torch.Tensor

def physics_residual_loss(net, X_torch: torch.Tensor,
                          u_t: torch.Tensor, v_t: torch.Tensor, w_t: torch.Tensor,
                          p_t: torch.Tensor, q_t: torch.Tensor, r_t: torch.Tensor,
                          phi_t: torch.Tensor, theta_t: torch.Tensor,
                          u_dot_t: torch.Tensor, v_dot_t: torch.Tensor, w_dot_t: torch.Tensor,
                          p_dot_t: torch.Tensor, q_dot_t: torch.Tensor, r_dot_t: torch.Tensor,
                          reg_weight: float = 1e-5) -> PhysicsLossOutput:
    """Modified Non-Determinant PINN 物理残差损失。"""
    coeffs = net(X_torch)
    CX, CY, CZ, Cl, Cm, Cn = (coeffs[:,0], coeffs[:,1], coeffs[:,2], coeffs[:,3], coeffs[:,4], coeffs[:,5])

    V_mag = torch.sqrt(u_t*u_t + v_t*v_t + w_t*w_t) + 1e-6
    qbar  = 0.5 * RHO * V_mag*V_mag

    X = qbar * S_REF * CX + T_CONST
    Y = qbar * S_REF * CY
    Z = qbar * S_REF * CZ

    sin_th  = torch.sin(theta_t)
    cos_th  = torch.cos(theta_t)
    sin_phi = torch.sin(phi_t)
    cos_phi = torch.cos(phi_t)

    u_dot_pred = r_t*v_t - q_t*w_t + X/MASS - G*sin_th
    v_dot_pred = p_t*w_t - r_t*u_t + Y/MASS + G*cos_th*sin_phi
    w_dot_pred = q_t*u_t - p_t*v_t + Z/MASS + G*cos_th*cos_phi

    L = qbar * S_REF * B_REF * Cl
    M = qbar * S_REF * C_REF * Cm
    N = qbar * S_REF * B_REF * Cn

    denom = IX*IZ - IXZ*IXZ

    p_dot_pred = ((IZ*L + IXZ*N) + (IXZ*(IX - IY + IZ)*p_t*q_t) - ((IZ*IZ + IXZ*IX)*q_t*r_t)) / denom
    r_dot_pred = ((IX*N + IXZ*L) + (IXZ*(IX + IY - IZ)*q_t*r_t) - ((IX*IX + IXZ*IZ)*p_t*q_t)) / denom
    q_dot_pred = (M - (IX - IZ)*p_t*r_t - IXZ*(p_t*p_t - r_t*r_t)) / IY

    ru = u_dot_pred - u_dot_t
    rv = v_dot_pred - v_dot_t
    rw = w_dot_pred - w_dot_t
    rp = p_dot_pred - p_dot_t
    rq = q_dot_pred - q_dot_t
    rr = r_dot_pred - r_dot_t

    loss_trans = (ru*ru + rv*rv + rw*rw).mean()
    loss_rot   = (rp*rp + rq*rq + rr*rr).mean()

    reg = reg_weight * (CX.pow(2).mean() + CY.pow(2).mean() + CZ.pow(2).mean()
                        + Cl.pow(2).mean() + Cm.pow(2).mean() + Cn.pow(2).mean())

    total = loss_trans + loss_rot + reg
    return PhysicsLossOutput(total=total, loss_trans=loss_trans, loss_rot=loss_rot)
