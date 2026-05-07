import math
import numpy as np

from .config import MASS, IX, IY, IZ, IXZ, S_REF, B_REF, C_REF, RHO, G, T_CONST
from .controls import control_inputs
from .aero_true import aero_true

def f16_dynamics(t: float, state: np.ndarray) -> np.ndarray:
    """F-16 类飞机 6DOF 动力学。

    state = [u, v, w, p, q, r, phi, theta, psi, xE, yE, zE]
    """
    u, v, w, p, q, r, phi, theta, psi, xE, yE, zE = state

    de, da, dr = control_inputs(t)

    V = math.sqrt(u*u + v*v + w*w) + 1e-8
    alpha = math.atan2(w, u)
    beta  = math.asin(v / V)

    p_hat = p * B_REF / (2.0 * V)
    q_hat = q * C_REF / (2.0 * V)
    r_hat = r * B_REF / (2.0 * V)

    CX, CY, CZ, Cl, Cm, Cn = aero_true(alpha, beta, p_hat, q_hat, r_hat, de, da, dr)

    qbar = 0.5 * RHO * V*V

    X = qbar * S_REF * CX + T_CONST
    Y = qbar * S_REF * CY
    Z = qbar * S_REF * CZ

    u_dot = r*v - q*w + X/MASS - G*math.sin(theta)
    v_dot = p*w - r*u + Y/MASS + G*math.cos(theta)*math.sin(phi)
    w_dot = q*u - p*v + Z/MASS + G*math.cos(theta)*math.cos(phi)

    L = qbar * S_REF * B_REF * Cl
    M = qbar * S_REF * C_REF * Cm
    N = qbar * S_REF * B_REF * Cn

    denom = IX*IZ - IXZ*IXZ

    p_dot = ( (IZ*L + IXZ*N) +
              (IXZ*(IX - IY + IZ)*p*q) -
              ((IZ*IZ + IXZ*IX)*q*r) ) / denom

    r_dot = ( (IX*N + IXZ*L) +
              (IXZ*(IX + IY - IZ)*q*r) -
              ((IX*IX + IXZ*IZ)*p*q) ) / denom

    q_dot = ( M - (IX - IZ)*p*r - IXZ*(p*p - r*r) ) / IY

    sin_phi = math.sin(phi); cos_phi = math.cos(phi)
    sin_th  = math.sin(theta); cos_th = math.cos(theta)
    tan_th  = math.tan(theta) if abs(cos_th) > 1e-6 else 0.0

    phi_dot   = p + tan_th*(q*sin_phi + r*cos_phi)
    theta_dot = q*cos_phi - r*sin_phi
    psi_dot   = (q*sin_phi + r*cos_phi)/max(cos_th,1e-6)

    xE_dot = (math.cos(theta)*math.cos(psi))*u + (math.cos(theta)*math.sin(psi))*v - (math.sin(theta))*w
    yE_dot = (math.sin(phi)*math.sin(theta)*math.cos(psi) - math.cos(phi)*math.sin(psi))*u +              (math.sin(phi)*math.sin(theta)*math.sin(psi) + math.cos(phi)*math.cos(psi))*v +              (math.sin(phi)*math.cos(theta))*w
    zE_dot = (math.cos(phi)*math.sin(theta)*math.cos(psi) + math.sin(phi)*math.sin(psi))*u +              (math.cos(phi)*math.sin(theta)*math.sin(psi) - math.sin(phi)*math.cos(phi))*v +              (math.cos(phi)*math.cos(theta))*w

    return np.array([u_dot, v_dot, w_dot, p_dot, q_dot, r_dot, phi_dot, theta_dot, psi_dot, xE_dot, yE_dot, zE_dot], dtype=float)

def rk4_step(fun, t: float, state: np.ndarray, dt: float) -> np.ndarray:
    """RK4 单步积分。"""
    k1 = fun(t, state)
    k2 = fun(t + 0.5*dt, state + 0.5*dt*k1)
    k3 = fun(t + 0.5*dt, state + 0.5*dt*k2)
    k4 = fun(t + dt, state + dt*k3)
    return state + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0
