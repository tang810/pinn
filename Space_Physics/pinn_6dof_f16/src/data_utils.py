import math
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

from .dynamics import f16_dynamics, rk4_step
from .controls import control_inputs
from .config import B_REF, C_REF

@dataclass
class FlightDataset:
    t: np.ndarray                  # [N]
    states_meas: np.ndarray         # [N,12]
    controls: np.ndarray            # [N,3]
    acc_meas: np.ndarray            # [N,6]  u_dot..r_dot (meas)

def generate_flight(t_end: float = 20.0, dt: float = 0.02, noise_level: float = 0.02,
                    seed: int = 0) -> FlightDataset:
    """模拟一段 6DOF 飞行，并对状态/加速度加入高斯噪声。"""
    N = int(t_end/dt) + 1
    t_vec = np.linspace(0.0, t_end, N)

    V0 = 150.0
    alpha0 = 2.0 * math.pi/180.0
    u0 = V0*math.cos(alpha0)
    w0 = V0*math.sin(alpha0)
    v0 = 0.0

    p0 = q0 = r0 = 0.0
    phi0 = 0.0
    theta0 = alpha0
    psi0 = 0.0
    xE0 = yE0 = 0.0
    zE0 = -1000.0

    state = np.array([u0, v0, w0, p0, q0, r0, phi0, theta0, psi0, xE0, yE0, zE0], dtype=float)

    states = np.zeros((N, 12), dtype=float)
    derivs = np.zeros((N, 12), dtype=float)
    controls = np.zeros((N, 3), dtype=float)

    for i, tt in enumerate(t_vec):
        states[i] = state
        deriv = f16_dynamics(tt, state)
        derivs[i] = deriv
        de, da, dr = control_inputs(tt)
        controls[i] = [de, da, dr]
        state = rk4_step(f16_dynamics, tt, state, dt)

    acc_body = derivs[:, 0:6]

    rng = np.random.default_rng(seed)
    states_noisy = states.copy()
    states_noisy[:, 0:6] += rng.normal(0.0, noise_level, size=states_noisy[:,0:6].shape)
    acc_noisy = acc_body + rng.normal(0.0, noise_level, size=acc_body.shape)

    return FlightDataset(t=t_vec, states_meas=states_noisy, controls=controls, acc_meas=acc_noisy)

def save_dataset_npz(ds: FlightDataset, path: str) -> None:
    np.savez_compressed(path, t=ds.t, states_meas=ds.states_meas, controls=ds.controls, acc_meas=ds.acc_meas)

def load_dataset_npz(path: str) -> FlightDataset:
    d = np.load(path)
    return FlightDataset(t=d["t"], states_meas=d["states_meas"], controls=d["controls"], acc_meas=d["acc_meas"])

def build_nn_inputs(ds: FlightDataset, v_norm_div: float = 200.0) -> Tuple[np.ndarray, dict]:
    """从测量状态构造网络输入 X_in，并返回额外的派生变量字典。"""
    states = ds.states_meas
    u = states[:,0]; v = states[:,1]; w = states[:,2]
    p = states[:,3]; q = states[:,4]; r = states[:,5]

    de = ds.controls[:,0]; da = ds.controls[:,1]; dr = ds.controls[:,2]

    V = np.sqrt(u*u + v*v + w*w) + 1e-6
    alpha = np.arctan2(w, u)
    beta  = np.arcsin(np.clip(v/V, -1.0, 1.0))

    p_hat = p * B_REF / (2.0 * V)
    q_hat = q * C_REF / (2.0 * V)
    r_hat = r * B_REF / (2.0 * V)

    V_norm = V / v_norm_div

    X_in = np.stack([alpha, beta, p_hat, q_hat, r_hat, de, da, dr, V_norm], axis=1)

    extras = {
        "u": u, "v": v, "w": w,
        "p": p, "q": q, "r": r,
        "phi": states[:,6], "theta": states[:,7], "psi": states[:,8],
        "alpha": alpha, "beta": beta, "p_hat": p_hat, "q_hat": q_hat, "r_hat": r_hat,
        "de": de, "da": da, "dr": dr, "V": V
    }
    return X_in, extras
