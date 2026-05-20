"""
Notebook-friendly train file for pinn_6dof_f16.

Online Jupyter usage:
1. (Optional) Upload a pre-generated flight_dataset.npz as a public asset.
2. Fill DATA_HASH and MODEL_HASH.
3. Run this whole file/cell. It saves trained_model_f16_aero.pt to
   ~/minio/Resource/{MODEL_HASH}/ for later public-asset upload.

Physics: PINN identifying 6 aerodynamic coefficients (CX, CY, CZ, Cl, Cm, Cn)
         of an F-16-like aircraft from 6DOF flight data using physics residual loss.
"""

import math
import os
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = "069f4fc910334d9a8f22aad8b77c8eda"
MODEL_HASH = "069f4fc910334d9a8f22aad8b77c8eda"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/minio/Resource/{MODEL_HASH}/trained_model_f16_aero.pt")


# =========================
# 2. Physics constants (F-16-like)
# =========================

MASS = 9298.0; IX = 12874.0; IY = 75673.0; IZ = 85552.0; IXZ = 1331.0
S_REF = 27.87; B_REF = 9.14; C_REF = 3.45
RHO = 1.225; G = 9.81; T_CONST = 20000.0


# =========================
# 3. Aerodynamic true model (synthetic data generator)
# =========================

def aero_true(alpha, beta, p_hat, q_hat, r_hat, de, da, dr):
    CX = -0.1 - 0.3*alpha - 0.5*q_hat
    CY = -0.8*beta + 0.08*da + 0.105*dr - 0.03*p_hat + 0.21*r_hat
    CZ = -0.3 - 5.5*alpha - 8.0*q_hat - 0.9*de
    Cl = -0.12*beta + 0.16*da + 0.105*dr - 0.5*p_hat + 0.25*r_hat
    Cm = 0.05 - 1.5*alpha - 20.0*q_hat - 1.0*de
    Cn = 0.25*beta + 0.06*da - 0.2*dr - 0.02*p_hat - 0.3*r_hat
    return CX, CY, CZ, Cl, Cm, Cn


# =========================
# 4. Control inputs and 6DOF dynamics
# =========================

def multi_sine(t, amps, freqs, phases):
    y = 0.0
    for A, f, p in zip(amps, freqs, phases):
        y += A * math.sin(2.0*math.pi*f*t + p)
    return y


def control_inputs(t):
    de = multi_sine(t, [10.0*math.pi/180, 5.0*math.pi/180], [0.2, 0.7], [0.0, 1.0])
    da = multi_sine(t, [8.0*math.pi/180, 4.0*math.pi/180], [0.15, 0.55], [0.5, 2.0])
    dr = multi_sine(t, [6.0*math.pi/180, 3.0*math.pi/180], [0.25, 0.9], [1.0, 0.3])
    return de, da, dr


def f16_dynamics(t, state):
    u, v, w, p, q, r, phi, theta, psi, xE, yE, zE = state
    de, da, dr = control_inputs(t)
    V = math.sqrt(u*u + v*v + w*w) + 1e-8
    alpha = math.atan2(w, u)
    beta = math.asin(max(-1.0, min(1.0, v/V)))
    p_hat = p*B_REF/(2.0*V); q_hat = q*C_REF/(2.0*V); r_hat = r*B_REF/(2.0*V)
    CX, CY, CZ, Cl, Cm, Cn = aero_true(alpha, beta, p_hat, q_hat, r_hat, de, da, dr)
    qbar = 0.5*RHO*V*V
    X = qbar*S_REF*CX + T_CONST; Y = qbar*S_REF*CY; Z = qbar*S_REF*CZ
    u_dot = r*v - q*w + X/MASS - G*math.sin(theta)
    v_dot = p*w - r*u + Y/MASS + G*math.cos(theta)*math.sin(phi)
    w_dot = q*u - p*v + Z/MASS + G*math.cos(theta)*math.cos(phi)
    L = qbar*S_REF*B_REF*Cl; M_ = qbar*S_REF*C_REF*Cm; N = qbar*S_REF*B_REF*Cn
    denom = IX*IZ - IXZ*IXZ
    p_dot = ((IZ*L+IXZ*N)+(IXZ*(IX-IY+IZ)*p*q)-((IZ*IZ+IXZ*IX)*q*r)) / denom
    r_dot = ((IX*N+IXZ*L)+(IXZ*(IX+IY-IZ)*q*r)-((IX*IX+IXZ*IZ)*p*q)) / denom
    q_dot = (M_-(IX-IZ)*p*r-IXZ*(p*p-r*r)) / IY
    sin_ph=math.sin(phi); cos_ph=math.cos(phi); sin_th=math.sin(theta); cos_th=math.cos(theta)
    tan_th = math.tan(theta) if abs(cos_th)>1e-6 else 0.0
    phi_dot = p + tan_th*(q*sin_ph+r*cos_ph)
    theta_dot = q*cos_ph - r*sin_ph
    psi_dot = (q*sin_ph+r*cos_ph)/max(cos_th,1e-6)
    xE_dot = cos_th*math.cos(psi)*u + cos_th*math.sin(psi)*v - sin_th*w
    yE_dot = (sin_ph*sin_th*math.cos(psi)-cos_ph*math.sin(psi))*u + (sin_ph*sin_th*math.sin(psi)+cos_ph*math.cos(psi))*v + sin_ph*cos_th*w
    zE_dot = (cos_ph*sin_th*math.cos(psi)+sin_ph*math.sin(psi))*u + (cos_ph*sin_th*math.sin(psi)-sin_ph*cos_ph)*v + cos_ph*cos_th*w
    return np.array([u_dot,v_dot,w_dot,p_dot,q_dot,r_dot,phi_dot,theta_dot,psi_dot,xE_dot,yE_dot,zE_dot])


def rk4_step(fun, t, state, dt):
    k1 = fun(t, state); k2 = fun(t+0.5*dt, state+0.5*dt*k1)
    k3 = fun(t+0.5*dt, state+0.5*dt*k2); k4 = fun(t+dt, state+dt*k3)
    return state + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0


# =========================
# 5. Flight data generation
# =========================

def generate_flight(t_end=20.0, dt=0.02, noise_level=0.02, seed=0):
    N = int(t_end/dt) + 1; t_vec = np.linspace(0.0, t_end, N)
    V0=150.0; alpha0=2.0*math.pi/180; u0=V0*math.cos(alpha0); w0=V0*math.sin(alpha0)
    state = np.array([u0,0.0,w0,0.0,0.0,0.0,0.0,alpha0,0.0,0.0,0.0,-1000.0])
    states = np.zeros((N,12)); derivs = np.zeros((N,12)); controls = np.zeros((N,3))
    for i, tt in enumerate(t_vec):
        states[i] = state; derivs[i] = f16_dynamics(tt, state)
        de, da, dr = control_inputs(tt); controls[i] = [de, da, dr]
        state = rk4_step(f16_dynamics, tt, state, dt)
    acc_body = derivs[:, 0:6]
    rng = np.random.default_rng(seed)
    states_n = states.copy()
    states_n[:, 0:6] += rng.normal(0, noise_level, size=states_n[:, 0:6].shape)
    acc_n = acc_body + rng.normal(0, noise_level, size=acc_body.shape)
    return t_vec, states_n, controls, acc_n


def build_nn_inputs(t_vec, states, controls):
    u=states[:,0]; v=states[:,1]; w=states[:,2]; p=states[:,3]; q=states[:,4]; r=states[:,5]
    de=controls[:,0]; da=controls[:,1]; dr=controls[:,2]
    V = np.sqrt(u*u+v*v+w*w) + 1e-6
    alpha = np.arctan2(w, u); beta = np.arcsin(np.clip(v/V, -1, 1))
    p_hat = p*B_REF/(2*V); q_hat = q*C_REF/(2*V); r_hat = r*B_REF/(2*V)
    V_norm = V/200.0
    X_in = np.stack([alpha, beta, p_hat, q_hat, r_hat, de, da, dr, V_norm], axis=1)
    extras = {"u":u,"v":v,"w":w,"p":p,"q":q,"r":r,"phi":states[:,6],"theta":states[:,7],
              "alpha":alpha,"beta":beta,"p_hat":p_hat,"q_hat":q_hat,"r_hat":r_hat,
              "de":de,"da":da,"dr":dr,"V":V}
    return X_in, extras


# =========================
# 6. Model definition
# =========================

class AeroNet(nn.Module):
    def __init__(self, in_dim=9, hidden=64, layers=4):
        super().__init__()
        seq = [nn.Linear(in_dim, hidden), nn.Softplus()]
        for _ in range(layers - 1):
            seq += [nn.Linear(hidden, hidden), nn.Softplus()]
        seq += [nn.Linear(hidden, 6)]
        self.net = nn.Sequential(*seq)

    def forward(self, x):
        return self.net(x)


# =========================
# 7. Physics residual loss
# =========================

def physics_residual_loss(net, X_t, u_t, v_t, w_t, p_t, q_t, r_t, phi_t, theta_t,
                           u_dot_t, v_dot_t, w_dot_t, p_dot_t, q_dot_t, r_dot_t, reg_weight=1e-5):
    coeffs = net(X_t)
    CX,CY,CZ,Cl,Cm,Cn = (coeffs[:,0],coeffs[:,1],coeffs[:,2],coeffs[:,3],coeffs[:,4],coeffs[:,5])
    V_mag = torch.sqrt(u_t*u_t+v_t*v_t+w_t*w_t) + 1e-6
    qbar = 0.5*RHO*V_mag*V_mag
    X = qbar*S_REF*CX + T_CONST; Y = qbar*S_REF*CY; Z = qbar*S_REF*CZ
    sin_th=torch.sin(theta_t); cos_th=torch.cos(theta_t)
    sin_ph=torch.sin(phi_t); cos_ph=torch.cos(phi_t)
    u_dot_p = r_t*v_t - q_t*w_t + X/MASS - G*sin_th
    v_dot_p = p_t*w_t - r_t*u_t + Y/MASS + G*cos_th*sin_ph
    w_dot_p = q_t*u_t - p_t*v_t + Z/MASS + G*cos_th*cos_ph
    L = qbar*S_REF*B_REF*Cl; M_ = qbar*S_REF*C_REF*Cm; N = qbar*S_REF*B_REF*Cn
    denom = IX*IZ - IXZ*IXZ
    p_dot_p = ((IZ*L+IXZ*N)+(IXZ*(IX-IY+IZ)*p_t*q_t)-((IZ*IZ+IXZ*IX)*q_t*r_t))/denom
    r_dot_p = ((IX*N+IXZ*L)+(IXZ*(IX+IY-IZ)*q_t*r_t)-((IX*IX+IXZ*IZ)*p_t*q_t))/denom
    q_dot_p = (M_-(IX-IZ)*p_t*r_t-IXZ*(p_t*p_t-r_t*r_t))/IY
    ru=(u_dot_p-u_dot_t); rv=(v_dot_p-v_dot_t); rw=(w_dot_p-w_dot_t)
    rp=(p_dot_p-p_dot_t); rq=(q_dot_p-q_dot_t); rr=(r_dot_p-r_dot_t)
    loss_trans = (ru*ru+rv*rv+rw*rw).mean()
    loss_rot = (rp*rp+rq*rq+rr*rr).mean()
    reg = reg_weight*(CX.pow(2).mean()+CY.pow(2).mean()+CZ.pow(2).mean()
                      +Cl.pow(2).mean()+Cm.pow(2).mean()+Cn.pow(2).mean())
    return loss_trans+loss_rot+reg, loss_trans.detach(), loss_rot.detach()


# =========================
# 8. Visualization
# =========================

def show_train_result(history):
    if not len(history):
        return
    plt.figure(figsize=(7, 4))
    plt.plot(history, linewidth=1); plt.yscale("log")
    plt.xlabel("Epoch"); plt.ylabel("Total Loss")
    plt.title("Training Loss (Physics Residuals)"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.show()


# =========================
# 9. Train entry
# =========================

def train(data_path=DATA_PATH, model_path=MODEL_PATH, t_end=20.0, dt=0.02,
          noise=0.02, seed=0, epochs=5000, lr=1e-3, verbose_every=500):
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    t_vec, states, controls, acc = generate_flight(t_end=t_end, dt=dt, noise_level=noise, seed=seed)
    X_in, extras = build_nn_inputs(t_vec, states, controls)
    print(f"Data: {len(t_vec)} points, t_end={t_end}s")

    X_t = torch.tensor(X_in, dtype=torch.float32, device=device)
    st = states; ac = acc
    tensors = {
        "u": torch.tensor(st[:,0], dtype=torch.float32, device=device),
        "v": torch.tensor(st[:,1], dtype=torch.float32, device=device),
        "w": torch.tensor(st[:,2], dtype=torch.float32, device=device),
        "p": torch.tensor(st[:,3], dtype=torch.float32, device=device),
        "q": torch.tensor(st[:,4], dtype=torch.float32, device=device),
        "r": torch.tensor(st[:,5], dtype=torch.float32, device=device),
        "phi": torch.tensor(st[:,6], dtype=torch.float32, device=device),
        "theta": torch.tensor(st[:,7], dtype=torch.float32, device=device),
        "u_dot": torch.tensor(ac[:,0], dtype=torch.float32, device=device),
        "v_dot": torch.tensor(ac[:,1], dtype=torch.float32, device=device),
        "w_dot": torch.tensor(ac[:,2], dtype=torch.float32, device=device),
        "p_dot": torch.tensor(ac[:,3], dtype=torch.float32, device=device),
        "q_dot": torch.tensor(ac[:,4], dtype=torch.float32, device=device),
        "r_dot": torch.tensor(ac[:,5], dtype=torch.float32, device=device),
    }

    net = AeroNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    history = []

    print(f"Training {epochs} epochs...")
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        loss, lt, lr_ = physics_residual_loss(
            net, X_t,
            tensors["u"], tensors["v"], tensors["w"],
            tensors["p"], tensors["q"], tensors["r"],
            tensors["phi"], tensors["theta"],
            tensors["u_dot"], tensors["v_dot"], tensors["w_dot"],
            tensors["p_dot"], tensors["q_dot"], tensors["r_dot"])
        loss.backward(); opt.step()
        history.append(float(loss.detach().cpu()))
        if ep == 1 or ep % verbose_every == 0 or ep == epochs:
            print(f"Epoch {ep:5d}/{epochs} | total={history[-1]:.3e} trans={float(lt):.3e} rot={float(lr_):.3e}")

    model_path = os.path.expanduser(model_path)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    checkpoint = {
        "project": "pinn_6dof_f16",
        "state_dict": net.state_dict(),
        "history": history,
        "in_dim": 9, "hidden": 64, "layers": 4,
    }
    torch.save(checkpoint, model_path)

    print("Training finished")
    print(f"model saved to: {model_path}")
    print(f"elapsed: {time.time() - t0:.1f}s")
    print(f"final loss: {history[-1]:.6e}")

    show_train_result(history)
    return {"model": net, "model_path": model_path, "history": history}


train_result = train()
