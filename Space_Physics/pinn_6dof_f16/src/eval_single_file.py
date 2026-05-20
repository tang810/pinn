"""
Notebook-friendly eval file for pinn_6dof_f16.

Online Jupyter usage:
1. Upload the trained .pt model as a public asset.
2. Fill MODEL_HASH and DATA_HASH, then run this whole file/cell.

This file loads the model, generates flight data, evaluates the 6 aerodynamic
coefficients against ground truth, and shows figures with plt.show().
It does not save images.
"""

import math
import os

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = "069f4fc910334d9a8f22aad8b77c8eda"
MODEL_HASH = "921a01c792a34c57a86a47c2194f1600"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/trained_model_f16_aero.pt")


# =========================
# 2. Physics constants (F-16-like)
# =========================

MASS = 9298.0; IX = 12874.0; IY = 75673.0; IZ = 85552.0; IXZ = 1331.0
S_REF = 27.87; B_REF = 9.14; C_REF = 3.45
RHO = 1.225; G = 9.81; T_CONST = 20000.0


# =========================
# 3. Aerodynamic true model
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
    V = math.sqrt(u*u+v*v+w*w) + 1e-8
    alpha = math.atan2(w, u); beta = math.asin(max(-1.0, min(1.0, v/V)))
    p_hat=p*B_REF/(2*V); q_hat=q*C_REF/(2*V); r_hat=r*B_REF/(2*V)
    CX,CY,CZ,Cl,Cm,Cn = aero_true(alpha,beta,p_hat,q_hat,r_hat,de,da,dr)
    qbar = 0.5*RHO*V*V
    X=qbar*S_REF*CX+T_CONST; Y=qbar*S_REF*CY; Z=qbar*S_REF*CZ
    u_dot=r*v-q*w+X/MASS-G*math.sin(theta)
    v_dot=p*w-r*u+Y/MASS+G*math.cos(theta)*math.sin(phi)
    w_dot=q*u-p*v+Z/MASS+G*math.cos(theta)*math.cos(phi)
    L=qbar*S_REF*B_REF*Cl; M_=qbar*S_REF*C_REF*Cm; N=qbar*S_REF*B_REF*Cn
    denom=IX*IZ-IXZ*IXZ
    p_dot=((IZ*L+IXZ*N)+(IXZ*(IX-IY+IZ)*p*q)-((IZ*IZ+IXZ*IX)*q*r))/denom
    r_dot=((IX*N+IXZ*L)+(IXZ*(IX+IY-IZ)*q*r)-((IX*IX+IXZ*IZ)*p*q))/denom
    q_dot=(M_-(IX-IZ)*p*r-IXZ*(p*p-r*r))/IY
    sin_ph=math.sin(phi);cos_ph=math.cos(phi);sin_th=math.sin(theta);cos_th=math.cos(theta)
    tan_th=math.tan(theta) if abs(cos_th)>1e-6 else 0.0
    phi_dot=p+tan_th*(q*sin_ph+r*cos_ph)
    theta_dot=q*cos_ph-r*sin_ph; psi_dot=(q*sin_ph+r*cos_ph)/max(cos_th,1e-6)
    xE_dot=cos_th*math.cos(psi)*u+cos_th*math.sin(psi)*v-sin_th*w
    yE_dot=(sin_ph*sin_th*math.cos(psi)-cos_ph*math.sin(psi))*u+(sin_ph*sin_th*math.sin(psi)+cos_ph*math.cos(psi))*v+sin_ph*cos_th*w
    zE_dot=(cos_ph*sin_th*math.cos(psi)+sin_ph*math.sin(psi))*u+(cos_ph*sin_th*math.sin(psi)-sin_ph*cos_ph)*v+cos_ph*cos_th*w
    return np.array([u_dot,v_dot,w_dot,p_dot,q_dot,r_dot,phi_dot,theta_dot,psi_dot,xE_dot,yE_dot,zE_dot])


def rk4_step(fun, t, state, dt):
    k1=fun(t,state);k2=fun(t+0.5*dt,state+0.5*dt*k1)
    k3=fun(t+0.5*dt,state+0.5*dt*k2);k4=fun(t+dt,state+dt*k3)
    return state+dt*(k1+2*k2+2*k3+k4)/6.0


# =========================
# 5. Flight data generation
# =========================

def generate_flight(t_end=20.0, dt=0.02, noise_level=0.02, seed=0):
    N = int(t_end/dt)+1; t_vec=np.linspace(0.0,t_end,N)
    V0=150.0;alpha0=2.0*math.pi/180;u0=V0*math.cos(alpha0);w0=V0*math.sin(alpha0)
    state=np.array([u0,0.0,w0,0.0,0.0,0.0,0.0,alpha0,0.0,0.0,0.0,-1000.0])
    states=np.zeros((N,12));derivs=np.zeros((N,12));controls=np.zeros((N,3))
    for i,tt in enumerate(t_vec):
        states[i]=state;derivs[i]=f16_dynamics(tt,state)
        de,da,dr=control_inputs(tt);controls[i]=[de,da,dr]
        state=rk4_step(f16_dynamics,tt,state,dt)
    acc_body=derivs[:,0:6]
    rng=np.random.default_rng(seed)
    states_n=states.copy();states_n[:,0:6]+=rng.normal(0,noise_level,size=states_n[:,0:6].shape)
    acc_n=acc_body+rng.normal(0,noise_level,size=acc_body.shape)
    return t_vec,states_n,controls,acc_n


def build_nn_inputs(t_vec, states, controls):
    u=states[:,0];v=states[:,1];w=states[:,2];p=states[:,3];q=states[:,4];r=states[:,5]
    de=controls[:,0];da=controls[:,1];dr=controls[:,2]
    V=np.sqrt(u*u+v*v+w*w)+1e-6;alpha=np.arctan2(w,u);beta=np.arcsin(np.clip(v/V,-1,1))
    p_hat=p*B_REF/(2*V);q_hat=q*C_REF/(2*V);r_hat=r*B_REF/(2*V);V_norm=V/200.0
    X_in=np.stack([alpha,beta,p_hat,q_hat,r_hat,de,da,dr,V_norm],axis=1)
    extras={"u":u,"v":v,"w":w,"p":p,"q":q,"r":r,"phi":states[:,6],"theta":states[:,7],
            "alpha":alpha,"beta":beta,"p_hat":p_hat,"q_hat":q_hat,"r_hat":r_hat,
            "de":de,"da":da,"dr":dr,"V":V}
    return X_in, extras


# =========================
# 6. Model definition
# =========================

class AeroNet(nn.Module):
    def __init__(self, in_dim=9, hidden=64, layers=4):
        super().__init__()
        seq=[nn.Linear(in_dim,hidden),nn.Softplus()]
        for _ in range(layers-1):
            seq+=[nn.Linear(hidden,hidden),nn.Softplus()]
        seq+=[nn.Linear(hidden,6)]
        self.net=nn.Sequential(*seq)

    def forward(self,x):
        return self.net(x)


def load_model_package(model_path, device):
    model_path = os.path.expanduser(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"MODEL_PATH does not exist: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    net = AeroNet(in_dim=checkpoint.get("in_dim",9), hidden=checkpoint.get("hidden",64),
                  layers=checkpoint.get("layers",4)).to(device)
    state = checkpoint.get("state_dict") or checkpoint.get("model")
    if state is None:
        raise KeyError("Checkpoint missing 'state_dict' or 'model' key.")
    net.load_state_dict(state)
    net.eval()
    return net, checkpoint


def evaluate_coefficients(net, X_t, extras):
    net.eval()
    with torch.no_grad():
        coeff_pred = net(X_t).detach().cpu().numpy()
    alpha=extras["alpha"];beta=extras["beta"];p_hat=extras["p_hat"];q_hat=extras["q_hat"]
    r_hat=extras["r_hat"];de=extras["de"];da=extras["da"];dr=extras["dr"]
    coeff_true = np.zeros_like(coeff_pred)
    for i in range(len(alpha)):
        coeff_true[i]=aero_true(alpha[i],beta[i],p_hat[i],q_hat[i],r_hat[i],de[i],da[i],dr[i])
    def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))
    names=["CX","CY","CZ","Cl","Cm","Cn"]
    metrics={n:rmse(coeff_pred[:,k],coeff_true[:,k]) for k,n in enumerate(names)}
    return coeff_pred, coeff_true, metrics


# =========================
# 7. Visualization
# =========================

def show_eval_result(t_vec, coeff_true, coeff_pred, history, metrics):
    if len(history):
        plt.figure(figsize=(7, 4))
        plt.plot(history, linewidth=1); plt.yscale("log")
        plt.xlabel("Epoch"); plt.ylabel("Total Loss")
        plt.title("Training Loss (Physics Residuals)"); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.show()

    names = ["CX", "CY", "CZ", "Cl", "Cm", "Cn"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    for i, (ax, name) in enumerate(zip(axes.flat, names)):
        ax.plot(t_vec, coeff_true[:, i], label=f"{name} true", linewidth=2)
        ax.plot(t_vec, coeff_pred[:, i], "--", label=f"{name} PINN", linewidth=2)
        ax.set_ylabel(name); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
        if i >= 4:
            ax.set_xlabel("time [s]")
    fig.suptitle("Aerodynamic Coefficient Comparison", fontsize=14)
    plt.tight_layout(); plt.show()


# =========================
# 8. Eval entry
# =========================

def evaluate(model_path=MODEL_PATH, data_path=DATA_PATH, t_end=20.0, dt=0.02,
             noise=0.02, seed=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    net, checkpoint = load_model_package(model_path, device)
    history = checkpoint.get("history", [])

    t_vec, states, controls, acc = generate_flight(t_end=t_end, dt=dt, noise_level=noise, seed=seed)
    X_in, extras = build_nn_inputs(t_vec, states, controls)
    X_t = torch.tensor(X_in, dtype=torch.float32, device=device)

    coeff_pred, coeff_true, metrics = evaluate_coefficients(net, X_t, extras)

    print("Eval result")
    print(f"model_path: {model_path}")
    print(f"data points: {len(t_vec)}, t_end={t_end}s")
    print("Aero coefficient RMSEs:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4e}")

    show_eval_result(t_vec, coeff_true, coeff_pred, history, metrics)
    return {"coeff_pred": coeff_pred, "coeff_true": coeff_true, "metrics": metrics}


eval_result = evaluate()
