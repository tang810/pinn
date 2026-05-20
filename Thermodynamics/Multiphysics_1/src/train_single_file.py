"""
Notebook-friendly train file for Multiphysics_1 (PINNsformer).

Online Jupyter usage:
1. Upload data_7246.txt as a public asset (for eval during training).
2. Fill DATA_HASH and MODEL_HASH.
3. Run this whole file/cell. It saves trained_model_multiphysics1.pt to
   ~/minio/Resource/{MODEL_HASH}/ for later public-asset upload.

Physics: PINNsformer (Transformer) for 3D heat equation.
         Input (x,y,z,t) → Output T. Collocation via LHS.

Self-contained PyTorch-only.
"""

import copy
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

def get_device():
    """Detect device in priority: GPU (cuda) -> NPU (npu/ascend) -> CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_npu
        if torch_npu.npu.is_available():
            return torch.device("npu")
    except (ImportError, AttributeError):
        pass
    return torch.device("cpu")


# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = ""
MODEL_HASH = ""

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/minio/Resource/{MODEL_HASH}/trained_model_multiphysics1.pt")


# =========================
# 2. LHS Sampling
# =========================

def LHSample(D, bounds, N):
    result = np.empty([N, D]); temp = np.empty([N])
    d = 1.0 / N
    for i in range(D):
        for j in range(N):
            temp[j] = np.random.uniform(low=j * d, high=(j + 1) * d, size=1)[0]
        np.random.shuffle(temp)
        result[:, i] = temp
    b = np.array(bounds)
    np.add(np.multiply(result, (b[:, 1] - b[:, 0]), out=result), b[:, 0], out=result)
    return result


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# =========================
# 3. PINNsformer Model
# =========================

class WaveAct(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.ones(1))
        self.w2 = nn.Parameter(torch.ones(1))
    def forward(self, x):
        return self.w1 * torch.sin(x) + self.w2 * torch.cos(x)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=256):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(d_model, d_ff), WaveAct(),
            nn.Linear(d_ff, d_ff), WaveAct(),
            nn.Linear(d_ff, d_model))
    def forward(self, x): return self.linear(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.ff = FeedForward(d_model)
        self.act1 = WaveAct(); self.act2 = WaveAct()
    def forward(self, x): x2=self.act1(x); x=x+self.attn(x2,x2,x2)[0]; x2=self.act2(x); x=x+self.ff(x2); return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.ff = FeedForward(d_model)
        self.act1 = WaveAct(); self.act2 = WaveAct()
    def forward(self, x, e):
        x2 = self.act1(x); x = x + self.attn(x2, e, e)[0]
        x2 = self.act2(x); x = x + self.ff(x2); return x


class Encoder(nn.Module):
    def __init__(self, d_model, N, heads):
        super().__init__()
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.act = WaveAct()
    def forward(self, x):
        for l in self.layers: x = l(x); return self.act(x)

class Decoder(nn.Module):
    def __init__(self, d_model, N, heads):
        super().__init__()
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.act = WaveAct()
    def forward(self, x, e):
        for l in self.layers: x = l(x, e); return self.act(x)

class PINNsformer(nn.Module):
    def __init__(self, d_out=1, d_model=64, d_hidden=512, N=1, heads=2):
        super().__init__()
        self.linear_emb = nn.Linear(4, d_model)
        self.encoder = Encoder(d_model, N, heads)
        self.decoder = Decoder(d_model, N, heads)
        self.linear_out = nn.Sequential(
            nn.Linear(d_model, d_hidden), WaveAct(),
            nn.Linear(d_hidden, d_hidden), WaveAct(),
            nn.Linear(d_hidden, d_out))
    def forward(self, x, y, z, t):
        src = self.linear_emb(torch.cat([x, y, z, t], dim=-1))
        e = self.encoder(src); d = self.decoder(src, e)
        return self.linear_out(d)


# =========================
# 4. Training data (LHS)
# =========================

def get_pinn_domain_points(device, num_res=4096, num_ic=1024, num_bc=400):
    domain_bounds = [[0, 1]] * 4
    res = torch.tensor(LHSample(4, domain_bounds, num_res), dtype=torch.float32, requires_grad=True, device=device)
    ic = torch.tensor(np.hstack([LHSample(3, domain_bounds[:3], num_ic), np.zeros((num_ic, 1))]),
                      dtype=torch.float32, requires_grad=True, device=device)
    bc = {}
    fb = [[0, 1]] * 3
    for face, const in [('x0', 0.0), ('x1', 1.0), ('y0', 0.0), ('y1', 1.0), ('z0', 0.0), ('z1', 1.0)]:
        pts = LHSample(3, fb, num_bc)
        if 'x' in face: arr = np.hstack([np.full((num_bc,1),const), pts[:,:1], pts[:,1:]])
        elif 'y' in face: arr = np.hstack([pts[:,:1], np.full((num_bc,1),const), pts[:,1:]])
        else: arr = np.hstack([pts[:,:2], np.full((num_bc,1),const), pts[:,2:]])
        bc[face] = torch.tensor(arr, dtype=torch.float32, requires_grad=True, device=device)
    return res, ic, bc


def make_time_sequence(t, num_step, step):
    return t.unsqueeze(1).repeat(1, num_step, 1)


# =========================
# 5. Physics loss
# =========================

def compute_physics_loss(model, res, ic, bc, k=1.0, rho=1.0, Cp=1.0, eps=0.1, sigma=5.67e-8, Tamb=3.0, T0=273.0, qsum=400.0):
    # PDE: ρCp·∂T/∂t = k·∇²T
    x,y,z,t = (res[..., i:i+1] for i in range(4))
    T = model(x, y, z, t)
    ones = torch.ones_like(T)
    Tx = torch.autograd.grad(T, x, ones, create_graph=True, retain_graph=True)[0]
    Ty = torch.autograd.grad(T, y, ones, create_graph=True, retain_graph=True)[0]
    Tz = torch.autograd.grad(T, z, ones, create_graph=True, retain_graph=True)[0]
    Tt = torch.autograd.grad(T, t, ones, create_graph=True, retain_graph=True)[0]
    Txx = torch.autograd.grad(Tx, x, ones, create_graph=True, retain_graph=True)[0]
    Tyy = torch.autograd.grad(Ty, y, ones, create_graph=True, retain_graph=True)[0]
    Tzz = torch.autograd.grad(Tz, z, ones, create_graph=True, retain_graph=True)[0]
    loss_pde = torch.mean((rho * Cp * Tt - k * (Txx + Tyy + Tzz)) ** 2)

    # IC: T = T0 at t=0
    xi,yi,zi,ti = (ic[..., i:i+1] for i in range(4))
    Ti = model(xi, yi, zi, ti)
    loss_ic = torch.mean((Ti.squeeze() - T0) ** 2)

    # BC: radiation + Neumann (6 faces)
    loss_bc = 0.0
    for key in ['x0', 'x1', 'y0', 'y1', 'z0', 'z1']:
        xb,yb,zb,tb = (bc[key][..., i:i+1] for i in range(4))
        Tb = model(xb, yb, zb, tb)
        Tx_b = torch.autograd.grad(Tb, xb if 'x' in key else yb if 'y' in key else zb, ones, create_graph=True, retain_graph=True)[0]
        rad = eps * sigma * (Tamb**4 - Tb**4)
        if key == 'x1': res_b = k * Tx_b - qsum
        elif key in ('y1', 'z1'): res_b = rad - k * Tx_b
        else: res_b = rad + k * Tx_b
        loss_bc += torch.mean(res_b ** 2)

    return loss_pde + loss_ic + loss_bc, loss_pde.detach(), loss_ic.detach(), loss_bc.detach()


# =========================
# 6. Visualization
# =========================

def show_train_result(model, device, data_path, history):
    if history:
        plt.figure(figsize=(7,4)); plt.plot(history); plt.yscale("log")
        plt.xlabel("Epoch"); plt.ylabel("Total Loss"); plt.title("Training Loss")
        plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

    data_path = os.path.expanduser(data_path) if data_path else "data/data_7246.txt"
    if os.path.isfile(data_path):
        df = pd.read_csv(data_path, delimiter=' ', header=None).to_numpy()
        idx = np.linspace(0, len(df)-1, min(5000, len(df))).astype(int)
        x, y, z, t, T_true = (torch.tensor(df[idx, i:i+1], dtype=torch.float32, device=device) for i in range(5))
        model.eval()
        with torch.no_grad():
            T_pred = model(x, y, z, t).cpu().numpy()
        plt.figure(figsize=(6,6))
        plt.scatter(T_true.cpu().numpy(), T_pred, s=3, alpha=0.4)
        plt.plot([250,500],[250,500],'r--'); plt.xlabel("True T (K)"); plt.ylabel("Pred T (K)")
        plt.title("PINNsformer Prediction"); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()


# =========================
# 7. Train entry
# =========================

def train(data_path=DATA_PATH, model_path=MODEL_PATH, d_model=64, d_hidden=512,
          N=1, heads=2, epochs=100, lr=1.0, time_steps=5, step_size=0.02,
          k=1.0, rho=1.0, Cp=1.0, eps=0.1, T0=273.0, Tamb=3.0, qsum=400.0, sigma=5.67e-8):
    t0 = time.time()
    device = get_device()
    print(f"device={device}")

    res, ic, bc = get_pinn_domain_points(device)
    res = make_time_sequence(res, time_steps, step_size)
    ic = make_time_sequence(ic, time_steps, step_size)
    for k in bc: bc[k] = make_time_sequence(bc[k], time_steps, step_size)

    model = PINNsformer(d_out=1, d_model=d_model, d_hidden=d_hidden, N=N, heads=heads).to(device)
    model.apply(lambda m: (torch.nn.init.xavier_uniform_(m.weight), setattr(m.bias, 'data', m.bias.data.fill_(0.01))) if isinstance(m, nn.Linear) else None)
    print(f"Model: {sum(p.numel() for p in model.parameters())} params")

    opt = torch.optim.LBFGS(model.parameters(), lr=lr, line_search_fn='strong_wolfe')
    history = []

    print(f"Training {epochs} epochs...")
    for ep in range(epochs):
        def closure():
            opt.zero_grad()
            loss, _, _, _ = compute_physics_loss(model, res, ic, bc, k=k, rho=rho, Cp=Cp, eps=eps, sigma=sigma, Tamb=Tamb, T0=T0, qsum=qsum)
            loss.backward(retain_graph=True); return loss
        opt.step(closure)
        with torch.no_grad():
            loss, lp, li, lb = compute_physics_loss(model, res, ic, bc, k=k, rho=rho, Cp=Cp, eps=eps, sigma=sigma, Tamb=Tamb, T0=T0, qsum=qsum)
        history.append(float(loss.cpu()))
        print(f"Epoch {ep+1:4d}/{epochs} | total={history[-1]:.3e} pde={float(lp):.3e} ic={float(li):.3e} bc={float(lb):.3e}")

    model_path = os.path.expanduser(model_path)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "history": history, "d_model": d_model, "d_hidden": d_hidden, "N": N, "heads": heads}, model_path)

    print("Training finished"); print(f"model saved to: {model_path}")
    print(f"elapsed: {time.time() - t0:.1f}s")
    show_train_result(model, device, data_path, history)
    return {"model": model, "model_path": model_path, "history": history}


train_result = train()
