"""
Notebook-friendly train file for Solar_Wind_Forecast.

Online Jupyter usage:
1. Fill DATA_HASH (optional) and MODEL_HASH.
2. Run this whole file/cell. It saves trained_model_solar_wind.pt to
   ~/minio/Resource/{MODEL_HASH}/ for later public-asset upload.

Physics: GRU forecasting 7 solar-wind features with a physics penalty
         enforcing |E| <= alpha * ||V x B|| (ideal MHD constraint).
"""

import json
import math
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = ""
MODEL_HASH = ""

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/minio/Resource/{MODEL_HASH}/trained_model_solar_wind.pt")

FEATURES = ["E", "Vx", "Vy", "Vz", "Bx", "By", "Bz"]


# =========================
# 2. Synthetic data generation
# =========================

def make_synthetic_hourly(n_hours=24*240, seed=0, alpha_true=0.02):
    rng = np.random.default_rng(seed)
    t = np.arange(n_hours, dtype=np.float64)
    daily = np.sin(2*np.pi*t/24.0); weekly = np.sin(2*np.pi*t/(24.0*7.0))
    slow = np.sin(2*np.pi*t/(24.0*60.0))

    def ar1(mu, sig, phi, noise):
        base = mu + sig*rng.normal(0,1,n_hours)
        x = np.zeros(n_hours); eps = rng.normal(0, noise/10.0, n_hours)
        for i in range(1,n_hours): x[i]=phi*x[i-1]+eps[i]
        return base+x

    Vx=ar1(-450,15,0.999,0.6)+30*daily+20*weekly+10*slow
    Vy=ar1(10,8,0.998,0.4)+12*daily+6*weekly
    Vz=ar1(0,7,0.998,0.4)+8*daily-5*weekly
    Bx=ar1(0,1.5,0.995,0.12)+1.2*daily+0.8*weekly
    By=ar1(0,1.5,0.995,0.12)-1.0*daily+0.6*weekly
    Bz=ar1(0,1.5,0.995,0.12)+0.7*daily+0.9*slow
    V=np.stack([Vx,Vy,Vz],1); B=np.stack([Bx,By,Bz],1)
    cross_norm=np.linalg.norm(np.cross(V,B),axis=1)
    lag=12; cross_lag=np.roll(cross_norm,lag); cross_lag[:lag]=cross_norm[:lag]
    noise=rng.normal(0,0.1/10.0,n_hours)
    E=0.85*alpha_true*cross_norm+0.15*alpha_true*cross_lag+noise
    E=np.clip(E,-np.inf,alpha_true*cross_norm)
    violate=rng.random(n_hours)<0.03
    E[violate]=alpha_true*cross_norm[violate]+np.abs(rng.normal(0,0.3,violate.sum()))
    df=pd.DataFrame(np.column_stack([E,V,B]),columns=FEATURES,
                    index=pd.date_range("1992-01-01",periods=n_hours,freq="h"))
    return df


def normalize_z(df):
    mu=df.mean(); sig=df.std(ddof=0).replace(0,1.0)
    return (df-mu)/sig, mu, sig


def split_by_time(df, train_ratio=0.6, val_ratio=0.3):
    T=len(df); i1=int(T*train_ratio); i2=int(T*(train_ratio+val_ratio))
    return df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]


def fit_alpha_raw(df, eps=1e-6):
    arr=df[FEATURES].to_numpy(dtype=np.float64)
    E=np.abs(arr[:,0]); V=arr[:,1:4]; B=arr[:,4:7]
    cross_norm=np.linalg.norm(np.cross(V,B),axis=1)
    ratio=E/(cross_norm+eps); ratio=ratio[np.isfinite(ratio)]
    return float(np.median(ratio)) if ratio.size else 1.0


# =========================
# 3. Dataset & Model
# =========================

class WindowDataset(Dataset):
    def __init__(self, arr, span=24, prior=12):
        self.arr=torch.as_tensor(arr,dtype=torch.float32)
        self.span=span; self.prior=prior; self.T,self.D=self.arr.shape
        self.idxs=list(range(self.T-(span+prior)))
    def __len__(self):
        return len(self.idxs)
    def __getitem__(self,i):
        t=self.idxs[i]
        return self.arr[t:t+self.span], self.arr[t+self.span+self.prior-1]


class GRUForecast(nn.Module):
    def __init__(self, input_dim=7, hidden=128, num_layers=4, dropout=0.1):
        super().__init__()
        self.gru=nn.GRU(input_dim,hidden,num_layers=num_layers,batch_first=True,
                         dropout=(dropout if num_layers>1 else 0.0))
        self.head=nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,input_dim))
    def forward(self,x):
        out,_=self.gru(x); return self.head(out[:,-1,:])


# =========================
# 4. Physics penalty & metrics
# =========================

def inverse_z_torch(x_norm, mu_t, sig_t):
    return x_norm*sig_t+mu_t


def physics_penalty(yhat_norm, alpha, mu_t, sig_t):
    yhat_raw=inverse_z_torch(yhat_norm,mu_t,sig_t)
    E=yhat_raw[:,0].abs(); V=yhat_raw[:,1:4]; B=yhat_raw[:,4:7]
    cross=torch.cross(V,B,dim=1)
    cross_norm=torch.linalg.vector_norm(cross,ord=2,dim=1)
    g=E-alpha*cross_norm
    return torch.relu(g).mean()


def compute_r2(y, yhat):
    ss_res=np.sum((y-yhat)**2,axis=0); ss_tot=np.sum((y-y.mean(axis=0))**2,axis=0)+1e-9
    return 1-ss_res/ss_tot


# =========================
# 5. Visualization
# =========================

def show_train_result(history, y_test, yhat_test, alpha, mu, sig):
    epochs=np.arange(1,len(history["train_loss"])+1)
    fig, axes = plt.subplots(2,2,figsize=(12,9))

    axes[0,0].plot(epochs,history["train_loss"]); axes[0,0].set_title("Train Loss")
    axes[0,0].set_xlabel("Epoch"); axes[0,0].grid(alpha=0.3)

    axes[0,1].plot(epochs,history["val_phys"]); axes[0,1].set_title("Val Physics Penalty")
    axes[0,1].set_xlabel("Epoch"); axes[0,1].grid(alpha=0.3)

    iVx=FEATURES.index("Vx"); N=y_test.shape[0]; m=min(2000,N); idx=np.linspace(0,N-1,m).astype(int)
    axes[1,0].scatter(y_test[idx,iVx],yhat_test[idx,iVx],s=6,alpha=0.5)
    mm=max(abs(y_test[idx,iVx]).max(),abs(yhat_test[idx,iVx]).max())
    axes[1,0].plot([-mm,mm],[-mm,mm],"r--"); axes[1,0].set_title("Scatter: Vx (norm.)")
    axes[1,0].set_xlabel("True"); axes[1,0].grid(alpha=0.3)

    yhat_raw=yhat_test*sig.values[None,:]+mu.values[None,:]
    E_abs=np.abs(yhat_raw[:,0]); V=yhat_raw[:,1:4]; B=yhat_raw[:,4:7]
    rhs=alpha*np.linalg.norm(np.cross(V,B),axis=1)
    idx2=np.linspace(0,N-1,m).astype(int)
    axes[1,1].scatter(rhs[idx2],E_abs[idx2],s=6,alpha=0.5)
    mx=max(rhs[idx2].max(),E_abs[idx2].max())
    axes[1,1].plot([0,mx],[0,mx],"r--"); axes[1,1].set_title("Physics: |E| vs alpha*||VxB||")
    axes[1,1].set_xlabel("alpha*||VxB||"); axes[1,1].grid(alpha=0.3)

    plt.tight_layout(); plt.show()

    r2=compute_r2(y_test,yhat_test)
    plt.figure(figsize=(7,4))
    plt.bar(np.arange(len(FEATURES)),r2); plt.axhline(0,color="gray",ls="--")
    plt.xticks(np.arange(len(FEATURES)),FEATURES,rotation=45)
    plt.ylabel("R2"); plt.title("R2 per Feature (test, normalized)"); plt.tight_layout(); plt.show()


# =========================
# 6. Train entry
# =========================

def train(data_path=DATA_PATH, model_path=MODEL_PATH,
          n_hours=24*240, seed=0, span=24, prior=12, hidden=128, num_layers=4,
          dropout=0.1, epochs=200, lr=2e-3, weight_decay=1e-4, batch_size=256,
          lam=0.2, clip=1.0, verbose_every=20):
    t0=time.time()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    df_raw=make_synthetic_hourly(n_hours=n_hours,seed=seed)
    train_raw,val_raw,test_raw=split_by_time(df_raw)
    alpha=fit_alpha_raw(train_raw); print(f"alpha (fit)={alpha:.6f}")

    train_n,mu,sig=normalize_z(train_raw)
    val_n=(val_raw-mu)/sig; test_n=(test_raw-mu)/sig
    train_arr=train_n[FEATURES].to_numpy(dtype=np.float32)
    val_arr=val_n[FEATURES].to_numpy(dtype=np.float32)
    test_arr=test_n[FEATURES].to_numpy(dtype=np.float32)

    train_ld=DataLoader(WindowDataset(train_arr,span,prior),batch_size=batch_size,shuffle=True,drop_last=True)
    val_ld=DataLoader(WindowDataset(val_arr,span,prior),batch_size=batch_size,shuffle=False)
    test_ld=DataLoader(WindowDataset(test_arr,span,prior),batch_size=batch_size,shuffle=False)

    model=GRUForecast(len(FEATURES),hidden,num_layers,dropout).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
    mu_t=torch.tensor(mu[FEATURES].to_numpy(dtype=np.float32),device=device)
    sig_t=torch.tensor(sig[FEATURES].to_numpy(dtype=np.float32),device=device)

    history={"train_loss":[],"val_phys":[]}
    best_val=-1e9; best_state=None

    print(f"Training {epochs} epochs (n_train={len(train_ld.dataset)})...")
    for ep in range(1,epochs+1):
        model.train(); total,n=0.0,0
        for x,y in train_ld:
            x,y=x.to(device),y.to(device); opt.zero_grad()
            yhat=model(x)
            mse=torch.mean((yhat-y)**2); phy=physics_penalty(yhat,alpha,mu_t,sig_t)
            loss=(1-lam)*mse+lam*phy; loss.backward()
            if clip is not None: torch.nn.utils.clip_grad_norm_(model.parameters(),clip)
            opt.step(); total+=loss.item()*x.size(0); n+=x.size(0)
        history["train_loss"].append(total/max(n,1))
        model.eval(); ys,yhats=[],[]
        with torch.no_grad():
            for x,y in val_ld:
                x=x.to(device); yh=model(x)
                ys.append(y.numpy()); yhats.append(yh.cpu().numpy())
            yv=np.concatenate(ys,0); yhv=np.concatenate(yhats,0)
            r2v=float(np.mean(compute_r2(yv,yhv)))
            phy_v=float(physics_penalty(torch.tensor(yhv),alpha,mu_t,sig_t).cpu())
        history["val_phys"].append(phy_v)
        if r2v>best_val: best_val=r2v; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if ep==1 or ep%verbose_every==0 or ep==epochs:
            print(f"Epoch {ep:4d}/{epochs} | loss={history['train_loss'][-1]:.4f} | val_r2={r2v:.4f} | val_phys={phy_v:.4f}")

    if best_state is not None: model.load_state_dict(best_state)
    model.eval()

    model_path=os.path.expanduser(model_path)
    os.makedirs(os.path.dirname(model_path),exist_ok=True)
    checkpoint={"project":"Solar_Wind_Forecast","state_dict":model.state_dict(),"history":history,
                "hidden":hidden,"num_layers":num_layers,"dropout":dropout,
                "mu":mu,"sig":sig,"alpha":alpha,"span":span,"prior":prior,"seed":seed}
    torch.save(checkpoint,model_path)

    ys_t,yhats_t=[],[]
    with torch.no_grad():
        for x,y in test_ld: x=x.to(device); yh=model(x); ys_t.append(y.numpy()); yhats_t.append(yh.cpu().numpy())
    y_test=np.concatenate(ys_t,0); yhat_test=np.concatenate(yhats_t,0)
    r2=compute_r2(y_test,yhat_test)

    print("Training finished"); print(f"model saved to: {model_path}")
    print(f"elapsed: {time.time()-t0:.1f}s"); print(f"Test R2 macro: {float(np.mean(r2)):.4f}")
    for i,f in enumerate(FEATURES): print(f"  {f}: {r2[i]:.4f}")

    show_train_result(history,y_test,yhat_test,alpha,mu,sig)
    return {"model":model,"model_path":model_path,"history":history}


train_result = train()
