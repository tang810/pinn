"""
Notebook-friendly eval file for Solar_Wind_Forecast.

Online Jupyter usage:
1. Upload the trained .pt model as a public asset.
2. Fill MODEL_HASH and DATA_HASH, then run this whole file/cell.

This file loads the model, generates synthetic data (same seed/config),
evaluates on the test set, and shows figures with plt.show().
It does not save images.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = "4feacb93a22c4be0b69ef6055523fd50"
MODEL_HASH = "4888f353c86643ef92186984255eec2c"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/trained_model_solar_wind.pt")

FEATURES = ["E", "Vx", "Vy", "Vz", "Bx", "By", "Bz"]


# =========================
# 2. Synthetic data generation
# =========================

def make_synthetic_hourly(n_hours=24*240, seed=0, alpha_true=0.02):
    rng=np.random.default_rng(seed); t=np.arange(n_hours,dtype=np.float64)
    daily=np.sin(2*np.pi*t/24.0); weekly=np.sin(2*np.pi*t/(24.0*7.0))
    slow=np.sin(2*np.pi*t/(24.0*60.0))
    def ar1(mu,sig,phi,noise):
        base=mu+sig*rng.normal(0,1,n_hours); x=np.zeros(n_hours)
        eps=rng.normal(0,noise/10.0,n_hours)
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
    lag=12;cross_lag=np.roll(cross_norm,lag);cross_lag[:lag]=cross_norm[:lag]
    noise=rng.normal(0,0.1/10.0,n_hours)
    E=0.85*alpha_true*cross_norm+0.15*alpha_true*cross_lag+noise
    E=np.clip(E,-np.inf,alpha_true*cross_norm)
    violate=rng.random(n_hours)<0.03
    E[violate]=alpha_true*cross_norm[violate]+np.abs(rng.normal(0,0.3,violate.sum()))
    return pd.DataFrame(np.column_stack([E,V,B]),columns=FEATURES,
                        index=pd.date_range("1992-01-01",periods=n_hours,freq="h"))


def normalize_z(df):
    mu=df.mean();sig=df.std(ddof=0).replace(0,1.0)
    return (df-mu)/sig,mu,sig


def split_by_time(df,train_ratio=0.6,val_ratio=0.3):
    T=len(df);i1=int(T*train_ratio);i2=int(T*(train_ratio+val_ratio))
    return df.iloc[:i1],df.iloc[i1:i2],df.iloc[i2:]


def fit_alpha_raw(df,eps=1e-6):
    arr=df[FEATURES].to_numpy(dtype=np.float64)
    E=np.abs(arr[:,0]);V=arr[:,1:4];B=arr[:,4:7]
    cross_norm=np.linalg.norm(np.cross(V,B),axis=1)
    ratio=E/(cross_norm+eps);ratio=ratio[np.isfinite(ratio)]
    return float(np.median(ratio)) if ratio.size else 1.0


# =========================
# 3. Dataset & Model
# =========================

class WindowDataset(Dataset):
    def __init__(self,arr,span=24,prior=12):
        self.arr=torch.as_tensor(arr,dtype=torch.float32)
        self.span=span;self.prior=prior;self.T,self.D=self.arr.shape
        self.idxs=list(range(self.T-(span+prior)))
    def __len__(self): return len(self.idxs)
    def __getitem__(self,i):
        t=self.idxs[i]; return self.arr[t:t+self.span], self.arr[t+self.span+self.prior-1]


class GRUForecast(nn.Module):
    def __init__(self,input_dim=7,hidden=128,num_layers=4,dropout=0.1):
        super().__init__()
        self.gru=nn.GRU(input_dim,hidden,num_layers=num_layers,batch_first=True,
                         dropout=(dropout if num_layers>1 else 0.0))
        self.head=nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,input_dim))
    def forward(self,x): out,_=self.gru(x); return self.head(out[:,-1,:])


# =========================
# 4. Physics penalty & metrics
# =========================

def inverse_z_torch(x_norm,mu_t,sig_t):
    return x_norm*sig_t+mu_t


def physics_penalty(yhat_norm,alpha,mu_t,sig_t):
    if not torch.is_tensor(yhat_norm):
        yhat_norm=torch.as_tensor(yhat_norm,dtype=mu_t.dtype,device=mu_t.device)
    else:
        yhat_norm=yhat_norm.to(device=mu_t.device,dtype=mu_t.dtype)
    yhat_raw=inverse_z_torch(yhat_norm,mu_t,sig_t)
    E=yhat_raw[:,0].abs();V=yhat_raw[:,1:4];B=yhat_raw[:,4:7]
    cross=torch.cross(V,B,dim=1);cross_norm=torch.linalg.vector_norm(cross,ord=2,dim=1)
    return torch.relu(E-alpha*cross_norm).mean()


def compute_r2(y,yhat):
    ss_res=np.sum((y-yhat)**2,axis=0);ss_tot=np.sum((y-y.mean(axis=0))**2,axis=0)+1e-9
    return 1-ss_res/ss_tot


def load_model_package(model_path,device):
    model_path=os.path.expanduser(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"MODEL_PATH does not exist: {model_path}")
    ckpt=torch.load(model_path,map_location=device,weights_only=False)
    # Support: raw state_dict, or dict with "state_dict"/"model" key
    if isinstance(ckpt,dict) and any(k.startswith("gru.") or k.startswith("head.") for k in ckpt.keys()):
        state=ckpt; extras={}
    elif isinstance(ckpt,dict):
        state=ckpt.get("state_dict") or ckpt.get("model")
        extras=ckpt
    else:
        state=ckpt; extras={}
    if state is None:
        raise KeyError("Checkpoint format not recognized.")
    hidden=extras.get("hidden",128);num_layers=extras.get("num_layers",4)
    dropout=extras.get("dropout",0.1)
    model=GRUForecast(len(FEATURES),hidden,num_layers,dropout).to(device)
    model.load_state_dict(state);model.eval()
    return model,extras


# =========================
# 5. Visualization
# =========================

def show_eval_result(history,y_test,yhat_test,alpha,mu,sig):
    if history and "train_loss" in history and len(history["train_loss"]):
        epochs=np.arange(1,len(history["train_loss"])+1)
        fig,axes=plt.subplots(1,2,figsize=(10,3.5))
        axes[0].plot(epochs,history["train_loss"]);axes[0].set_title("Train Loss")
        axes[0].set_xlabel("Epoch");axes[0].grid(alpha=0.3)
        axes[1].plot(epochs,history.get("val_phys",[]));axes[1].set_title("Val Physics Penalty")
        axes[1].set_xlabel("Epoch");axes[1].grid(alpha=0.3)
        plt.tight_layout();plt.show()

    iVx=FEATURES.index("Vx");N=y_test.shape[0];m=min(2000,N)
    idx=np.linspace(0,N-1,m).astype(int)
    plt.figure(figsize=(5,5))
    plt.scatter(y_test[idx,iVx],yhat_test[idx,iVx],s=6,alpha=0.5)
    mm=max(abs(y_test[idx,iVx]).max(),abs(yhat_test[idx,iVx]).max())
    plt.plot([-mm,mm],[-mm,mm],"r--");plt.xlabel("True");plt.ylabel("Pred")
    plt.title("Scatter: Vx (normalized)");plt.grid(alpha=0.3);plt.tight_layout();plt.show()

    yhat_raw=yhat_test*sig.values[None,:]+mu.values[None,:]
    E_abs=np.abs(yhat_raw[:,0]);V=yhat_raw[:,1:4];B=yhat_raw[:,4:7]
    rhs=alpha*np.linalg.norm(np.cross(V,B),axis=1)
    idx2=np.linspace(0,N-1,m).astype(int)
    plt.figure(figsize=(5,5))
    plt.scatter(rhs[idx2],E_abs[idx2],s=6,alpha=0.5)
    mx=max(rhs[idx2].max(),E_abs[idx2].max())
    plt.plot([0,mx],[0,mx],"r--");plt.xlabel("alpha*||VxB||");plt.ylabel("|E|")
    plt.title("Physics: |E| vs alpha*||VxB|| (original units)");plt.grid(alpha=0.3)
    plt.tight_layout();plt.show()

    r2=compute_r2(y_test,yhat_test)
    plt.figure(figsize=(7,4))
    plt.bar(np.arange(len(FEATURES)),r2);plt.axhline(0,color="gray",ls="--")
    plt.xticks(np.arange(len(FEATURES)),FEATURES,rotation=45)
    plt.ylabel("R2");plt.title("R2 per Feature (test, normalized)");plt.tight_layout();plt.show()


# =========================
# 6. Eval entry
# =========================

def evaluate(model_path=MODEL_PATH,data_path=DATA_PATH,
             n_hours=24*240,span=24,prior=12,batch_size=256):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    model,checkpoint=load_model_package(model_path,device)
    seed=checkpoint.get("seed",0);history=checkpoint.get("history",{})
    alpha=checkpoint.get("alpha");mu=checkpoint.get("mu");sig=checkpoint.get("sig")

    df_raw=make_synthetic_hourly(n_hours=n_hours,seed=seed)
    train_raw,val_raw,test_raw=split_by_time(df_raw)
    if alpha is None: alpha=fit_alpha_raw(train_raw)
    if mu is None or sig is None:
        train_n,mu,sig=normalize_z(train_raw)

    test_n=(test_raw-mu)/sig; test_arr=test_n[FEATURES].to_numpy(dtype=np.float32)
    test_ld=DataLoader(WindowDataset(test_arr,span,prior),batch_size=batch_size,shuffle=False)

    mu_t=torch.tensor(mu[FEATURES].to_numpy(dtype=np.float32),device=device)
    sig_t=torch.tensor(sig[FEATURES].to_numpy(dtype=np.float32),device=device)

    ys,yhats=[],[]
    with torch.no_grad():
        for x,y in test_ld:
            x=x.to(device);yh=model(x);ys.append(y.numpy());yhats.append(yh.cpu().numpy())
    y_test=np.concatenate(ys,0);yhat_test=np.concatenate(yhats,0)
    r2=compute_r2(y_test,yhat_test)

    yhat_t=torch.as_tensor(yhat_test,dtype=torch.float32,device=device)
    phy_v=float(physics_penalty(yhat_t,alpha,mu_t,sig_t).detach().cpu())

    print("Eval result")
    print(f"model_path: {model_path}")
    print(f"alpha: {alpha:.6f}")
    print(f"test points: {len(y_test)}")
    print(f"Test R2 macro: {float(np.mean(r2)):.4f}")
    print(f"Physics penalty: {phy_v:.4e}")
    print("R2 per feature:")
    for i,f in enumerate(FEATURES): print(f"  {f}: {r2[i]:.4f}")

    show_eval_result(history,y_test,yhat_test,alpha,mu,sig)
    return {"y_test":y_test,"yhat_test":yhat_test,"r2":r2}


eval_result = evaluate()
