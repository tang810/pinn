"""
Notebook-friendly eval file for Phone2D_HeatPINN.

Online Jupyter usage:
1. Upload the trained .pt model as a public asset.
2. Upload phone_thermal_case.yaml as a public asset (optional, default config built-in).
3. Fill MODEL_HASH and DATA_HASH, then run this whole file/cell.

This file loads the model, evaluates the phone thermal field at key times,
and shows figures with plt.show(). It does not save images.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, BoxStyle, Rectangle

import torch
import torch.nn as nn
import yaml


# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = "a4ee35a2e2114e90924a67429bf88b5c"
MODEL_HASH = "359c399170f544d6b5b95e8ea07006e5"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/trained_model_phone_thermal.pt")


# =========================
# 2. Default config
# =========================

DEFAULT_CFG = {
    "geometry": {"Lx": 0.07, "Ly": 0.15, "t_max": 1800.0},
    "material": {"rho": 2500.0, "cp": 900.0, "k": 5.0},
    "environment": {"T_inf": 298.0, "h_env": 5.0},
    "soc": {"x_center": 0.035, "y_center": 0.075, "width": 0.01, "height": 0.01,
            "q0": 3e6, "tau_q": 100.0, "use_slow_source": True},
    "nondim": {"theta_scale": 10.0},
}


def load_or_default_config(data_path):
    data_path = os.path.expanduser(data_path) if data_path else ""
    yaml_path = None
    if data_path and os.path.isfile(data_path) and data_path.endswith((".yaml", ".yml")):
        yaml_path = data_path
    elif data_path and os.path.isdir(data_path):
        for f in sorted(os.listdir(data_path)):
            if f.endswith((".yaml", ".yml")): yaml_path = os.path.join(data_path, f); break
    if yaml_path:
        with open(yaml_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    return DEFAULT_CFG


def build_physical_params(cfg):
    rho=cfg["material"]["rho"];cp=cfg["material"]["cp"];k=cfg["material"]["k"]
    Lx=cfg["geometry"]["Lx"];Ly=cfg["geometry"]["Ly"];t_max=cfg["geometry"]["t_max"]
    T_inf=cfg["environment"]["T_inf"];h_env=cfg["environment"]["h_env"]
    theta_scale=cfg["nondim"]["theta_scale"]
    alpha=k/(rho*cp)
    return dict(rho=rho,cp=cp,k=k,alpha=alpha,Lx=Lx,Ly=Ly,t_max=t_max,
                T_inf=T_inf,h_env=h_env,theta_scale=theta_scale,
                Lx_scale=Lx,Ly_scale=Ly,t_scale=t_max,
                alpha_x_hat=alpha*t_max/Lx**2,alpha_y_hat=alpha*t_max/Ly**2,
                Bi_x=h_env*Lx/k,Bi_y=h_env*Ly/k,q_to_theta=1.0/(rho*cp))


# =========================
# 3. Model definition
# =========================

class HeatPINN(nn.Module):
    def __init__(self,in_dim=3,hidden=64,n_layers=5):
        super().__init__()
        layers=[nn.Linear(in_dim,hidden),nn.Tanh()]
        for _ in range(n_layers-2): layers+=[nn.Linear(hidden,hidden),nn.Tanh()]
        layers+=[nn.Linear(hidden,1)]
        self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)


def load_model_package(model_path,device):
    model_path=os.path.expanduser(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"MODEL_PATH does not exist: {model_path}")
    ckpt=torch.load(model_path,map_location=device,weights_only=False)
    if any(k.startswith("net.") for k in ckpt.keys()):
        state=ckpt;extras={}
    else:
        state=ckpt.get("state_dict") or ckpt.get("model") or ckpt
        extras=ckpt if isinstance(ckpt,dict) else {}
    hidden=extras.get("hidden")
    n_layers=extras.get("n_layers")
    if hidden is None or n_layers is None:
        weight_keys=sorted([k for k,v in state.items() if k.startswith("net.") and k.endswith(".weight")],
                           key=lambda k:int(k.split(".")[1]))
        if weight_keys:
            hidden=int(state[weight_keys[0]].shape[0])
            n_layers=len(weight_keys)
        else:
            hidden=64;n_layers=5
    model=HeatPINN(hidden=hidden,n_layers=n_layers).to(device);model.load_state_dict(state);model.eval()
    return model,extras


# =========================
# 4. Evaluation
# =========================

def predict_grid_at_time(model,device,cfg,params,t_val,nx=60,ny=60):
    Lx=params["Lx"];Ly=params["Ly"]
    xl=np.linspace(0,Lx,nx);yl=np.linspace(0,Ly,ny);X,Y=np.meshgrid(xl,yl,indexing="xy")
    xf=X.reshape(-1,1);yf=Y.reshape(-1,1);tf=np.full_like(xf,t_val)
    xyh=torch.tensor(np.concatenate([xf/Lx,yf/Ly,tf/params["t_scale"]],1),dtype=torch.float32,device=device)
    model.eval()
    with torch.no_grad():
        Th=model(xyh).cpu().numpy().reshape(ny,nx)
    return X,Y,params["T_inf"]+params["theta_scale"]*Th


# =========================
# 5. Visualization
# =========================

def show_phone_heat(X,Y,T_grid,t_val):
    Xm=X*1000;Ym=Y*1000;fig,ax=plt.subplots(figsize=(5.5,10),dpi=180)
    ax.set_facecolor("white");cr=12.0
    x_min,x_max=Xm.min(),Xm.max();y_min,y_max=Ym.min(),Ym.max()
    pw,ph=x_max-x_min,y_max-y_min
    im=ax.pcolormesh(Xm,Ym,T_grid,shading="auto",cmap="hot",vmin=298,vmax=325,zorder=5)
    ax.set_aspect("equal");ax.set_xlim(x_min,x_max);ax.set_ylim(y_min,y_max)
    ax.set_title(f"Phone Thermal Field (t={int(t_val)}s)",fontsize=14)

    outer=FancyBboxPatch((x_min,y_min),pw,ph,boxstyle=BoxStyle("Round",pad=0,rounding_size=cr),
                         linewidth=3,edgecolor="white",facecolor="none",zorder=10)
    ax.add_patch(outer)
    inner=FancyBboxPatch((x_min+4,y_min+4),pw-8,ph-8,boxstyle=BoxStyle("Round",pad=0,rounding_size=cr-2),
                         linewidth=2,edgecolor="white",facecolor="none",zorder=11)
    ax.add_patch(inner)

    lc="#0088ff";ax.add_patch(Rectangle((x_min+pw/2-16,y_min+ph/2-20+2),32,40,linewidth=1.6,edgecolor=lc,facecolor="none",linestyle="--",zorder=12))
    ax.add_patch(Rectangle((x_min+pw/2-10,y_max-4-12-10),20,12,linewidth=1.6,edgecolor=lc,facecolor="none",linestyle="--",zorder=12))
    ax.text(0.5,0.93,"Camera",color=lc,ha="center",fontsize=10,fontweight="bold",transform=ax.transAxes,zorder=20)
    ax.text(0.5,0.55,"SoC",color=lc,ha="center",fontsize=11,fontweight="bold",transform=ax.transAxes,zorder=20)

    cbar=fig.colorbar(im,ax=ax,fraction=0.035,pad=0.03);cbar.set_label("T (K)",fontsize=11)
    ax.set_xlabel("x (mm)");ax.set_ylabel("y (mm)");plt.tight_layout();plt.show()


def show_eval_result(model,device,cfg,params,history):
    if history and len(history):
        plt.figure(figsize=(7,4));plt.plot(history,linewidth=1);plt.yscale("log")
        plt.xlabel("Iter");plt.ylabel("Total Loss");plt.title("Training Loss")
        plt.grid(alpha=0.3);plt.tight_layout();plt.show()

    for t_val in [300.0,900.0,1800.0]:
        X,Y,T_grid=predict_grid_at_time(model,device,cfg,params,t_val)
        show_phone_heat(X,Y,T_grid,t_val)

    # Center temperature curve
    t_max=params["t_max"];t_line=np.linspace(0,t_max,200)
    soc=cfg["soc"];xc=soc["x_center"];yc=soc["y_center"]
    xca=np.full_like(t_line,xc);yca=np.full_like(t_line,yc)
    xyh=torch.tensor(np.stack([xca/params["Lx_scale"],yca/params["Ly_scale"],t_line/params["t_scale"]],1),
                     dtype=torch.float32,device=device)
    model.eval()
    with torch.no_grad():
        Tc=params["T_inf"]+params["theta_scale"]*model(xyh).cpu().numpy().reshape(-1)
    plt.figure(figsize=(7.5,4.5))
    plt.plot(t_line,Tc,lw=2,label="PINN T at SoC center")
    plt.axhline(params["T_inf"],ls="--",c="gray",label=f"T_inf={params['T_inf']:.0f} K")
    plt.xlabel("Time (s)");plt.ylabel("Temperature (K)");plt.legend();plt.grid(alpha=0.3)
    plt.title("Temperature at SoC Center vs Time");plt.tight_layout();plt.show()


# =========================
# 6. Eval entry
# =========================

def evaluate(model_path=MODEL_PATH,data_path=DATA_PATH):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    model,extras=load_model_package(model_path,device)
    cfg=extras.get("cfg") or load_or_default_config(data_path)
    params=build_physical_params(cfg)
    history=extras.get("history",[])

    print("Eval result");print(f"model_path: {model_path}")
    print(f"Lx={params['Lx']:.3f}m, Ly={params['Ly']:.3f}m, t_max={params['t_max']:.0f}s")
    if history and len(history): print(f"final_loss: {history[-1]:.6e}")

    show_eval_result(model,device,cfg,params,history)
    return {"model":model,"params":params}


eval_result = evaluate()
