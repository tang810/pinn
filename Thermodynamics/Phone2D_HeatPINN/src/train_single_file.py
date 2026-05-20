"""
Notebook-friendly train file for Phone2D_HeatPINN.

Online Jupyter usage:
1. (Optional) Upload phone_thermal_case.yaml as a public asset.
2. Fill DATA_HASH and MODEL_HASH.
3. Run this whole file/cell. It saves trained_model_phone_thermal.pt to
   ~/minio/Resource/{MODEL_HASH}/ for later public-asset upload.

Physics: 2D heat conduction PINN for smartphone SoC heating with convective BCs.
"""

import os
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, BoxStyle

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
import yaml


# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = "a4ee35a2e2114e90924a67429bf88b5c"
MODEL_HASH = "a4ee35a2e2114e90924a67429bf88b5c"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/minio/Resource/{MODEL_HASH}/trained_model_phone_thermal.pt")
QUICK_MODE = True
QUICK_N_ITERS = 800
QUICK_PRINT_EVERY = 100


# =========================
# 2. Default config (overridable by YAML from data_path)
# =========================

DEFAULT_CFG = {
    "geometry": {"Lx": 0.07, "Ly": 0.15, "t_max": 1800.0},
    "material": {"rho": 2500.0, "cp": 900.0, "k": 5.0},
    "environment": {"T_inf": 298.0, "h_env": 5.0},
    "soc": {"x_center": 0.035, "y_center": 0.075, "width": 0.01, "height": 0.01,
            "q0": 3e6, "tau_q": 100.0, "use_slow_source": True},
    "nondim": {"theta_scale": 10.0},
    "sampling": {"N_r": 3000, "N_ic": 2000, "N_ic2": 1000, "N_bc_edge": 500,
                 "use_extra_t0_samples": False},
    "training": {"n_iters": 30000, "print_every": 1000, "lr": 1e-3,
                 "w_pde": 1.0, "w_ic": 20.0, "w_ic2": 10.0, "w_bc": 1.0},
}


def load_or_default_config(data_path):
    data_path = os.path.expanduser(data_path) if data_path else ""
    yaml_path = None
    if data_path and os.path.isfile(data_path) and data_path.endswith((".yaml", ".yml")):
        yaml_path = data_path
    elif data_path and os.path.isdir(data_path):
        for f in sorted(os.listdir(data_path)):
            if f.endswith((".yaml", ".yml")):
                yaml_path = os.path.join(data_path, f); break
    if yaml_path:
        with open(yaml_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    return DEFAULT_CFG


def apply_quick_mode(cfg):
    if not QUICK_MODE:
        return cfg
    cfg = {
        key: (value.copy() if isinstance(value, dict) else value)
        for key, value in cfg.items()
    }
    cfg.setdefault("sampling", {})
    cfg.setdefault("training", {})
    cfg["sampling"].update({
        "N_r": min(int(cfg["sampling"].get("N_r", 3000)), 500),
        "N_ic": min(int(cfg["sampling"].get("N_ic", 2000)), 300),
        "N_ic2": min(int(cfg["sampling"].get("N_ic2", 1000)), 0),
        "N_bc_edge": min(int(cfg["sampling"].get("N_bc_edge", 500)), 100),
        "use_extra_t0_samples": False,
    })
    cfg["training"].update({
        "n_iters": min(int(cfg["training"].get("n_iters", QUICK_N_ITERS)), QUICK_N_ITERS),
        "print_every": min(int(cfg["training"].get("print_every", QUICK_PRINT_EVERY)), QUICK_PRINT_EVERY),
    })
    return cfg


# =========================
# 3. Physics parameter builder
# =========================

def build_physical_params(cfg):
    rho=cfg["material"]["rho"];cp=cfg["material"]["cp"];k=cfg["material"]["k"]
    Lx=cfg["geometry"]["Lx"];Ly=cfg["geometry"]["Ly"];t_max=cfg["geometry"]["t_max"]
    T_inf=cfg["environment"]["T_inf"];h_env=cfg["environment"]["h_env"]
    theta_scale=cfg["nondim"]["theta_scale"]
    alpha=k/(rho*cp)
    alpha_x_hat=alpha*t_max/(Lx**2);alpha_y_hat=alpha*t_max/(Ly**2)
    Bi_x=h_env*Lx/k;Bi_y=h_env*Ly/k;q_to_theta=1.0/(rho*cp)
    return dict(rho=rho,cp=cp,k=k,alpha=alpha,Lx=Lx,Ly=Ly,t_max=t_max,
                T_inf=T_inf,h_env=h_env,theta_scale=theta_scale,
                Lx_scale=Lx,Ly_scale=Ly,t_scale=t_max,
                alpha_x_hat=alpha_x_hat,alpha_y_hat=alpha_y_hat,
                Bi_x=Bi_x,Bi_y=Bi_y,q_to_theta=q_to_theta)


# =========================
# 4. Heat source & sampling
# =========================

def q_source_theta(x,y,t,cfg,params):
    soc=cfg["soc"];xc=soc["x_center"];yc=soc["y_center"]
    w=soc["width"];h=soc["height"];q0=soc["q0"];tau=soc["tau_q"]
    mask=((torch.abs(x-xc)<=w/2)&(torch.abs(y-yc)<=h/2)).float()
    ft=(1.0-torch.exp(-t/tau)) if soc.get("use_slow_source",True) else torch.ones_like(t)
    return q0*mask*ft*params["q_to_theta"]


def sample_pde_points(N_r,cfg,params):
    Lx=params["Lx"];Ly=params["Ly"];t_max=params["t_max"]
    soc=cfg["soc"];xc=soc["x_center"];yc=soc["y_center"];w=soc["width"];h=soc["height"]
    Ne=int(0.6*N_r);Nf=N_r-Ne
    xe=torch.rand(Ne,1)*Lx;ye=torch.rand(Ne,1)*Ly;te=torch.rand(Ne,1)*(0.3*t_max)
    xf=torch.rand(Nf,1)*Lx;yf=torch.rand(Nf,1)*Ly;tf=torch.rand(Nf,1)*t_max
    xr=torch.cat([xe,xf],0);yr=torch.cat([ye,yf],0);tr=torch.cat([te,tf],0)
    Nh=N_r//2;idx=torch.randperm(N_r)[:Nh]
    mx,my=w*0.5,h*0.5
    xs=xc+(torch.rand(Nh,1)-0.5)*(w+2*mx);ys=yc+(torch.rand(Nh,1)-0.5)*(h+2*my)
    xs=torch.clamp(xs,0,Lx);ys=torch.clamp(ys,0,Ly)
    xr[idx,:]=xs;yr[idx,:]=ys
    return xr,yr,tr


def sample_points(cfg,params):
    sp=cfg["sampling"];Lx=params["Lx"];Ly=params["Ly"];t_max=params["t_max"]
    xr,yr,tr=sample_pde_points(sp["N_r"],cfg,params)
    xi=torch.rand(sp["N_ic"],1)*Lx;yi=torch.rand(sp["N_ic"],1)*Ly;ti=torch.zeros(sp["N_ic"],1)
    if sp.get("use_extra_t0_samples",False) and sp["N_ic2"]>0:
        xi2=torch.rand(sp["N_ic2"],1)*Lx;yi2=torch.rand(sp["N_ic2"],1)*Ly
        ti2=torch.rand(sp["N_ic2"],1)*(0.02*t_max)
    else: xi2=torch.zeros(0,1);yi2=torch.zeros(0,1);ti2=torch.zeros(0,1)
    Nb=sp["N_bc_edge"]
    return (xr,yr,tr, xi,yi,ti, xi2,yi2,ti2,
            torch.zeros(Nb,1),torch.rand(Nb,1)*Ly,torch.rand(Nb,1)*t_max,
            torch.full((Nb,1),Lx),torch.rand(Nb,1)*Ly,torch.rand(Nb,1)*t_max,
            torch.rand(Nb,1)*Lx,torch.zeros(Nb,1),torch.rand(Nb,1)*t_max,
            torch.rand(Nb,1)*Lx,torch.full((Nb,1),Ly),torch.rand(Nb,1)*t_max)


# =========================
# 5. Model definition
# =========================

class HeatPINN(nn.Module):
    def __init__(self,in_dim=3,hidden=64,n_layers=5):
        super().__init__()
        layers=[nn.Linear(in_dim,hidden),nn.Tanh()]
        for _ in range(n_layers-2):
            layers+=[nn.Linear(hidden,hidden),nn.Tanh()]
        layers+=[nn.Linear(hidden,1)]
        self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)


# =========================
# 6. Loss computation
# =========================

def compute_loss(model,device,batch,cfg,params):
    (xr,yr,tr,xi,yi,ti,xi2,yi2,ti2,
     xl,yl,tl,xr_bc,yr_bc,tr_bc,xb,yb,tb,xt,yt,tt)=batch
    Lx_s=params["Lx_scale"];Ly_s=params["Ly_scale"];t_s=params["t_scale"]
    th_s=params["theta_scale"];ax=params["alpha_x_hat"];ay=params["alpha_y_hat"]
    Bx=params["Bi_x"];By=params["Bi_y"]

    (xr,yr,tr,xi,yi,ti,xi2,yi2,ti2,xl,yl,tl,xr_bc,yr_bc,tr_bc,xb,yb,tb,xt,yt,tt)=(
        t.to(device) for t in (xr,yr,tr,xi,yi,ti,xi2,yi2,ti2,xl,yl,tl,xr_bc,yr_bc,tr_bc,xb,yb,tb,xt,yt,tt))

    # PDE residual
    xhr=(xr/Lx_s).requires_grad_(True);yhr=(yr/Ly_s).requires_grad_(True)
    thr=(tr/t_s).requires_grad_(True)
    u=model(torch.cat([xhr,yhr,thr],1))
    ut=torch.autograd.grad(u,thr,torch.ones_like(u),create_graph=True,retain_graph=True)[0]
    ux=torch.autograd.grad(u,xhr,torch.ones_like(u),create_graph=True,retain_graph=True)[0]
    uy=torch.autograd.grad(u,yhr,torch.ones_like(u),create_graph=True,retain_graph=True)[0]
    uxx=torch.autograd.grad(ux,xhr,torch.ones_like(ux),create_graph=True,retain_graph=True)[0]
    uyy=torch.autograd.grad(uy,yhr,torch.ones_like(uy),create_graph=True,retain_graph=True)[0]
    qh=q_source_theta(xr,yr,tr,cfg,params)*(t_s/th_s)
    loss_pde=torch.mean((ut-ax*uxx-ay*uyy-qh)**2)

    # IC
    loss_ic=torch.mean(model(torch.cat([xi/Lx_s,yi/Ly_s,ti/t_s],1))**2)
    loss_ic2=torch.mean(model(torch.cat([xi2/Lx_s,yi2/Ly_s,ti2/t_s],1))**2) if xi2.numel()>0 else torch.tensor(0.)

    # BC: 4 convective boundaries
    def bc_loss(xh,yh,th,sign):
        xg=xh.requires_grad_(True);u=model(torch.cat([xg,yh,th],1))
        du=torch.autograd.grad(u,xg,torch.ones_like(u),create_graph=True,retain_graph=True)[0]
        return torch.mean((sign*du-Bx*u)**2)
    def bc_loss_y(xh,yh,th,sign):
        yg=yh.requires_grad_(True);u=model(torch.cat([xh,yg,th],1))
        du=torch.autograd.grad(u,yg,torch.ones_like(u),create_graph=True,retain_graph=True)[0]
        return torch.mean((sign*du-By*u)**2)
    loss_bc=(bc_loss(xl/Lx_s,yl/Ly_s,tl/t_s,+1)+bc_loss(xr_bc/Lx_s,yr_bc/Ly_s,tr_bc/t_s,-1)+
             bc_loss_y(xb/Lx_s,yb/Ly_s,tb/t_s,+1)+bc_loss_y(xt/Lx_s,yt/Ly_s,tt/t_s,-1))

    tr=cfg["training"];loss=tr["w_pde"]*loss_pde+tr["w_ic"]*loss_ic+tr["w_ic2"]*loss_ic2+tr["w_bc"]*loss_bc
    return loss,{"pde":loss_pde.item(),"ic":loss_ic.item(),"ic2":loss_ic2.item(),"bc":loss_bc.item()}


# =========================
# 7. Visualization (train)
# =========================

def show_train_result(model,device,cfg,params,history):
    if len(history):
        plt.figure(figsize=(7,4))
        plt.plot(history,linewidth=1);plt.yscale("log")
        plt.xlabel("Iter");plt.ylabel("Total Loss");plt.title("Training Loss")
        plt.grid(alpha=0.3);plt.tight_layout();plt.show()

    # phone heat snapshot at 900s
    Lx=params["Lx"];Ly=params["Ly"];t_val=900.0;nx=ny=60
    xl=np.linspace(0,Lx,nx);yl=np.linspace(0,Ly,ny);X,Y=np.meshgrid(xl,yl,indexing="xy")
    xf=X.reshape(-1,1);yf=Y.reshape(-1,1);tf=np.full_like(xf,t_val)
    xyh=torch.tensor(np.concatenate([xf/Lx,yf/Ly,tf/params["t_scale"]],1),dtype=torch.float32,device=device)
    model.eval()
    with torch.no_grad():
        Th=model(xyh).cpu().numpy().reshape(ny,nx)
    T=params["T_inf"]+params["theta_scale"]*Th

    Xm=X*1000;Ym=Y*1000;fig,ax=plt.subplots(figsize=(5,8),dpi=180)
    im=ax.pcolormesh(Xm,Ym,T,shading="auto",cmap="hot",vmin=298,vmax=320,zorder=5)
    ax.set_aspect("equal");ax.set_xlim(Xm.min(),Xm.max());ax.set_ylim(Ym.min(),Ym.max())
    ax.set_title(f"Phone Thermal Field (t={int(t_val)}s)");ax.set_xlabel("x (mm)");ax.set_ylabel("y (mm)")
    plt.colorbar(im,ax=ax,fraction=0.04,label="T (K)");plt.tight_layout();plt.show()


# =========================
# 8. Train entry
# =========================

def train(data_path=DATA_PATH,model_path=MODEL_PATH,n_iters=None,print_every=None):
    t0=time.time();device=get_device()
    print(f"device={device}")
    cfg=apply_quick_mode(load_or_default_config(data_path));params=build_physical_params(cfg)
    tr=cfg["training"]
    if n_iters is None: n_iters=tr["n_iters"]
    if print_every is None: print_every=tr["print_every"]

    if QUICK_MODE:
        print("QUICK_MODE enabled: reduced samples and iterations for faster notebook training.")

    hidden=32 if QUICK_MODE else 64
    n_layers=4 if QUICK_MODE else 5
    model=HeatPINN(hidden=hidden,n_layers=n_layers).to(device);opt=torch.optim.Adam(model.parameters(),lr=tr["lr"])
    batch=sample_points(cfg,params);history=[]

    print(f"Training {n_iters} iters (Lx={params['Lx']},Ly={params['Ly']},t_max={params['t_max']})...")
    for it in range(1,n_iters+1):
        opt.zero_grad();loss,d=compute_loss(model,device,batch,cfg,params)
        loss.backward();opt.step();history.append(float(loss.detach().cpu()))
        if it==1 or it%print_every==0 or it==n_iters:
            print(f"Iter {it:5d}/{n_iters} | loss={history[-1]:.3e} | pde={d['pde']:.3e} ic={d['ic']:.3e} ic2={d['ic2']:.3e} bc={d['bc']:.3e}")

    model_path=os.path.expanduser(model_path)
    os.makedirs(os.path.dirname(model_path),exist_ok=True)
    ckpt={"project":"Phone2D_HeatPINN","state_dict":model.state_dict(),"history":history,
          "cfg":cfg,"hidden":hidden,"n_layers":n_layers}
    torch.save(ckpt,model_path)

    print("Training finished");print(f"model saved to: {model_path}")
    print(f"elapsed: {time.time()-t0:.1f}s");print(f"final loss: {history[-1]:.6e}")
    show_train_result(model,device,cfg,params,history)
    return {"model":model,"model_path":model_path,"history":history}


train_result = train()
