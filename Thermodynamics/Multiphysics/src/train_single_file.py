"""
Notebook-friendly train file for Multiphysics (PINNsformer).

Online Jupyter usage:
1. Upload data_7246.txt as a public asset.
2. Fill DATA_HASH and MODEL_HASH.
3. Run this whole file/cell. It saves trained_model_multiphysics.pt to
   ~/minio/Resource/{MODEL_HASH}/ for later public-asset upload.

Physics: PINNsformer (Transformer) for thermo-elastic multiphysics.
         Input (x,y,z,t) → Output T (temperature).
         Heat equation + elastic equilibrium + thermal stress.

Note: imports from src/ for complex geometry and Transformer modules.
"""

import os
import time

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.model import PINNsformer
from src.data_loader import get_training_data, get_evaluation_data
from src.physics import ThermoElasticPINNLoss



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
# 1. Parameters
# =========================

DATA_HASH = ""
MODEL_HASH = ""

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/minio/Resource/{MODEL_HASH}/trained_model_multiphysics.pt")


# =========================
# 2. Train entry
# =========================

def generate_small_data(path):
    """Generate a small synthetic data_7246.txt if missing."""
    import pandas as pd
    rng = np.random.default_rng(42); n = 2000
    x = rng.uniform(0, 1, n); y = rng.uniform(0, 1, n)
    z = rng.uniform(0, 1, n); t = rng.uniform(0, 1, n)
    T = 273.15 + 30*np.exp(-((x-0.5)**2+(y-0.5)**2+(z-0.5)**2)/0.1)*(1+0.1*np.sin(2*np.pi*t))
    df = pd.DataFrame({"x": x, "y": y, "z": z, "t": t, "T": T})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep=" ", index=False, header=False, float_format="%.6f")
    print(f"Generated small dataset: {len(df)} points, {os.path.getsize(path)/1024:.0f} KB")


def train(data_path=DATA_PATH, model_path=MODEL_PATH,
          d_model=128, d_hidden=128, N=1, heads=2, epochs=50, lr=1.0,
          res_points=5000, bc_points=2000, ic_points=2000,
          rho=7800.0, Cp=500.0, k=45.0, E=210e9, nu=0.3, alpha=1.2e-5, T0=273.15):
    t0 = time.time()

    device = get_device()
    print(f"device={device}")

    # Resolve data path, generate if missing
    data_path = os.path.expanduser(data_path) if data_path else "data/data_7246.txt"
    if not os.path.isfile(data_path):
        if os.path.isdir(data_path):
            data_path = os.path.join(data_path, "data_7246.txt")
        if not os.path.isfile(data_path):
            generate_small_data(data_path)

    class Args:
        pass
    args = Args()
    args.data_path = data_path
    args.device = str(device)
    args.d_model = d_model; args.d_hidden = d_hidden; args.N = N; args.heads = heads
    args.res_points = res_points; args.bc_points = bc_points; args.ic_points = ic_points
    args.rho = rho; args.Cp = Cp; args.k = k; args.E = E; args.nu = nu
    args.alpha = alpha; args.T0 = T0; args.T_amb = 300.0; args.eps = 0.9
    args.sigma = 5.670374419e-8; args.q_sum = 0.0
    args.w_heat = 1.0; args.w_elastic = 0.1; args.w_ic = 1.0; args.w_reg = 1e-4; args.w_bc = 1.0
    args.geometry = "csg"; args.geometry_backend = "sym"; args.sampler = "lhs"
    args.time_steps = 10; args.step_size = 0.01
    args.bc_outer_type = "dirichlet"; args.bc_outer_value = 273.15
    args.bc_hole_type = "neumann"; args.bc_hole_value = 0.0
    args.csg_bound = 1.05; args.csg_hole_ratio = 0.60
    args.csg_chunk = 300000; args.csg_max_rounds = 50; args.sdf_eps = 1e-3
    args.proj_iters = 2; args.interior_margin = 0.0

    print(f"data={args.data_path}")
    res_seq, bc_seq, bc_normals, bc_tag, ic_seq, ic_T = get_training_data(args, device)

    model = PINNsformer(d_model=d_model, d_hidden=d_hidden, N=N, heads=heads).to(device)
    loss_fn = ThermoElasticPINNLoss(args, device)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=20,
                                   history_size=50, line_search_fn="strong_wolfe")
    history = []

    print(f"Training {epochs} epochs (PINNsformer, d_model={d_model})...")
    for ep in range(epochs):
        def closure():
            optimizer.zero_grad()
            losses = loss_fn(model, res_seq, bc_seq, bc_normals, bc_tag, ic_seq, ic_T)
            losses["total"].backward()
            return losses["total"]
        loss_val = optimizer.step(closure)
        with torch.no_grad():
            losses = loss_fn(model, res_seq, bc_seq, bc_normals, bc_tag, ic_seq, ic_T)
        history.append(float(losses["total"].cpu()))
        print(f"Epoch {ep+1:4d}/{epochs} | total={history[-1]:.3e} heat={float(losses.get('heat',0)):.3e}")

    model_path = os.path.expanduser(model_path)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "history": history,
                "d_model": d_model, "d_hidden": d_hidden, "N": N, "heads": heads}, model_path)

    print("Training finished"); print(f"model saved to: {model_path}")
    print(f"elapsed: {time.time() - t0:.1f}s")

    if history:
        plt.figure(figsize=(7,4)); plt.plot(history); plt.yscale("log")
        plt.xlabel("Epoch"); plt.ylabel("Total Loss"); plt.title("Training Loss")
        plt.grid(alpha=0.3); plt.tight_layout(); plt.show()
    return {"model": model, "model_path": model_path, "history": history}


train_result = train()
