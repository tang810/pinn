"""
Notebook-friendly train file for Helmholtz PINN.

Online Jupyter usage:
1. (Optional) Upload simul_placeholder.npz as a public asset.
2. Fill DATA_HASH and MODEL_HASH.
3. Run this whole file/cell. It saves trained_model_helmholtz.pt to
   ~/minio/Resource/{MODEL_HASH}/ for later public-asset upload.

Physics: 1D Helmholtz equation: d2u/dx2 + k^2 * u = 0, x in [-1,1].
         BCs: u(-1)=0, u(1)=1. Trainable parameter k (wavenumber).

If no data uploaded, generates collocation points on the fly.
"""

import os
import time

import numpy as np
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

DATA_HASH = "50da6cb8def9402ea3657b04e47d3d1c"
MODEL_HASH = "50da6cb8def9402ea3657b04e47d3d1c"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/minio/Resource/{MODEL_HASH}/trained_model_helmholtz.pt")


# =========================
# 2. Model definition
# =========================

class HelmholtzPINN(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=4, k=3.0):
        super().__init__()
        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, 1)]
        self.net = nn.Sequential(*layers)
        self.k = nn.Parameter(torch.tensor(float(k)))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# =========================
# 3. Data generation
# =========================

def save_simul_placeholder(path, n_domain=256):
    """Generate and save a small .npz placeholder for upload."""
    x_domain = np.random.uniform(-1, 1, (n_domain, 1)).astype(np.float32)
    x_boundary = np.array([[-1.0], [1.0]], dtype=np.float32)
    u_boundary = np.array([[0.0], [1.0]], dtype=np.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, x_domain=x_domain, x_boundary=x_boundary, u_boundary=u_boundary)
    return x_domain, x_boundary, u_boundary


def load_or_generate(data_path, n_domain=256):
    """Load uploaded simul_placeholder.npz, or generate on the fly if missing."""
    data_path = os.path.expanduser(data_path) if data_path else ""
    if data_path and os.path.isfile(data_path):
        d = np.load(data_path)
        if 'x_domain' in d:
            return torch.tensor(d['x_domain'], dtype=torch.float32), torch.tensor(d['x_boundary'], dtype=torch.float32), torch.tensor(d['u_boundary'], dtype=torch.float32)
    if data_path and os.path.isdir(data_path):
        candidate = os.path.join(data_path, "simul_placeholder.npz")
        if os.path.isfile(candidate):
            d = np.load(candidate)
            return torch.tensor(d['x_domain'], dtype=torch.float32), torch.tensor(d['x_boundary'], dtype=torch.float32), torch.tensor(d['u_boundary'], dtype=torch.float32)
    # Generate on the fly
    x_domain = torch.rand(n_domain, 1) * 2.0 - 1.0
    x_boundary = torch.tensor([[-1.0], [1.0]], dtype=torch.float32)
    u_boundary = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    return x_domain, x_boundary, u_boundary


# =========================
# 4. Loss computation
# =========================

def compute_loss(model, x_domain, x_boundary, u_boundary):
    x_domain = x_domain.clone().detach().requires_grad_(True)
    u = model(x_domain)
    du_dx = torch.autograd.grad(u, x_domain, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    d2u_dx2 = torch.autograd.grad(du_dx, x_domain, torch.ones_like(du_dx), create_graph=True, retain_graph=True)[0]
    loss_pde = torch.mean((d2u_dx2 + model.k**2 * u)**2)
    u_b = model(x_boundary)
    loss_bc = torch.mean((u_b - u_boundary)**2)
    return loss_pde + loss_bc, loss_pde.detach(), loss_bc.detach()


# =========================
# 5. Visualization
# =========================

def show_train_result(model, history, device):
    if history and history["total"]:
        plt.figure(figsize=(7,4)); plt.plot(history["total"]); plt.yscale("log")
        plt.xlabel("Epoch"); plt.ylabel("Total Loss"); plt.title("Training Loss")
        plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

    x = torch.linspace(-1, 1, 200, device=device).reshape(-1, 1)
    model.eval()
    with torch.no_grad():
        u = model(x).cpu().numpy()
    plt.figure(figsize=(7, 4))
    plt.plot(x.cpu().numpy(), u, 'b-', lw=2)
    plt.scatter([-1, 1], [0, 1], c='r', s=50, zorder=5, label="BCs")
    plt.xlabel("x"); plt.ylabel("u(x)"); plt.title(f"Helmholtz PINN (k={model.k.item():.4f})")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()
    print(f"Identified k = {model.k.item():.6f}")


# =========================
# 6. Train entry
# =========================

def train(data_path=DATA_PATH, model_path=MODEL_PATH, k=3.0, hidden_dim=64, num_layers=4,
          n_domain=256, epochs=500, lr=1e-3, log_every=50):
    t0 = time.time()
    device = get_device()
    print(f"device={device}")

    model = HelmholtzPINN(hidden_dim=hidden_dim, num_layers=num_layers, k=k).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = {"total": [], "pde": [], "bc": []}

    print(f"Training {epochs} epochs (k0={k})...")
    for ep in range(epochs):
        xd, xb, ub = load_or_generate(data_path, n_domain)
        xd, xb, ub = xd.to(device), xb.to(device), ub.to(device)
        total, pde, bc = compute_loss(model, xd, xb, ub)
        opt.zero_grad(); total.backward(); opt.step()
        hist["total"].append(float(total.cpu())); hist["pde"].append(float(pde.cpu())); hist["bc"].append(float(bc.cpu()))
        if (ep+1) % log_every == 0 or ep == 0:
            print(f"Epoch {ep+1:5d}/{epochs} | total={hist['total'][-1]:.3e} pde={hist['pde'][-1]:.3e} bc={hist['bc'][-1]:.3e} k={model.k.item():.4f}")

    model_path = os.path.expanduser(model_path)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "k": float(model.k.detach().cpu()),
                "hidden_dim": hidden_dim, "num_layers": num_layers}, model_path)

    print("Training finished"); print(f"model saved to: {model_path}")
    print(f"elapsed: {time.time() - t0:.1f}s"); print(f"final k = {model.k.item():.6f}")
    show_train_result(model, hist, device)
    return {"model": model, "model_path": model_path, "k": model.k.item()}


train_result = train()
