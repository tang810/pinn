"""
Notebook-friendly train file for power_system_PINN.

Online Jupyter usage:
1. Fill DATA_HASH (optional) and MODEL_HASH.
2. Run this whole file/cell. It saves trained_model_power_system.pt to
   ~/minio/Resource/{MODEL_HASH}/ for later public-asset upload.

Physics: PINN for 10-node power system frequency dynamics with 3 constraints:
         1) Rotor motion: M·df/dt = ΔP_m - ΔP_e
         2) Power balance across neighboring nodes
         3) Power transfer via line reactance
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
import torch.optim as optim


# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = "4ce4c950b82341ec8ba49f1f17607336"
MODEL_HASH = "4ce4c950b82341ec8ba49f1f17607336"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/minio/Resource/{MODEL_HASH}/trained_model_power_system.pt")
QUICK_MODE = True
QUICK_EPOCHS = 400
QUICK_BATCH_SIZE = 256
QUICK_HIDDEN_DIM = 32
QUICK_NUM_LAYERS = 2


# =========================
# 2. Model definition
# =========================

class PINN(nn.Module):
    def __init__(self, num_nodes=10, hidden_dim=64, num_layers=4):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_layer = nn.Linear(2, hidden_dim)
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.hidden_layers.append(nn.Tanh())
        self.output_layer = nn.Linear(hidden_dim, 3)
        device = get_device()
        self.M = torch.tensor(6.0, device=device)
        self.X_line = torch.tensor(0.1, device=device)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, t, x):
        inputs = torch.stack([t.unsqueeze(1) if t.dim()==1 else t,
                              x.unsqueeze(1) if x.dim()==1 else x], dim=1).squeeze(-1)
        if inputs.dim() == 1: inputs = inputs.unsqueeze(0)
        out = self.input_layer(inputs)
        out = torch.tanh(out)
        for layer in self.hidden_layers:
            out = layer(out)
        out = self.output_layer(out)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]

    def compute_derivatives(self, t, x):
        t.requires_grad_(True); x.requires_grad_(True)
        delta_f, delta_P_e, delta_P_m = self.forward(t, x)
        df_dt = torch.autograd.grad(delta_f, t, torch.ones_like(delta_f), create_graph=True, retain_graph=True)[0]
        delta_theta = torch.autograd.grad(delta_f.sum(), t, create_graph=True, retain_graph=True)[0]
        return delta_f, delta_P_e, delta_P_m, df_dt, delta_theta


# =========================
# 3. Data generation
# =========================

def generate_training_data(num_samples=10000, num_nodes=10, t_max=1.0):
    device = get_device()
    t = torch.rand(num_samples, device=device) * t_max
    x = torch.randint(0, num_nodes, (num_samples,), device=device).float()
    return t, x


# =========================
# 4. Loss computation
# =========================

def compute_loss(model, t, x, num_nodes=10):
    device = next(model.parameters()).device
    delta_f, delta_P_e, delta_P_m, df_dt, delta_theta = model.compute_derivatives(t, x)

    # 1) Rotor motion equation
    eq1_loss = torch.mean((model.M * df_dt - (delta_P_m - delta_P_e)) ** 2)

    eq2_loss = torch.tensor(0.0, device=device)
    eq3_loss = torch.tensor(0.0, device=device)
    if not QUICK_MODE:
        batch_size = t.shape[0]
        for i in range(batch_size):
            node = x[i]
            if 0 < node < num_nodes - 1:
                prev_mask = (x == (node - 1)) & (torch.abs(t - t[i]) < 1e-6)
                next_mask = (x == (node + 1)) & (torch.abs(t - t[i]) < 1e-6)
                if torch.any(prev_mask) and torch.any(next_mask):
                    dP_prev = (delta_theta[i] - delta_theta[prev_mask].mean()) / model.X_line
                    dP_next = (delta_theta[next_mask].mean() - delta_theta[i]) / model.X_line
                    eq2_loss += ((delta_P_e[i] - (dP_next + dP_prev)) ** 2)
            if node < num_nodes - 1:
                nm = (x == (node + 1)) & (torch.abs(t - t[i]) < 1e-6)
                if torch.any(nm):
                    eq3_loss += ((delta_theta[i] - delta_theta[nm].mean()) / model.X_line) ** 2
        eq2_loss = eq2_loss / max(batch_size, 1)
        eq3_loss = eq3_loss / max(batch_size, 1)

    # Boundary conditions
    bc_loss = torch.tensor(0.0, device=device)
    im = (t < 0.01)
    if torch.any(im): bc_loss += torch.mean(delta_f[im] ** 2)
    dm = (t > 0.09) & (t < 0.11) & (x < 0.1)
    if torch.any(dm): bc_loss += torch.mean((delta_P_m[dm] - 0.1) ** 2)
    em = (x > num_nodes - 1.1) & (x < num_nodes - 0.9)
    if torch.any(em):
        df_dx = torch.autograd.grad(delta_f, x, torch.ones_like(delta_f), create_graph=True, retain_graph=True)[0]
        bc_loss += torch.mean(df_dx[em] ** 2)

    total = 1.0 * eq1_loss + 1.0 * eq2_loss + 1.0 * eq3_loss + 2.0 * bc_loss
    return total, eq1_loss, eq2_loss, eq3_loss, bc_loss


# =========================
# 5. Visualization
# =========================

def show_train_result(model, losses, num_nodes=10, t_max=1.0, num_time_steps=200):
    if losses:
        plt.figure(figsize=(7,4))
        plt.plot([l["Total Loss"] for l in losses]); plt.yscale("log")
        plt.xlabel("Hundreds of Epochs"); plt.ylabel("Total Loss")
        plt.title("Training Loss"); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

    device = next(model.parameters()).device
    t = torch.linspace(0, t_max, num_time_steps, device=device)
    x = torch.linspace(0, num_nodes - 1, num_nodes, device=device)
    tg, xg = torch.meshgrid(t, x, indexing="ij")
    tf, xf = tg.flatten(), xg.flatten()
    model.eval()
    with torch.no_grad():
        df, _, _ = model(tf, xf)
    df = df.cpu().numpy().reshape(tg.shape)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im = axes[0].imshow(df.T, aspect="auto", extent=[0, t_max, 0, num_nodes-1],
                         origin="lower", cmap="coolwarm")
    axes[0].set_title("Frequency Increment Δf (Hz)"); axes[0].set_xlabel("Time (s)"); axes[0].set_ylabel("Node")
    plt.colorbar(im, ax=axes[0])
    for i in range(num_nodes):
        axes[1].plot(t.cpu(), df[:, i], label=f"Node {i}")
    axes[1].set_title("Node Frequencies vs Time"); axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Δf (Hz)"); axes[1].legend(fontsize=7, ncol=2); axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.show()


# =========================
# 6. Train entry
# =========================

def train(data_path=DATA_PATH, model_path=MODEL_PATH, num_nodes=10, num_epochs=3000,
          batch_size=2048, t_max=1.0, lr=0.001):
    t0 = time.time()
    device = get_device()
    print(f"device={device}")

    hidden_dim = QUICK_HIDDEN_DIM if QUICK_MODE else 64
    num_layers = QUICK_NUM_LAYERS if QUICK_MODE else 4
    if QUICK_MODE:
        num_epochs = min(num_epochs, QUICK_EPOCHS)
        batch_size = min(batch_size, QUICK_BATCH_SIZE)
        print("QUICK_MODE enabled: reduced epochs, batch size, and network width.")

    model = PINN(num_nodes=num_nodes, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=500)
    losses = []

    print(f"Training {num_epochs} epochs ({num_nodes} nodes, hidden={hidden_dim}, layers={num_layers}, batch={batch_size})...")
    for ep in range(num_epochs):
        t, x = generate_training_data(batch_size, num_nodes, t_max)
        total_loss, eq1, eq2, eq3, bc = compute_loss(model, t, x, num_nodes)
        optimizer.zero_grad(); total_loss.backward(); optimizer.step()
        scheduler.step(total_loss)

        log_every = 50 if QUICK_MODE else 100
        if ep == 0 or (ep + 1) % log_every == 0 or ep == num_epochs - 1:
            losses.append({"Total Loss": total_loss.item(), "Rotor": eq1.item(),
                           "PowerBal": eq2.item(), "PowerTrans": eq3.item(), "BC": bc.item()})
            print(f"Epoch {ep+1:5d}/{num_epochs} | Total={total_loss.item():.6f}")

    elapsed = time.time() - t0
    model_path = os.path.expanduser(model_path)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    ckpt = {"project": "power_system_PINN", "state_dict": model.state_dict(),
            "num_nodes": num_nodes, "t_max": t_max, "history": losses,
            "hidden_dim": hidden_dim, "num_layers": num_layers}
    torch.save(ckpt, model_path)

    print("Training finished"); print(f"model saved to: {model_path}")
    print(f"elapsed: {elapsed:.1f}s"); print(f"final loss: {losses[-1]['Total Loss']:.6e}" if losses else "")
    show_train_result(model, losses, num_nodes, t_max)
    return {"model": model, "model_path": model_path, "losses": losses}


train_result = train()
