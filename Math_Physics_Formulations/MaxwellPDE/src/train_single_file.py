"""
Notebook-friendly train file for 3D frequency-domain Maxwell PINN.

Online Jupyter usage:
1. Upload training_data.npy as a public dataset asset.
2. Fill DATA_HASH and MODEL_HASH.
3. Run this whole file/cell. It saves trained_model_maxwell_pde.pt to
   ~/minio/Resource/{MODEL_HASH}/ for later public-asset upload.

Physics: curl(curl(E)/mu) - epsilon * omega^2 * E = i * omega * mu * J
Network: 4 inputs [x,y,z,omega] -> 6 outputs [Ex_r,Ey_r,Ez_r,Ex_i,Ey_i,Ez_i]
"""

import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = "00acab00eb074642a4b90f32d3bb561e"
MODEL_HASH = "00acab00eb074642a4b90f32d3bb561e"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/minio/Resource/{MODEL_HASH}/trained_model_maxwell_pde.pt")

EPOCHS = 80
PRINT_EVERY = 20
SEED = 42

BATCH_SIZE = 64
MAX_TRAIN_POINTS = 512
HIDDEN_DIM = 64
N_LAYERS = 2
LR = 5e-4
LAMBDA_PDE = 1.0
LAMBDA_BC = 10.0


# =========================
# 2. Utilities
# =========================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
# 3. Data loading
# =========================

def generate_small_data(n_points=512):
    """Generate small synthetic collocation data [x,y,z,omega]."""
    rng = np.random.default_rng(42)
    x = rng.uniform(-1, 1, n_points)
    y = rng.uniform(-1, 1, n_points)
    z = rng.uniform(-1, 1, n_points)
    omega = rng.uniform(0.5, 5.0, n_points)
    return np.column_stack([x, y, z, omega]).astype(np.float32)


def resolve_data_path(data_path):
    if not data_path:
        raise FileNotFoundError("DATA_PATH is empty. Fill DATA_HASH first.")
    data_path = os.path.expanduser(data_path)
    if os.path.isfile(data_path):
        return data_path
    candidate = os.path.join(data_path, "training_data.npy")
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(
        "Could not find training_data.npy. Upload it to the data asset folder. "
        f"Current DATA_PATH={data_path!r}"
    )


def load_or_generate_data(data_path, device="cpu"):
    """Load training_data.npy or generate synthetic data if not found."""
    try:
        data_path = resolve_data_path(data_path)
        X = np.load(data_path).astype(np.float32)
        print(f"Loaded data from {data_path}, shape={X.shape}")
    except (FileNotFoundError, OSError):
        print("No data file found, generating synthetic collocation points.")
        X = generate_small_data()
    if X.ndim != 2 or X.shape[1] != 4:
        raise ValueError(f"Data must have shape N x 4 [x,y,z,omega], got {X.shape}.")
    if len(X) > MAX_TRAIN_POINTS:
        rng = np.random.default_rng(SEED)
        idx = np.sort(rng.choice(len(X), size=MAX_TRAIN_POINTS, replace=False))
        X = X[idx]
        print(f"Using a quick subset: {len(X)} points")

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    # Do NOT set requires_grad here — each batch gets fresh grad via clone().detach().requires_grad_()
    return X_t


# =========================
# 4. Network
# =========================

class HighPrecisionPINN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=256, output_dim=6, n_layers=5):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for i in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU() if i % 2 == 0 else nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# =========================
# 5. Physics operators
# =========================

class Gradient:
    @staticmethod
    def compute_gradient(f, x):
        assert x.requires_grad
        if f.dim() == 1:
            f = f.unsqueeze(1)
        grads = []
        for i in range(f.shape[1]):
            gi = torch.autograd.grad(
                f[:, i].sum(), x,
                create_graph=True, retain_graph=True, allow_unused=True
            )[0]
            if gi is None:
                gi = torch.zeros_like(x)
            grads.append(gi.unsqueeze(1))
        return torch.cat(grads, dim=1)


class Curl:
    @staticmethod
    def curl(E, x):
        grad_E_full = Gradient.compute_gradient(E, x)
        grad_E = grad_E_full[:, :, :3]
        curl_E = torch.zeros_like(E)
        curl_E[:, 0] = grad_E[:, 2, 1] - grad_E[:, 1, 2]
        curl_E[:, 1] = grad_E[:, 0, 2] - grad_E[:, 2, 0]
        curl_E[:, 2] = grad_E[:, 1, 0] - grad_E[:, 0, 1]
        return curl_E

    @staticmethod
    def helmholtz_operator(E, x, epsilon, mu, omega):
        curl_E = Curl.curl(E, x)
        mu_exp = mu.expand_as(E)
        eps_exp = epsilon.expand_as(E)
        curl_mu_inv_curl_E = Curl.curl(curl_E / mu_exp, x)
        return curl_mu_inv_curl_E - eps_exp * omega**2 * E


# =========================
# 6. Material properties & source
# =========================

class MaterialProperties:
    @staticmethod
    def epsilon_r(x, interface_position=0.0):
        transition_width = 0.05
        sigma = 1.0 / (1.0 + torch.exp(-(x[:, 0:1] - interface_position) / transition_width))
        return 1.0 + 3.0 * (1.0 - sigma)

    @staticmethod
    def mu_r(x):
        return torch.ones_like(x[:, 0:1])


class SourceTerms:
    @staticmethod
    def plane_wave_source(x, omega, device):
        direction = torch.tensor([1.0, 0.0, 0.0], device=device)
        epsilon = MaterialProperties.epsilon_r(x)
        mu = MaterialProperties.mu_r(x)
        k = omega * torch.sqrt(epsilon * mu)
        phase = torch.sum(k * x[:, :3] * direction.unsqueeze(0), dim=1, keepdim=True)
        J = torch.zeros(x.shape[0], 3, device=device, dtype=torch.complex64)
        J[:, 0] = torch.exp(-1j * phase).squeeze()
        return J


# =========================
# 7. PDE residual & boundary conditions
# =========================

def compute_full_wave_residual(net, x, device):
    omega = x[:, 3:4]
    out = net(x)
    E_real = out[:, 0:3]
    E_imag = out[:, 3:6]

    epsilon = MaterialProperties.epsilon_r(x)
    mu = MaterialProperties.mu_r(x)

    helmholtz_real = Curl.helmholtz_operator(E_real, x, epsilon, mu, omega)
    helmholtz_imag = Curl.helmholtz_operator(E_imag, x, epsilon, mu, omega)

    J = SourceTerms.plane_wave_source(x, omega, device)
    J_real = J.real
    J_imag = J.imag

    residual_real = helmholtz_real + omega * mu.expand_as(E_real) * J_imag
    residual_imag = helmholtz_imag - omega * mu.expand_as(E_imag) * J_real

    return residual_real, residual_imag


def compute_pde_loss(net, x, device):
    res_r, res_i = compute_full_wave_residual(net, x, device)
    return res_r.square().mean() + res_i.square().mean()


def compute_bc_loss(net, x):
    out = net(x)
    E_real = out[:, 0:3]
    E_imag = out[:, 3:6]
    E_mag = torch.sqrt(E_real**2 + E_imag**2)

    loss = torch.tensor(0.0, device=x.device)

    mask_pec = (x[:, 0].abs() > 0.99)
    if mask_pec.any():
        E_tan = torch.cat([E_mag[mask_pec, 1:2], E_mag[mask_pec, 2:3]], dim=1)
        loss = loss + E_tan.square().mean()

    mask_rad = (x[:, 2] > 0.99)
    if mask_rad.any():
        x_bc = x[mask_rad].clone().detach().requires_grad_(True)
        omega = x_bc[:, 3:4]
        Ez = net(x_bc)[:, 2:3]
        grad = torch.autograd.grad(Ez.sum(), x_bc, create_graph=True, allow_unused=True)[0]
        grad_z = grad[:, 2:3] if grad is not None else torch.zeros_like(Ez)
        epsilon = MaterialProperties.epsilon_r(x_bc)
        mu = MaterialProperties.mu_r(x_bc)
        beta = omega * torch.sqrt(epsilon * mu)
        radiation_residual = grad_z + beta * Ez
        loss = loss + radiation_residual.square().mean()

    return loss


def compute_total_loss(net, x, device, lambda_pde=1.0, lambda_bc=10.0):
    lpde = compute_pde_loss(net, x, device)
    lbc = compute_bc_loss(net, x)
    return lambda_pde * lpde + lambda_bc * lbc, lpde, lbc


# =========================
# 8. Training
# =========================

def train(data_path=DATA_PATH, model_path=MODEL_PATH, epochs=EPOCHS):
    set_seed(SEED)
    device = get_device()
    t0 = time.time()

    X = load_or_generate_data(data_path, device=device)
    N = X.shape[0]

    model = HighPrecisionPINN(
        input_dim=4, hidden_dim=HIDDEN_DIM, output_dim=6, n_layers=N_LAYERS
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=50
    )

    history = {"total": [], "pde": [], "bc": []}

    print("Using device:", device)
    print(f"Collocation points: {N}")
    print(f"Model: {sum(p.numel() for p in model.parameters())} params")
    print(f"Epochs: {epochs}, Batch: {BATCH_SIZE}, LR: {LR}")
    print("\nStart training")

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(N, device=device)
        sum_loss, sum_pde, sum_bc = 0.0, 0.0, 0.0
        nb = 0

        for i in range(0, N, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            x_batch = X[idx].clone().detach().requires_grad_(True)

            optimizer.zero_grad()
            loss, lpde, lbc = compute_total_loss(
                model, x_batch, device, lambda_pde=LAMBDA_PDE, lambda_bc=LAMBDA_BC
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            sum_loss += loss.item()
            sum_pde += lpde.item()
            sum_bc += lbc.item()
            nb += 1

        avg_loss = sum_loss / nb
        avg_pde = sum_pde / nb
        avg_bc = sum_bc / nb

        history["total"].append(avg_loss)
        history["pde"].append(avg_pde)
        history["bc"].append(avg_bc)

        scheduler.step(avg_loss)

        if epoch == 1 or epoch % PRINT_EVERY == 0 or epoch == epochs:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:5d}/{epochs} | "
                f"Loss={avg_loss:.3e} | PDE={avg_pde:.3e} | BC={avg_bc:.3e} | LR={lr:.1e}"
            )

    model.eval()

    model_path = os.path.expanduser(model_path)
    os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else ".", exist_ok=True)

    checkpoint = {
        "project": "MaxwellPDE",
        "state_dict": model.state_dict(),
        "input_dim": 4,
        "hidden_dim": HIDDEN_DIM,
        "output_dim": 6,
        "n_layers": N_LAYERS,
        "history": history,
        "data_path": data_path,
    }
    torch.save(checkpoint, model_path)

    elapsed = time.time() - t0
    print(f"\nTraining finished in {elapsed:.1f}s")
    print(f"Model saved to: {model_path}")

    show_train_result(history)
    return checkpoint, model


# =========================
# 9. Visualization
# =========================

def show_train_result(history):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    total = np.maximum(np.asarray(history["total"], dtype=np.float32), 1e-12)
    pde = np.maximum(np.asarray(history["pde"], dtype=np.float32), 1e-12)
    bc = np.maximum(np.asarray(history["bc"], dtype=np.float32), 1e-12)

    axes[0].semilogy(total, "b-", linewidth=0.8, alpha=0.9)
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(pde, "r-", linewidth=0.8, alpha=0.9)
    axes[1].set_title("PDE Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)

    axes[2].semilogy(bc, "g-", linewidth=0.8, alpha=0.9)
    axes[2].set_title("Boundary Condition Loss")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# =========================
# 10. Run
# =========================

checkpoint, model = train()
