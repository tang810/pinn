"""
Notebook-friendly eval file for 3D frequency-domain Maxwell PINN.

Upload trained_model_maxwell_pde.pt as a public model asset. If you also upload
training_data.npy as a public data asset, this file evaluates physics losses on
that data; otherwise it evaluates a small synthetic collocation set.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = "00acab00eb074642a4b90f32d3bb561e"
MODEL_HASH = "4861b53221f949789b75a051b9af092d"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/trained_model_maxwell_pde.pt")

SLICE_RESOLUTION = 32
MAX_EVAL_POINTS = 512
EVAL_OMEGA = np.pi


# =========================
# 2. Utilities
# =========================

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


def is_filled(value):
    return bool(value) and not value.startswith("FILL_")


def generate_small_data(n_points=512):
    rng = np.random.default_rng(42)
    x = rng.uniform(-1, 1, n_points)
    y = rng.uniform(-1, 1, n_points)
    z = rng.uniform(-1, 1, n_points)
    omega = rng.uniform(0.5, 5.0, n_points)
    return np.column_stack([x, y, z, omega]).astype(np.float32)


def resolve_data_path(data_path):
    if not is_filled(DATA_HASH):
        return None
    data_path = os.path.expanduser(data_path)
    if os.path.isfile(data_path):
        return data_path
    candidate = os.path.join(data_path, "training_data.npy")
    if os.path.isfile(candidate):
        return candidate
    return None


def load_eval_data(data_path):
    resolved = resolve_data_path(data_path)
    if resolved:
        data = np.load(resolved).astype(np.float32)
        source = resolved
    else:
        data = generate_small_data(MAX_EVAL_POINTS)
        source = "synthetic collocation points"
    if data.ndim != 2 or data.shape[1] != 4:
        raise ValueError(f"Data must have shape N x 4 [x,y,z,omega], got {data.shape}.")
    if len(data) > MAX_EVAL_POINTS:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(len(data), size=MAX_EVAL_POINTS, replace=False))
        data = data[idx]
    return data, source


# =========================
# 3. Network
# =========================

class HighPrecisionPINN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=6, n_layers=2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for i in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU() if i % 2 == 0 else nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def infer_model_shape(state_dict):
    weight_keys = sorted(
        [key for key in state_dict if key.startswith("net.") and key.endswith(".weight")],
        key=lambda key: int(key.split(".")[1]),
    )
    if not weight_keys:
        return 4, 64, 6, 2
    input_dim = int(state_dict[weight_keys[0]].shape[1])
    hidden_dim = int(state_dict[weight_keys[0]].shape[0])
    output_dim = int(state_dict[weight_keys[-1]].shape[0])
    n_layers = max(2, len(weight_keys))
    return input_dim, hidden_dim, output_dim, n_layers


# =========================
# 4. Physics operators
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
                f[:, i].sum(), x, create_graph=True, retain_graph=True, allow_unused=True
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
        curl_mu_inv_curl_E = Curl.curl(curl_E / mu.expand_as(E), x)
        return curl_mu_inv_curl_E - epsilon.expand_as(E) * omega**2 * E


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


def compute_pde_loss(net, x, device):
    omega = x[:, 3:4]
    out = net(x)
    E_real = out[:, 0:3]
    E_imag = out[:, 3:6]
    epsilon = MaterialProperties.epsilon_r(x)
    mu = MaterialProperties.mu_r(x)
    helmholtz_real = Curl.helmholtz_operator(E_real, x, epsilon, mu, omega)
    helmholtz_imag = Curl.helmholtz_operator(E_imag, x, epsilon, mu, omega)
    J = SourceTerms.plane_wave_source(x, omega, device)
    residual_real = helmholtz_real + omega * mu.expand_as(E_real) * J.imag
    residual_imag = helmholtz_imag - omega * mu.expand_as(E_imag) * J.real
    return residual_real.square().mean() + residual_imag.square().mean()


def compute_bc_loss(net, x):
    out = net(x)
    E_real = out[:, 0:3]
    E_imag = out[:, 3:6]
    E_mag = torch.sqrt(E_real**2 + E_imag**2)
    loss = torch.tensor(0.0, device=x.device)

    mask_pec = x[:, 0].abs() > 0.99
    if mask_pec.any():
        E_tan = torch.cat([E_mag[mask_pec, 1:2], E_mag[mask_pec, 2:3]], dim=1)
        loss = loss + E_tan.square().mean()

    mask_rad = x[:, 2] > 0.99
    if mask_rad.any():
        x_bc = x[mask_rad].clone().detach().requires_grad_(True)
        omega = x_bc[:, 3:4]
        Ez = net(x_bc)[:, 2:3]
        grad = torch.autograd.grad(Ez.sum(), x_bc, create_graph=True, allow_unused=True)[0]
        grad_z = grad[:, 2:3] if grad is not None else torch.zeros_like(Ez)
        epsilon = MaterialProperties.epsilon_r(x_bc)
        mu = MaterialProperties.mu_r(x_bc)
        beta = omega * torch.sqrt(epsilon * mu)
        loss = loss + (grad_z + beta * Ez).square().mean()
    return loss


# =========================
# 5. Model loading and eval
# =========================

def load_checkpoint(model_path, device):
    model_path = os.path.expanduser(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"MODEL_PATH does not exist: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    state = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    input_dim, hidden_dim, output_dim, n_layers = infer_model_shape(state)
    model = HighPrecisionPINN(
        input_dim=checkpoint.get("input_dim", input_dim) if isinstance(checkpoint, dict) else input_dim,
        hidden_dim=checkpoint.get("hidden_dim", hidden_dim) if isinstance(checkpoint, dict) else hidden_dim,
        output_dim=checkpoint.get("output_dim", output_dim) if isinstance(checkpoint, dict) else output_dim,
        n_layers=checkpoint.get("n_layers", n_layers) if isinstance(checkpoint, dict) else n_layers,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return checkpoint if isinstance(checkpoint, dict) else {}, model


def evaluate(model_path=MODEL_PATH, data_path=DATA_PATH):
    device = get_device()
    checkpoint, model = load_checkpoint(model_path, device)
    X_np, source = load_eval_data(data_path)
    X_t = torch.tensor(X_np, dtype=torch.float32, device=device).requires_grad_(True)

    lpde_val = float(compute_pde_loss(model, X_t, device).detach().cpu())
    lbc_val = float(compute_bc_loss(model, X_t).detach().cpu())
    metrics = {"pde_loss": lpde_val, "bc_loss": lbc_val, "total_loss": lpde_val + 10.0 * lbc_val}

    print("Device:", device)
    print("Model path:", os.path.expanduser(model_path))
    print(f"Data: {source}, points: {len(X_np)}")
    print("\nEval metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6e}")

    show_eval_result(checkpoint, model, device)
    return {"metrics": metrics, "model": model}


# =========================
# 6. Visualization
# =========================

def make_slice_grid(omega_val, z_val=0.0, resolution=32):
    xs = np.linspace(-1, 1, resolution, dtype=np.float32)
    ys = np.linspace(-1, 1, resolution, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    n = resolution * resolution
    return np.stack([
        xx.reshape(-1),
        yy.reshape(-1),
        np.full(n, z_val, dtype=np.float32),
        np.full(n, omega_val, dtype=np.float32),
    ], axis=1)


def show_eval_result(checkpoint, model, device):
    history = checkpoint.get("history", {}) if isinstance(checkpoint, dict) else {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    if history and "total" in history:
        axes[0].semilogy(np.maximum(np.asarray(history["total"], dtype=np.float32), 1e-12), label="total")
        axes[0].semilogy(np.maximum(np.asarray(history.get("pde", []), dtype=np.float32), 1e-12), label="pde")
        axes[0].semilogy(np.maximum(np.asarray(history.get("bc", []), dtype=np.float32), 1e-12), label="bc")
        axes[0].legend()
    axes[0].set_title("Training loss")
    axes[0].grid(True, alpha=0.3)

    X_slice = make_slice_grid(EVAL_OMEGA, resolution=SLICE_RESOLUTION)
    X_t = torch.tensor(X_slice, dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(X_t).cpu().numpy()
    Ex = out[:, 0].reshape(SLICE_RESOLUTION, SLICE_RESOLUTION)
    Ey = out[:, 1].reshape(SLICE_RESOLUTION, SLICE_RESOLUTION)
    E_mag = np.sqrt(out[:, :3]**2 + out[:, 3:6]**2).sum(axis=1).reshape(
        SLICE_RESOLUTION, SLICE_RESOLUTION
    )

    for ax, title, data in [(axes[1], "Ex real", Ex), (axes[2], "|E|", E_mag)]:
        im = ax.imshow(data.T, origin="lower", cmap="RdBu_r", extent=[-1, 1, -1, 1], aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.show()


# =========================
# 7. Run
# =========================

eval_result = evaluate()
