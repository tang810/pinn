"""
Notebook-friendly train file for 3D_ElasticForce_Deformation.

Upload Coord.mat and FEA.mat as one public dataset asset, fill DATA_HASH and
MODEL_HASH, then run this whole file/cell. The trained model is saved to:
~/minio/Resource/{MODEL_HASH}/pinn_elasticity_model.pt
"""

import os
import time
import random

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = "FILL_DATA_HASH"
MODEL_HASH = "FILL_MODEL_SAVE_HASH"
DATA_FILE_PATH = ""

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/minio/Resource/{MODEL_HASH}/pinn_elasticity_model.pt")

COORD_FILE_NAME = "Coord.mat"
FEA_FILE_NAME = "FEA.mat"

TRAIN_EPOCHS = 2000
PRINT_EVERY = 100
SEED = 42

LR = 1e-2
E_MODULUS = 1.0
POISSON = 0.25
LAYERS = [3, 20, 20, 20, 20, 1]
PREDICT_BATCH = 9261


# =========================
# 2. Path and data helpers
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


def is_filled(value):
    return bool(value) and not value.startswith("FILL_")


def resource_roots():
    return [
        os.path.expanduser("~/Resource"),
        os.path.expanduser("~/public/Resource"),
        "/mnt/minio/public/Resource",
        os.path.expanduser("~/minio/Resource"),
        "/mnt/minio/Resource",
        "/mnt/minio/userdk8e2v7l/Resource",
        "/mnt/minio/userdk8e2v7l/public/Resource",
    ]


def uniq(items):
    out = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


def candidate_data_dirs(data_path):
    candidates = []
    if DATA_FILE_PATH:
        exact = os.path.expanduser(DATA_FILE_PATH)
        candidates.append(exact)
        candidates.append(os.path.dirname(exact))
        if exact.startswith("jupyterfile-"):
            converted = "/mnt/minio/" + exact.replace("jupyterfile-", "", 1)
            candidates.append(converted)
            candidates.append(os.path.dirname(converted))
    if data_path:
        expanded = os.path.expanduser(data_path)
        candidates.append(expanded)
        candidates.append(os.path.dirname(expanded))
        if expanded.startswith("jupyterfile-"):
            converted = "/mnt/minio/" + expanded.replace("jupyterfile-", "", 1)
            candidates.append(converted)
            candidates.append(os.path.dirname(converted))
    if is_filled(DATA_HASH):
        for root in resource_roots():
            candidates.append(os.path.join(root, DATA_HASH))
    return uniq(candidates)


def resolve_dataset_dir(data_path):
    tried = []
    for path in candidate_data_dirs(data_path):
        if not path:
            continue
        tried.append(path)
        if os.path.isfile(path) and os.path.basename(path) == COORD_FILE_NAME:
            folder = os.path.dirname(path)
            if os.path.isfile(os.path.join(folder, FEA_FILE_NAME)):
                return folder
        if os.path.isdir(path):
            if os.path.isfile(os.path.join(path, COORD_FILE_NAME)) and os.path.isfile(os.path.join(path, FEA_FILE_NAME)):
                return path
    raise FileNotFoundError(
        f"Cannot find {COORD_FILE_NAME} and {FEA_FILE_NAME}. Upload both files into one public dataset asset, "
        f"then set DATA_HASH. If the platform gives a concrete file path, paste it into DATA_FILE_PATH.\n"
        f"Tried paths: {tried}"
    )


def load_mat(path):
    try:
        import scipy.io as sio
    except Exception as exc:
        raise ImportError("This file needs scipy to read .mat files.") from exc
    return sio.loadmat(path)


def load_training_dataset(data_dir, device):
    data = load_mat(os.path.join(data_dir, COORD_FILE_NAME))
    x = torch.tensor(data["xy"], dtype=torch.float32, device=device)
    x1u = torch.tensor(data["x1u"], dtype=torch.float32, device=device)
    x1b = torch.tensor(data["x1b"], dtype=torch.float32, device=device)
    x2u = torch.tensor(data["x2u"], dtype=torch.float32, device=device)
    x2b = torch.tensor(data["x2b"], dtype=torch.float32, device=device)
    x3u = torch.tensor(data["x3u"], dtype=torch.float32, device=device)
    x3b = torch.tensor(data["x3b"], dtype=torch.float32, device=device)
    target_z_stress = torch.cos(x3u[:, 0] * np.pi / 2.0) * torch.cos(x3u[:, 1] * np.pi / 2.0)
    return {
        "x": x,
        "x1u": x1u,
        "x1b": x1b,
        "x2u": x2u,
        "x2b": x2b,
        "x3u": x3u,
        "x3b": x3b,
        "target_z_stress": target_z_stress,
        "coord_path": os.path.join(data_dir, COORD_FILE_NAME),
    }


def load_eval_points(data_dir, max_points=20000):
    data = load_mat(os.path.join(data_dir, FEA_FILE_NAME))
    x = np.asarray(data["X"], dtype=np.float32)
    if len(x) > max_points:
        idx = np.linspace(0, len(x) - 1, max_points).astype(np.int64)
        x = x[idx]
    return x


# =========================
# 3. Model and physics
# =========================

class MLP(nn.Module):
    def __init__(self, layers):
        super().__init__()
        modules = []
        for i in range(len(layers) - 2):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            modules.append(nn.Tanh())
        modules.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*modules)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.net(x)


class MultiFNN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.u_net = MLP(layers)
        self.v_net = MLP(layers)
        self.w_net = MLP(layers)

    def u_model(self, x, y, z):
        inputs = torch.stack([x, y, z], dim=-1)
        return self.u_net(inputs).squeeze(-1) * x

    def v_model(self, x, y, z):
        inputs = torch.stack([x, y, z], dim=-1)
        return self.v_net(inputs).squeeze(-1) * y

    def w_model(self, x, y, z):
        inputs = torch.stack([x, y, z], dim=-1)
        return self.w_net(inputs).squeeze(-1) * z

    def compute_derivatives(self, x, y, z):
        x = x.clone().detach().requires_grad_(True)
        y = y.clone().detach().requires_grad_(True)
        z = z.clone().detach().requires_grad_(True)

        u = self.u_model(x, y, z)
        v = self.v_model(x, y, z)
        w = self.w_model(x, y, z)

        def grad(val, wrt):
            return torch.autograd.grad(val, wrt, torch.ones_like(val), create_graph=True)[0]

        u_x, u_y, u_z = grad(u, x), grad(u, y), grad(u, z)
        v_x, v_y, v_z = grad(v, x), grad(v, y), grad(v, z)
        w_x, w_y, w_z = grad(w, x), grad(w, y), grad(w, z)

        u_xx, u_xy, u_xz = grad(u_x, x), grad(u_x, y), grad(u_x, z)
        u_yy, u_yz, u_zz = grad(u_y, y), grad(u_y, z), grad(u_z, z)
        v_xx, v_xy, v_xz = grad(v_x, x), grad(v_x, y), grad(v_x, z)
        v_yy, v_yz, v_zz = grad(v_y, y), grad(v_y, z), grad(v_z, z)
        w_xx, w_xy, w_xz = grad(w_x, x), grad(w_x, y), grad(w_x, z)
        w_yy, w_yz, w_zz = grad(w_y, y), grad(w_y, z), grad(w_z, z)

        return (
            u_x, u_y, u_z, u_xx, u_xy, u_xz, u_yy, u_yz, u_zz,
            v_x, v_y, v_z, v_xx, v_xy, v_xz, v_yy, v_yz, v_zz,
            w_x, w_y, w_z, w_xx, w_xy, w_xz, w_yy, w_yz, w_zz,
        )

    def predict_displacement(self, coords, device, batch_size=9261):
        self.eval()
        result = []
        with torch.no_grad():
            for start in range(0, len(coords), batch_size):
                xb = torch.tensor(coords[start:start + batch_size], dtype=torch.float32, device=device)
                u = self.u_model(xb[:, 0], xb[:, 1], xb[:, 2])
                v = self.v_model(xb[:, 0], xb[:, 1], xb[:, 2])
                w = self.w_model(xb[:, 0], xb[:, 1], xb[:, 2])
                result.append(torch.stack([u, v, w], dim=1).cpu().numpy())
        return np.vstack(result)


def material(derivs, E=E_MODULUS, mu=POISSON):
    (
        u_x, u_y, u_z, u_xx, u_xy, u_xz, u_yy, u_yz, u_zz,
        v_x, v_y, v_z, v_xx, v_xy, v_xz, v_yy, v_yz, v_zz,
        w_x, w_y, w_z, w_xx, w_xy, w_xz, w_yy, w_yz, w_zz,
    ) = derivs
    lam = E * mu / (1 + mu) / (1 - 2 * mu)
    shear = E / (1 + mu) / 2

    e1, e2, e3 = u_x, v_y, w_z
    e12 = 0.5 * (u_y + v_x)
    e23 = 0.5 * (w_y + v_z)
    e13 = 0.5 * (u_z + w_x)

    s1 = (2 * shear + lam) * e1 + lam * e2 + lam * e3
    s2 = (2 * shear + lam) * e2 + lam * e1 + lam * e3
    s3 = (2 * shear + lam) * e3 + lam * e1 + lam * e2
    s12, s23, s13 = 2 * shear * e12, 2 * shear * e23, 2 * shear * e13

    gex = (shear + lam) * (u_xx + v_xy + w_xz) + shear * (u_xx + u_yy + u_zz)
    gey = (shear + lam) * (v_yy + u_xy + w_yz) + shear * (v_xx + v_yy + v_zz)
    gez = (shear + lam) * (w_zz + u_xz + v_yz) + shear * (w_xx + w_yy + w_zz)
    return e1, e2, e3, e12, e23, e13, s1, s2, s3, s12, s23, s13, gex, gey, gez


def compute_losses(model, data):
    x = data["x"]
    s3_target = data["target_z_stress"]

    mat_in = material(model.compute_derivatives(x[:, 0], x[:, 1], x[:, 2]))
    gex, gey, gez = mat_in[-3], mat_in[-2], mat_in[-1]
    loss_pde = torch.mean(gex ** 2 + gey ** 2 + gez ** 2)

    x1u, x1b = data["x1u"], data["x1b"]
    x2u, x2b = data["x2u"], data["x2b"]
    x3u, x3b = data["x3u"], data["x3b"]
    s_1u = material(model.compute_derivatives(x1u[:, 0], x1u[:, 1], x1u[:, 2]))
    s_1b = material(model.compute_derivatives(x1b[:, 0], x1b[:, 1], x1b[:, 2]))
    s_2u = material(model.compute_derivatives(x2u[:, 0], x2u[:, 1], x2u[:, 2]))
    s_2b = material(model.compute_derivatives(x2b[:, 0], x2b[:, 1], x2b[:, 2]))
    s_3u = material(model.compute_derivatives(x3u[:, 0], x3u[:, 1], x3u[:, 2]))
    s_3b = material(model.compute_derivatives(x3b[:, 0], x3b[:, 1], x3b[:, 2]))

    loss_x = torch.mean(s_1u[6] ** 2 + s_1u[9] ** 2 + s_1u[11] ** 2) + torch.mean(s_1b[9] ** 2 + s_1b[11] ** 2)
    loss_y = torch.mean(s_2u[7] ** 2 + s_2u[9] ** 2 + s_2u[10] ** 2) + torch.mean(s_2b[9] ** 2 + s_2b[10] ** 2)
    loss_z = torch.mean((s_3u[8] - s3_target) ** 2 + s_3u[10] ** 2 + s_3u[11] ** 2) + torch.mean(s_3b[10] ** 2 + s_3b[11] ** 2)
    total = loss_pde + loss_x + loss_y + loss_z
    return total, loss_pde, loss_x, loss_y, loss_z


def predict_stress_component(model, coords, device, batch_size=4096):
    model.eval()
    values = []
    for start in range(0, len(coords), batch_size):
        xb = torch.tensor(coords[start:start + batch_size], dtype=torch.float32, device=device)
        stress = material(model.compute_derivatives(xb[:, 0], xb[:, 1], xb[:, 2]))
        s3 = stress[8].detach().cpu().numpy()
        values.append(s3)
    return np.concatenate(values)


# =========================
# 4. Train and visualize
# =========================

def train(data_path=DATA_PATH, model_path=MODEL_PATH):
    set_seed(SEED)
    device = get_device()
    print("Using device:", device)

    data_dir = resolve_dataset_dir(data_path)
    train_data = load_training_dataset(data_dir, device)
    eval_points = load_eval_points(data_dir)
    model = MultiFNN(LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    history = {"total": [], "pde": [], "x_bc": [], "y_bc": [], "z_bc": []}
    t0 = time.time()

    print("Dataset folder:", data_dir)
    print("coord points:", tuple(train_data["x"].shape), "eval points:", tuple(eval_points.shape))
    print("\nStart training")

    for epoch in range(1, TRAIN_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        total, loss_pde, loss_x, loss_y, loss_z = compute_losses(model, train_data)
        total.backward()
        optimizer.step()
        if epoch % 2000 == 0:
            scheduler.step()

        history["total"].append(float(total.detach().cpu()))
        history["pde"].append(float(loss_pde.detach().cpu()))
        history["x_bc"].append(float(loss_x.detach().cpu()))
        history["y_bc"].append(float(loss_y.detach().cpu()))
        history["z_bc"].append(float(loss_z.detach().cpu()))

        if epoch == 1 or epoch % PRINT_EVERY == 0 or epoch == TRAIN_EPOCHS:
            print(
                f"Epoch {epoch:5d}/{TRAIN_EPOCHS} | loss={total.item():.6e} | "
                f"pde={loss_pde.item():.3e} | x={loss_x.item():.3e} | "
                f"y={loss_y.item():.3e} | z={loss_z.item():.3e}"
            )

    checkpoint = {
        "model_state": model.state_dict(),
        "logs": {
            "loss_total": history["total"],
            "loss_pde": history["pde"],
            "loss_x_bnd": history["x_bc"],
            "loss_y_bnd": history["y_bc"],
            "loss_z_bnd": history["z_bc"],
        },
        "layers": LAYERS,
        "E": E_MODULUS,
        "mu": POISSON,
        "coord_path": train_data["coord_path"],
        "fea_path": os.path.join(data_dir, FEA_FILE_NAME),
    }
    saved_model_path = save_checkpoint(checkpoint, model_path)

    disp = model.predict_displacement(eval_points[:PREDICT_BATCH], device)
    s3 = predict_stress_component(model, eval_points[:PREDICT_BATCH], device)

    print("\nTraining finished")
    print(f"elapsed: {time.time() - t0:.1f}s")
    print("final loss:", history["total"][-1])
    print("model saved to:", saved_model_path)
    show_result(history, eval_points[:PREDICT_BATCH], disp, s3)
    return checkpoint, model


def save_checkpoint(checkpoint, model_path):
    path = os.path.abspath(os.path.expanduser(model_path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)
    return path


def show_result(history, coords, displacement, s3):
    disp_mag = np.linalg.norm(displacement, axis=1)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    axes[0].semilogy(history["total"], label="total")
    axes[0].semilogy(history["pde"], label="pde")
    axes[0].semilogy(history["z_bc"], label="z_bc")
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    im1 = axes[1].scatter(coords[:, 0], coords[:, 1], c=disp_mag, s=5, cmap="viridis")
    axes[1].set_title("Predicted displacement magnitude")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].scatter(coords[:, 0], coords[:, 1], c=s3, s=5, cmap="coolwarm")
    axes[2].set_title("Predicted sigma_zz")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    plt.show()


checkpoint, model = train()
