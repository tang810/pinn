"""
Notebook-friendly eval file for 3D_ElasticForce_Deformation.

Upload Coord.mat and FEA.mat as one public dataset asset, upload the trained
pinn_elasticity_model.pt as a public model asset, fill DATA_HASH and MODEL_HASH,
then run this whole file/cell. Figures are shown with plt.show().
"""

import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = "FILL_DATA_HASH"
MODEL_HASH = "FILL_MODEL_ASSET_HASH"
DATA_FILE_PATH = ""

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/pinn_elasticity_model.pt")

COORD_FILE_NAME = "Coord.mat"
FEA_FILE_NAME = "FEA.mat"
E_MODULUS = 1.0
POISSON = 0.25
DEFAULT_LAYERS = [3, 20, 20, 20, 20, 1]
MAX_EVAL_POINTS = 20000


# =========================
# 2. Path and data helpers
# =========================

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        if os.path.isfile(path) and os.path.basename(path) in (COORD_FILE_NAME, FEA_FILE_NAME):
            folder = os.path.dirname(path)
            if os.path.isfile(os.path.join(folder, FEA_FILE_NAME)):
                return folder
        if os.path.isdir(path):
            if os.path.isfile(os.path.join(path, FEA_FILE_NAME)):
                return path
    raise FileNotFoundError(
        f"Cannot find {FEA_FILE_NAME}. Upload {COORD_FILE_NAME} and {FEA_FILE_NAME} into one public dataset asset, "
        f"then set DATA_HASH. If the platform gives a concrete file path, paste it into DATA_FILE_PATH.\n"
        f"Tried paths: {tried}"
    )


def load_mat(path):
    try:
        import scipy.io as sio
    except Exception as exc:
        raise ImportError("This file needs scipy to read .mat files.") from exc
    return sio.loadmat(path)


def load_eval_points(data_dir, max_points=MAX_EVAL_POINTS):
    data = load_mat(os.path.join(data_dir, FEA_FILE_NAME))
    coords = np.asarray(data["X"], dtype=np.float32)
    if len(coords) > max_points:
        idx = np.linspace(0, len(coords) - 1, max_points).astype(np.int64)
        coords = coords[idx]
    return coords


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
    return s1, s2, s3, 2 * shear * e12, 2 * shear * e23, 2 * shear * e13


def load_checkpoint(model_path, device):
    model_path = os.path.expanduser(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"MODEL_PATH does not exist: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    layers = checkpoint.get("layers", DEFAULT_LAYERS)
    model = MultiFNN(layers).to(device)
    state = checkpoint.get("model_state", checkpoint.get("model_state_dict", checkpoint))
    model.load_state_dict(state)
    model.eval()
    return checkpoint, model


def predict_stress_component(model, coords, device, batch_size=4096):
    model.eval()
    values = []
    for start in range(0, len(coords), batch_size):
        xb = torch.tensor(coords[start:start + batch_size], dtype=torch.float32, device=device)
        stress = material(model.compute_derivatives(xb[:, 0], xb[:, 1], xb[:, 2]))
        values.append(stress[2].detach().cpu().numpy())
    return np.concatenate(values)


# =========================
# 4. Eval and visualize
# =========================

def evaluate(model_path=MODEL_PATH, data_path=DATA_PATH):
    device = get_device()
    print("Using device:", device)

    data_dir = resolve_dataset_dir(data_path)
    checkpoint, model = load_checkpoint(model_path, device)
    coords = load_eval_points(data_dir)
    disp = model.predict_displacement(coords, device)
    s3 = predict_stress_component(model, coords, device)
    disp_mag = np.linalg.norm(disp, axis=1)

    logs = checkpoint.get("logs", {})
    print("\nModel loaded:", os.path.expanduser(model_path))
    print("Dataset folder:", data_dir)
    print("eval points:", len(coords))
    print("\nEval result")
    print(f"  displacement_abs_max: {np.max(np.abs(disp)):.6e}")
    print(f"  displacement_abs_mean: {np.mean(np.abs(disp)):.6e}")
    print(f"  sigma_zz_min: {np.min(s3):.6e}")
    print(f"  sigma_zz_max: {np.max(s3):.6e}")
    if logs.get("loss_total"):
        print(f"  train_final_loss: {logs['loss_total'][-1]:.6e}")

    show_result(coords, disp_mag, s3, logs)
    return {"coords": coords, "displacement": disp, "sigma_zz": s3, "checkpoint": checkpoint}


def show_result(coords, disp_mag, s3, logs):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    if logs.get("loss_total"):
        axes[0].semilogy(logs["loss_total"], label="total")
        if logs.get("loss_pde"):
            axes[0].semilogy(logs["loss_pde"], label="pde")
        if logs.get("loss_z_bnd"):
            axes[0].semilogy(logs["loss_z_bnd"], label="z_bc")
        axes[0].legend()
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("epoch")
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


result = evaluate()
