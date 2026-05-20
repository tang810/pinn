"""
Notebook-friendly train file for airfoil_force_pinn.

Online Jupyter usage:
1. Upload data.mat as a public asset.
2. Fill DATA_HASH and MODEL_HASH.
3. Run this whole file/cell. It saves trained_model_airfoil_force.pt to
   ~/minio/Resource/{MODEL_HASH}/ for later public-asset upload.

Physics: 3D linear elasticity PINN (Navier-Lame eqns) for airfoil force field.
         Input (x,y,z) → Output (u,v,w). E=100, nu=0.3, body forces f=(0.1,0.1,0.1).

Self-contained — only depends on the .mat dataset.
"""

import os
import time
from collections import OrderedDict

import numpy as np
import scipy.io
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

DATA_HASH = "f4ff6be555ac487fbbe0a866acac3d2b"
MODEL_HASH = "f4ff6be555ac487fbbe0a866acac3d2b"
DATA_FILENAME = "data.mat"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/minio/Resource/{MODEL_HASH}/trained_model_airfoil_force.pt")
FORCE_RETRAIN = False


# =========================
# 2. Model (FCNet, OrderedDict)
# =========================

class FCNet(nn.Module):
    def __init__(self, num_ins=3, num_outs=3, num_layers=6, hidden_size=50):
        super().__init__()
        sizes = [num_ins] + [hidden_size] * num_layers + [num_outs]
        layer_list = []
        for i in range(len(sizes) - 2):
            layer_list.append((f"layer_{i}", nn.Linear(sizes[i], sizes[i + 1])))
            layer_list.append((f"activation_{i}", nn.Tanh()))
        layer_list.append((f"layer_{len(sizes) - 2}", nn.Linear(sizes[-2], sizes[-1])))
        self.layers = nn.Sequential(OrderedDict(layer_list))

    def forward(self, x):
        return self.layers(x)


# =========================
# 3. Data loading from .mat
# =========================

def load_mat_data(filename):
    data = scipy.io.loadmat(filename)
    bx = data['bx']; by = data['by']; bz = data['bz'].astype(np.float32)
    lx = data['lx']; ly = data['ly']; lz = data['lz'].astype(np.float32)
    ux = data['ux']; uy = data['uy']; uz = data['uz'].astype(np.float32)
    b_u = data['b_u'].astype(np.float32); b_v = data['b_v'].astype(np.float32); b_w = data['b_w'].astype(np.float32)
    l_u = data['l_u']; l_v = data['l_v']; l_w = data['l_w']
    u_u = data['u_u']; u_v = data['u_v']; u_w = data['u_w']
    x_b = np.concatenate([bx, lx, ux], 0).reshape(-1, 1)
    y_b = np.concatenate([by, ly, uy], 0).reshape(-1, 1)
    z_b = np.concatenate([bz, lz, uz], 0).reshape(-1, 1)
    x_u = np.concatenate([b_u, l_u, u_u], 0).reshape(-1, 1)
    y_v = np.concatenate([b_v, l_v, u_v], 0).reshape(-1, 1)
    z_w = np.concatenate([b_w, l_w, u_w], 0).reshape(-1, 1)
    x_f = data['x']; y_f = data['y']; z_f = data['z'].astype(np.float32)
    x_s = data['x']; y_s = data['y']; z_s = data['z'].astype(np.float32)
    u_s = data['u_ref']; v_s = data['v_ref']; w_s = data['w_ref']
    return (x_b, y_b, z_b, x_u, y_v, z_w), (x_f, y_f, z_f), (x_s, y_s, z_s, u_s, v_s, w_s)


# =========================
# 4. PINN Solver (3D linear elasticity)
# =========================

class AirfoilForcePINN:
    def __init__(self, E=100, nu=0.3, hidden=50, layers=6, lr=1e-3):
        self.E = E; self.nu = nu
        self.G = E / (2 * (1 + nu))
        self.la = E * nu / ((1 + nu) * (1 - 2 * nu))
        self.device = get_device()
        self.net = FCNet(num_ins=3, num_outs=3, num_layers=layers, hidden_size=hidden).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def _forward(self, x, y, z):
        X = torch.cat([x, y, z], dim=1)
        out = self.net(X)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]

    def _grad(self, y, xs):
        g = torch.autograd.grad(y, xs, [torch.ones_like(y)] * len(xs), create_graph=True, allow_unused=True)
        return [gi if gi is not None else torch.zeros_like(xs[i]) for i, gi in enumerate(g)]

    def _pde(self, x, y, z):
        u, v, w = self._forward(x, y, z)
        u_x, u_y, u_z = self._grad(u, [x, y, z])
        u_xx = self._grad(u_x, [x])[0]; u_yy = self._grad(u_y, [y])[0]; u_zz = self._grad(u_z, [z])[0]
        v_x, v_y, v_z = self._grad(v, [x, y, z])
        v_xx = self._grad(v_x, [x])[0]; v_yy = self._grad(v_y, [y])[0]; v_zz = self._grad(v_z, [z])[0]
        w_x, w_y, w_z = self._grad(w, [x, y, z])
        w_xx = self._grad(w_x, [x])[0]; w_yy = self._grad(w_y, [y])[0]; w_zz = self._grad(w_z, [z])[0]
        u_xy = self._grad(u_x, [y])[0]; u_xz = self._grad(u_x, [z])[0]
        v_xy = self._grad(v_x, [y])[0]; v_yz = self._grad(v_y, [z])[0]
        w_xz = self._grad(w_x, [z])[0]; w_yz = self._grad(w_y, [z])[0]
        G, la, f = self.G, self.la, 0.1
        eq1 = (2*G+la)*u_xx + G*(u_yy+u_zz) + (G+la)*(v_xy+w_xz) + f
        eq2 = (2*G+la)*v_yy + G*(v_xx+v_zz) + (G+la)*(u_xy+w_yz) + f
        eq3 = (2*G+la)*w_zz + G*(w_xx+w_yy) + (G+la)*(u_xz+v_yz) + f
        return eq1, eq2, eq3

    def set_data(self, boundary_data, eq_data):
        (x_b, y_b, z_b, x_u, y_v, z_w) = boundary_data
        (x_f, y_f, z_f) = eq_data
        dev = self.device
        self.x_b = torch.tensor(x_b, dtype=torch.float32, device=dev)
        self.y_b = torch.tensor(y_b, dtype=torch.float32, device=dev)
        self.z_b = torch.tensor(z_b, dtype=torch.float32, device=dev)
        self.x_u = torch.tensor(x_u, dtype=torch.float32, device=dev)
        self.y_v = torch.tensor(y_v, dtype=torch.float32, device=dev)
        self.z_w = torch.tensor(z_w, dtype=torch.float32, device=dev)
        self.x_f = torch.tensor(x_f, dtype=torch.float32, requires_grad=True, device=dev)
        self.y_f = torch.tensor(y_f, dtype=torch.float32, requires_grad=True, device=dev)
        self.z_f = torch.tensor(z_f, dtype=torch.float32, requires_grad=True, device=dev)

    def compute_loss(self):
        u_b, v_b, w_b = self._forward(self.x_b, self.y_b, self.z_b)
        loss_bc = torch.mean((u_b - self.x_u)**2) + torch.mean((v_b - self.y_v)**2) + torch.mean((w_b - self.z_w)**2)
        eq1, eq2, eq3 = self._pde(self.x_f, self.y_f, self.z_f)
        loss_eq = torch.mean(eq1**2) + torch.mean(eq2**2) + torch.mean(eq3**2)
        return loss_eq + 10 * loss_bc, loss_eq, loss_bc

    def train(self, epochs=3000, print_every=100):
        history = []
        for ep in range(epochs):
            self.opt.zero_grad()
            loss, le, lb = self.compute_loss()
            loss.backward(); self.opt.step()
            history.append(float(loss.detach().cpu()))
            if ep == 0 or (ep + 1) % print_every == 0 or ep == epochs - 1:
                print(f"Epoch {ep+1:5d}/{epochs} | total={history[-1]:.3e} eq={le.item():.3e} bc={lb.item():.3e}")
        return history

    def predict(self, x, y, z):
        self.net.eval()
        with torch.no_grad():
            xx = torch.tensor(x.reshape(-1, 1), dtype=torch.float32, device=self.device)
            yy = torch.tensor(y.reshape(-1, 1), dtype=torch.float32, device=self.device)
            zz = torch.tensor(z.reshape(-1, 1), dtype=torch.float32, device=self.device)
            u, v, w = self._forward(xx, yy, zz)
        return u.cpu().numpy(), v.cpu().numpy(), w.cpu().numpy()


# =========================
# 5. Visualization
# =========================

def show_train_result(pinn, eval_data, history):
    x_s, y_s, z_s, u_s, v_s, w_s = eval_data
    if history:
        plt.figure(figsize=(7,4)); plt.plot(history); plt.yscale("log")
        plt.xlabel("Epoch"); plt.ylabel("Total Loss"); plt.title("Training Loss")
        plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

    u_pred, v_pred, w_pred = pinn.predict(x_s, y_s, z_s)
    eu = np.linalg.norm(u_s.reshape(-1) - u_pred.reshape(-1)) / (np.linalg.norm(u_s.reshape(-1)) + 1e-12)
    ev = np.linalg.norm(v_s.reshape(-1) - v_pred.reshape(-1)) / (np.linalg.norm(v_s.reshape(-1)) + 1e-12)
    ew = np.linalg.norm(w_s.reshape(-1) - w_pred.reshape(-1)) / (np.linalg.norm(w_s.reshape(-1)) + 1e-12)

    fig = plt.figure(figsize=(18, 8))
    for idx, (ref, pred, title) in enumerate([
        (u_s, u_pred, "U (Ref)"), (v_s, v_pred, "V (Ref)"), (w_s, w_pred, "W (Ref)"),
        (u_pred, u_pred, "U (PINN)"), (v_pred, v_pred, "V (PINN)"), (w_pred, w_pred, "W (PINN)"),
    ]):
        ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
        sc = ax.scatter(y_s.reshape(-1), x_s.reshape(-1), z_s.reshape(-1),
                         c=ref.reshape(-1) if idx < 3 else pred.reshape(-1), cmap='jet', s=5)
        ax.set_xlabel("y"); ax.set_ylabel("x"); ax.set_zlabel("z"); ax.set_title(title)
        ax.set_box_aspect([15, 5, 5])
    plt.suptitle(f"Airfoil Force — L2: u={eu:.3e} v={ev:.3e} w={ew:.3e}", fontsize=14)
    plt.tight_layout(); plt.show()
    print(f"Relative L2: u={eu:.4e} v={ev:.4e} w={ew:.4e}")
    return eu, ev, ew


def load_checkpoint(model_path, device):
    try:
        return torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(model_path, map_location=device)


# =========================
# 6. Train entry
# =========================

def train(data_path=DATA_PATH, model_path=MODEL_PATH, E=100, nu=0.3,
          hidden=50, layers=6, epochs=3000, lr=1e-3, print_every=100,
          force_retrain=FORCE_RETRAIN):
    t0 = time.time()
    model_path = os.path.expanduser(model_path)
    data_path = os.path.expanduser(data_path) if data_path else ""
    candidates = []
    if data_path:
        candidates.append(data_path)
        if os.path.isdir(data_path) or not os.path.splitext(data_path)[1]:
            candidates.append(os.path.join(data_path, DATA_FILENAME))
    candidates.extend([
        os.path.expanduser(f"~/public/Resource/{DATA_HASH}/{DATA_FILENAME}"),
        os.path.expanduser(f"~/Resource/{DATA_HASH}/{DATA_FILENAME}"),
        f"/Resource/{DATA_HASH}/{DATA_FILENAME}",
    ])
    for candidate in candidates:
        candidate = os.path.abspath(os.path.normpath(candidate))
        if os.path.isfile(candidate):
            data_path = candidate
            break
    else:
        raise FileNotFoundError("Data file not found in public assets. Expected one of: " + ", ".join(candidates))
    print(f"data={data_path}")

    bd, eq_d, eval_d = load_mat_data(data_path)
    print(f"boundary={bd[0].shape[0]}, interior={eq_d[0].shape[0]}, eval={eval_d[3].shape[0]}")

    if os.path.isfile(model_path) and not force_retrain:
        print(f"Found existing model, skip training: {model_path}")
        ckpt = load_checkpoint(model_path, get_device())
        hidden = ckpt.get("hidden", hidden) if isinstance(ckpt, dict) else hidden
        layers = ckpt.get("layers", layers) if isinstance(ckpt, dict) else layers
        E = ckpt.get("E", E) if isinstance(ckpt, dict) else E
        nu = ckpt.get("nu", nu) if isinstance(ckpt, dict) else nu
        state = ckpt.get("state_dict") if isinstance(ckpt, dict) else ckpt
        history = ckpt.get("history", []) if isinstance(ckpt, dict) else []
        pinn = AirfoilForcePINN(E=E, nu=nu, hidden=hidden, layers=layers, lr=lr)
        pinn.net.load_state_dict(state)
        show_train_result(pinn, eval_d, history)
        return {"model": pinn, "model_path": model_path, "history": history, "skipped_training": True}

    pinn = AirfoilForcePINN(E=E, nu=nu, hidden=hidden, layers=layers, lr=lr)
    pinn.set_data(bd, eq_d)
    print(f"Training {epochs} epochs (E={E}, nu={nu})...")
    history = pinn.train(epochs=epochs, print_every=print_every)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save({"state_dict": pinn.net.state_dict(), "history": history,
                "E": E, "nu": nu, "hidden": hidden, "layers": layers}, model_path)

    print("Training finished"); print(f"model saved to: {model_path}")
    print(f"elapsed: {time.time() - t0:.1f}s")
    show_train_result(pinn, eval_d, history)
    return {"model": pinn, "model_path": model_path, "history": history}


train_result = train()
