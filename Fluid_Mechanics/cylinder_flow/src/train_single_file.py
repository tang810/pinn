"""
Notebook-friendly train file for cylinder_flow (Re=100).

Online Jupyter usage:
1. Upload simul_cylinder_re100.mat as a public asset.
2. Fill DATA_HASH and MODEL_HASH.
3. Run this whole file/cell. It saves trained_model_cylinder_flow.pt to
   ~/minio/Resource/{MODEL_HASH}/ for later public-asset upload.

Physics: 2D unsteady Navier-Stokes (Re=100) around a cylinder.
         Outputs: (u, v, p) velocity and pressure fields.

Self-contained — only depends on the .mat dataset.
"""

import math
import os
import time

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

DATA_HASH = "6f9ab9dc1b784b8287cd41287a663053"
MODEL_HASH = "6f9ab9dc1b784b8287cd41287a663053"
DATA_FILENAME = "simul_cylinder_re100.mat"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/trained_model_cylinder_flow.pt")

DOMAIN_X = (-7.5, 22.5)
DOMAIN_Y = (-10.0, 10.0)
CYLINDER_R = 0.5
RE = 100.0


# =========================
# 2. Geometry sampling (inline)
# =========================

def sample_boundary_line(x_val, y_min, y_max, n_points):
    """Sample points on a vertical line at x_val."""
    y = np.random.uniform(y_min, y_max, n_points)
    x = np.full(n_points, x_val)
    return x.reshape(-1, 1), y.reshape(-1, 1)


def sample_boundary_wall(x_min, x_max, y_val, n_points):
    """Sample points on a horizontal wall at y_val."""
    x = np.random.uniform(x_min, x_max, n_points)
    y = np.full(n_points, y_val)
    return x.reshape(-1, 1), y.reshape(-1, 1)


def sample_cylinder(n_points):
    """Sample points on cylinder surface (r=0.5)."""
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    x = CYLINDER_R * np.cos(theta)
    y = CYLINDER_R * np.sin(theta)
    return x.reshape(-1, 1), y.reshape(-1, 1)


def sample_interior(n_points):
    """Sample interior points excluding the cylinder."""
    collected = 0
    xs, ys = [], []
    while collected < n_points:
        batch = int((n_points - collected) * 2.5 + 500)
        xb = np.random.uniform(*DOMAIN_X, batch)
        yb = np.random.uniform(*DOMAIN_Y, batch)
        mask = (xb**2 + yb**2) >= CYLINDER_R**2
        xs.append(xb[mask]); ys.append(yb[mask])
        collected = sum(len(a) for a in xs)
    x = np.concatenate(xs)[:n_points].reshape(-1, 1)
    y = np.concatenate(ys)[:n_points].reshape(-1, 1)
    return x, y


def tile_to_time(x, y, t_star):
    """Tile spatial points to all time frames: (N,) -> (N*T,)."""
    T = len(t_star)
    xx = np.tile(x.reshape(-1, 1), (1, T)).reshape(-1, 1)
    yy = np.tile(y.reshape(-1, 1), (1, T)).reshape(-1, 1)
    tt = np.tile(t_star.reshape(1, -1), (x.shape[0], 1)).reshape(-1, 1)
    return xx, yy, tt


# =========================
# 3. Data loading from .mat
# =========================

def generate_small_mat(filename, n_points=500, n_steps=10):
    """Generate a small synthetic cylinder-flow .mat file."""
    xs, ys = [], []
    while len(xs) < n_points:
        bx = np.random.uniform(-7.5, 22.5, n_points * 3)
        by = np.random.uniform(-10.0, 10.0, n_points * 3)
        mask = (bx**2 + by**2) >= 0.5**2
        xs.extend(bx[mask]); ys.extend(by[mask])
    X_star = np.column_stack([xs[:n_points], ys[:n_points]]).astype(np.float32)
    t = np.linspace(0.0, 6.0, n_steps, dtype=np.float32).reshape(-1, 1)
    U_star = np.zeros((n_points, 2, n_steps), dtype=np.float32)
    P_star = np.zeros((n_points, n_steps), dtype=np.float32)
    x_arr = X_star[:, 0:1]; y_arr = X_star[:, 1:2]
    for i in range(n_steps):
        ti = t[i, 0]
        u = 1.0 - 0.42 * np.exp(-((x_arr-0.8)**2+(1.4*y_arr)**2)/3.0) * (1.0+0.18*np.sin(1.8*ti))
        v = 0.15 * np.exp(-((x_arr-1.0)**2+y_arr**2)/4.5) * np.sin(0.35*y_arr+1.6*ti)
        p = 0.35 * np.exp(-((x_arr+0.2)**2+y_arr**2)/0.8) * np.cos(1.2*ti) - 0.015*x_arr
        U_star[:, 0, i] = u.reshape(-1); U_star[:, 1, i] = v.reshape(-1)
        P_star[:, i] = p.reshape(-1)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    scipy.io.savemat(filename, {'X_star': X_star, 'U_star': U_star, 'P_star': P_star, 't': t})
    return {'X_star': X_star, 'U_star': U_star, 'P_star': P_star, 't': t}


def load_mat_data(filename, time_window=5):
    if not os.path.isfile(filename):
        print('Generating small synthetic dataset...')
        data = generate_small_mat(filename)
    else:
        data = scipy.io.loadmat(filename)
    return {
        "U_star": data["U_star"],  # N x 2 x T
        "P_star": data["P_star"],  # N x T
        "t": data["t"][:time_window],  # T x 1
        "t_all": data["t"],  # T_all x 1
        "X_star": data["X_star"],  # N x 2
    }


def _script_dir():
    """Return the current notebook/script working directory."""
    return os.getcwd()


def resolve_data_path(data_path, data_hash=DATA_HASH, data_filename=DATA_FILENAME):
    """Resolve local/Jupyter asset paths to the actual .mat file."""
    base_dir = _script_dir()
    raw_path = os.path.expanduser(data_path) if data_path else ""
    candidates = []

    if raw_path:
        candidates.append(raw_path)
        if os.path.isdir(raw_path) or not os.path.splitext(raw_path)[1]:
            candidates.append(os.path.join(raw_path, data_filename))

    candidates.extend([
        os.path.join(base_dir, "..", "data", data_filename),
        os.path.join(base_dir, "data", data_filename),
        os.path.join(os.getcwd(), data_filename),
        os.path.join(os.getcwd(), "data", data_filename),
        os.path.expanduser(f"~/public/Resource/{data_hash}/{data_filename}"),
        os.path.expanduser(f"~/Resource/{data_hash}/{data_filename}"),
        f"/Resource/{data_hash}/{data_filename}",
    ])

    for candidate in candidates:
        candidate = os.path.abspath(os.path.normpath(candidate))
        if os.path.isfile(candidate):
            return candidate

    fallback = os.path.abspath(os.path.normpath(os.path.join(base_dir, "..", "data", data_filename)))
    print(f"Warning: data file not found, will create a small synthetic dataset at: {fallback}")
    return fallback


# =========================
# 4. Model definition (FCNet)
# =========================

class FCNet(nn.Module):
    def __init__(self, num_ins=3, num_outs=3, hidden_size=120, num_layers=6):
        super().__init__()
        layers = [nn.Linear(num_ins, hidden_size), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers += [nn.Linear(hidden_size, num_outs)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# =========================
# 5. PINN solver
# =========================

class CylinderPINN:
    def __init__(self, re=100, hidden=120, layers=6, lr=1e-3):
        self.Re = re; self.vis0 = 1.0 / re
        self.device = get_device()
        self.net = FCNet(num_ins=3, num_outs=3, hidden_size=hidden, num_layers=layers).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.vis_minus = 0.0

    def _forward(self, t, x, y):
        X = torch.cat([t, x, y], dim=1)
        out = self.net(X)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]

    def _grad(self, y, xs):
        g = torch.autograd.grad(y, xs, [torch.ones_like(y)] * len(xs), create_graph=True)
        return [gi if gi is not None else torch.zeros_like(xs[i]) for i, gi in enumerate(g)]

    def _pde(self, t, x, y):
        # Detach to avoid "backward through the graph a second time" across epochs
        t = t.detach().requires_grad_(True)
        x = x.detach().requires_grad_(True)
        y = y.detach().requires_grad_(True)
        u, v, p = self._forward(t, x, y)
        u_t, u_x, u_y = self._grad(u, [t, x, y])
        u_xx = self._grad(u_x, [x])[0]; u_yy = self._grad(u_y, [y])[0]
        v_t, v_x, v_y = self._grad(v, [t, x, y])
        v_xx = self._grad(v_x, [x])[0]; v_yy = self._grad(v_y, [y])[0]
        p_x, p_y = self._grad(p, [x, y])
        vis = max(self.vis0, self.vis_minus)
        eq1 = u_t + u * u_x + v * u_y + p_x - vis * (u_xx + u_yy)
        eq2 = v_t + u * v_x + v * v_y + p_y - vis * (v_xx + v_yy)
        eq3 = u_x + v_y
        return eq1, eq2, eq3

    def set_data(self, t_i, x_i, y_i, u_i, v_i, p_i,
                 t_b, x_b, y_b, u_b, v_b,
                 t_o, x_o, y_o, p_o,
                 t_f, x_f, y_f):
        self.t_i, self.x_i, self.y_i = [torch.tensor(d, dtype=torch.float32, device=self.device) for d in [t_i, x_i, y_i]]
        self.u_i = torch.tensor(u_i, dtype=torch.float32, device=self.device)
        self.v_i = torch.tensor(v_i, dtype=torch.float32, device=self.device)
        self.p_i = torch.tensor(p_i, dtype=torch.float32, device=self.device)
        self.t_b, self.x_b, self.y_b = [torch.tensor(d, dtype=torch.float32, device=self.device) for d in [t_b, x_b, y_b]]
        self.u_b = torch.tensor(u_b, dtype=torch.float32, device=self.device)
        self.v_b = torch.tensor(v_b, dtype=torch.float32, device=self.device)
        self.t_o, self.x_o, self.y_o = [torch.tensor(d, dtype=torch.float32, device=self.device) for d in [t_o, x_o, y_o]]
        self.p_o = torch.tensor(p_o, dtype=torch.float32, device=self.device)
        self.t_f = torch.tensor(t_f, dtype=torch.float32, requires_grad=True, device=self.device)
        self.x_f = torch.tensor(x_f, dtype=torch.float32, requires_grad=True, device=self.device)
        self.y_f = torch.tensor(y_f, dtype=torch.float32, requires_grad=True, device=self.device)

    def compute_loss(self):
        u_i, v_i, p_i = self._forward(self.t_i, self.x_i, self.y_i)
        loss_ic = torch.mean((u_i - self.u_i)**2) + 10 * torch.mean((v_i - self.v_i)**2)

        u_b, v_b, _ = self._forward(self.t_b, self.x_b, self.y_b)
        loss_bc = torch.mean((u_b - self.u_b)**2) + 10 * torch.mean((v_b - self.v_b)**2)

        _, _, p_o = self._forward(self.t_o, self.x_o, self.y_o)
        loss_out = torch.mean((p_o - self.p_o)**2)

        eq1, eq2, eq3 = self._pde(self.t_f, self.x_f, self.y_f)
        loss_eq = torch.mean(eq1**2) + torch.mean(eq2**2) + torch.mean(eq3**2)

        return loss_eq + 10 * loss_bc + 10 * loss_ic + 10 * loss_out, loss_eq, loss_bc, loss_ic, loss_out

    def train(self, epochs=5000, print_every=500):
        history = []
        for ep in range(1, epochs + 1):
            self.opt.zero_grad()
            loss, le, lb, li, lo = self.compute_loss()
            loss.backward(); self.opt.step()
            history.append(float(loss.detach().cpu()))
            if ep == 1 or ep % print_every == 0 or ep == epochs:
                print(f"Epoch {ep:5d}/{epochs} | total={history[-1]:.4e} eq={le.item():.3e} bc={lb.item():.3e} ic={li.item():.3e} out={lo.item():.3e}")
        return history

    def predict(self, t, x, y):
        self.net.eval()
        with torch.no_grad():
            tt = torch.tensor(t, dtype=torch.float32, device=self.device)
            xx = torch.tensor(x, dtype=torch.float32, device=self.device)
            yy = torch.tensor(y, dtype=torch.float32, device=self.device)
            u, v, p = self._forward(tt, xx, yy)
        return u.cpu().numpy(), v.cpu().numpy(), p.cpu().numpy()


# =========================
# 6. Visualization
# =========================

def show_train_result(pinn, data, history):
    if history:
        plt.figure(figsize=(7,4)); plt.plot(history); plt.yscale("log")
        plt.xlabel("Epoch"); plt.ylabel("Total Loss"); plt.title("Training Loss")
        plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

    snap = min(10, len(data["t_all"]))
    X = data["X_star"]
    x_eval = np.tile(X[:, 0:1], (1, snap)).reshape(-1, 1)
    y_eval = np.tile(X[:, 1:2], (1, snap)).reshape(-1, 1)
    t_eval = np.tile(data["t_all"][:snap].reshape(1, -1), (X.shape[0], 1)).reshape(-1, 1)
    u_true = data["U_star"][:, 0, :snap].reshape(-1, 1)
    v_true = data["U_star"][:, 1, :snap].reshape(-1, 1)
    p_true = data["P_star"][:, :snap].reshape(-1, 1)

    u_pred, v_pred, p_pred = pinn.predict(t_eval, x_eval, y_eval)
    speed_t = np.sqrt(u_true.reshape(-1)**2 + v_true.reshape(-1)**2)
    speed_p = np.sqrt(u_pred.reshape(-1)**2 + v_pred.reshape(-1)**2)

    def _sc(ax, val, title):
        sc = ax.scatter(x_eval.reshape(-1), y_eval.reshape(-1), c=val, s=1, cmap="jet")
        ax.set_title(title); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
        plt.colorbar(sc, ax=ax, shrink=0.8)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    _sc(axes[0, 0], speed_t, "Speed True"); _sc(axes[0, 1], speed_p, "Speed Pred")
    _sc(axes[1, 0], p_true.reshape(-1), "Pressure True"); _sc(axes[1, 1], p_pred.reshape(-1), "Pressure Pred")
    fig.suptitle("Cylinder Flow (Re=100) — PINN", fontsize=14); plt.tight_layout(); plt.show()

    eu = np.linalg.norm(u_true - u_pred) / (np.linalg.norm(u_true) + 1e-12)
    ev = np.linalg.norm(v_true - v_pred) / (np.linalg.norm(v_true) + 1e-12)
    ep = np.linalg.norm(p_true - p_pred) / (np.linalg.norm(p_true) + 1e-12)
    fig2, ax = plt.subplots(figsize=(6,4))
    ax.bar(["u","v","p"], [eu, ev, ep])
    ax.set_title("Relative L2 Error"); ax.set_ylabel("Relative L2")
    for i, v in enumerate([eu, ev, ep]): ax.text(i, v, f"{v:.3e}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout(); plt.show()
    return eu, ev, ep


# =========================
# 7. Train entry
# =========================

def train(data_path=DATA_PATH, model_path=MODEL_PATH,
          time_window=5, n_bdry=512, n_interior=2000, n_wall=256,
          epochs=5000, hidden=120, layers=6, lr=1e-3, re=100, print_every=500):
    t0 = time.time()
    print(f"device={'cuda' if torch.cuda.is_available() else 'cpu'}")

    data_path = resolve_data_path(data_path)
    data = load_mat_data(data_path, time_window=time_window)
    t_star = data["t"].reshape(-1)
    print(f"data={data_path}, t_steps={len(t_star)}, N={data['X_star'].shape[0]}")

    # ---- geometry sampling ----
    ix, iy = sample_boundary_line(DOMAIN_X[0], DOMAIN_Y[0], DOMAIN_Y[1], n_bdry)
    ox, oy = sample_boundary_line(DOMAIN_X[1], DOMAIN_Y[0], DOMAIN_Y[1], n_bdry)
    wbx, wby = sample_boundary_wall(DOMAIN_X[0], DOMAIN_X[1], DOMAIN_Y[0], n_wall)
    wtx, wty = sample_boundary_wall(DOMAIN_X[0], DOMAIN_X[1], DOMAIN_Y[1], n_wall)
    cx, cy = sample_cylinder(n_bdry * 2)

    bx = np.concatenate([ix.reshape(-1), cx.reshape(-1), wbx.reshape(-1), wtx.reshape(-1)]).reshape(-1, 1)
    by = np.concatenate([iy.reshape(-1), cy.reshape(-1), wby.reshape(-1), wty.reshape(-1)]).reshape(-1, 1)
    bu = np.concatenate([np.ones(ix.shape[0]), np.zeros(cx.shape[0]), np.ones(wbx.shape[0]), np.ones(wtx.shape[0])]).reshape(-1, 1)
    bv = np.zeros_like(bu)

    # Tile to time
    bxx, byy, btt = tile_to_time(bx, by, t_star)
    buu = np.tile(bu.reshape(-1, 1), (1, len(t_star))).reshape(-1, 1)
    bvv = np.tile(bv.reshape(-1, 1), (1, len(t_star))).reshape(-1, 1)

    oxx, oyy, ott = tile_to_time(ox, oy, t_star)
    opp = np.zeros_like(oxx)

    fx, fy = sample_interior(n_interior)
    fxx, fyy, ftt = tile_to_time(fx, fy, t_star)

    # Initial condition
    X = data["X_star"]; N = X.shape[0]
    ixx = X[:, 0:1]; iyy = X[:, 1:2]; itt = np.full((N, 1), t_star[0])
    iuu = data["U_star"][:, 0, 0].reshape(-1, 1)
    ivv = data["U_star"][:, 1, 0].reshape(-1, 1)
    ipp = data["P_star"][:, 0].reshape(-1, 1)

    pinn = CylinderPINN(re=re, hidden=hidden, layers=layers, lr=lr)
    pinn.set_data(itt, ixx, iyy, iuu, ivv, ipp, btt, bxx, byy, buu, bvv, ott, oxx, oyy, opp, ftt, fxx, fyy)

    print(f"Training {epochs} epochs (Re={re})...")
    history = pinn.train(epochs=epochs, print_every=print_every)

    model_path = os.path.expanduser(model_path)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save({"state_dict": pinn.net.state_dict(), "history": history, "re": re,
                "hidden": hidden, "layers": layers}, model_path)

    print("Training finished"); print(f"model saved to: {model_path}")
    print(f"elapsed: {time.time() - t0:.1f}s"); print(f"final loss: {history[-1]:.6e}")

    eu, ev, ep = show_train_result(pinn, data, history)
    print(f"Relative L2: u={eu:.4e} v={ev:.4e} p={ep:.4e}")
    return {"model": pinn, "model_path": model_path, "history": history}


train_result = train()
