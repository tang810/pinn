import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# =========================
# 1. 自动查找数据集
# =========================

data_path = None

for root, dirs, files in os.walk("/mnt/minio/userdk8e2v7l"):
    if "heat_eq_data.npz" in files:
        data_path = os.path.join(root, "heat_eq_data.npz")
        break

print("Using dataset:", data_path)
print("exists:", os.path.exists(data_path) if data_path else False)

if data_path is None:
    raise FileNotFoundError("heat_eq_data.npz not found")




# =========================
# 2. 工具函数
# =========================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_kdv_npz(data_path):
    data = np.load(data_path)

    x = data["x"].reshape(-1).astype(np.float32)
    t = data["t"].reshape(-1).astype(np.float32)
    u = data["usol"].astype(np.float32)

    if u.shape != (len(x), len(t)):
        raise ValueError(f"usol shape 应为 {(len(x), len(t))}, 当前是 {u.shape}")

    X_grid, T_grid = np.meshgrid(x, t, indexing="ij")
    X_raw = np.stack([X_grid.reshape(-1), T_grid.reshape(-1)], axis=1)
    y_raw = u.reshape(-1, 1)

    x_min, x_max = float(x.min()), float(x.max())
    t_min, t_max = float(t.min()), float(t.max())
    u_mean = float(y_raw.mean())
    u_std = float(y_raw.std() + 1e-8)

    X = X_raw.copy()
    X[:, 0] = 2.0 * (X[:, 0] - x_min) / (x_max - x_min) - 1.0
    X[:, 1] = 2.0 * (X[:, 1] - t_min) / (t_max - t_min) - 1.0
    y = (y_raw - u_mean) / u_std

    meta = {
        "x": x,
        "t": t,
        "u": u,
        "x_min": x_min,
        "x_max": x_max,
        "t_min": t_min,
        "t_max": t_max,
        "u_mean": u_mean,
        "u_std": u_std,
    }

    return X.astype(np.float32), y.astype(np.float32), meta


# =========================
# 3. 模型定义
# =========================

class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=64, n_layers=4):
        super().__init__()

        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, 1)]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class KDVDataPINN(nn.Module):
    def __init__(self, hidden_dim=64, n_layers=4):
        super().__init__()
        self.net = MLP(2, hidden_dim, n_layers)

        self.c0_raw = nn.Parameter(torch.tensor(0.0))
        self.c1_raw = nn.Parameter(torch.tensor(0.0))
        self.c2_raw = nn.Parameter(torch.tensor(-2.0))

    def forward(self, xt):
        return self.net(xt)

    def coefficients(self):
        c0 = self.c0_raw
        c1 = self.c1_raw
        c2 = torch.nn.functional.softplus(self.c2_raw)
        return c0, c1, c2


# =========================
# 4. PDE loss
# =========================

def pde_residual(model, xt):
    xt = xt.clone().detach().requires_grad_(True)

    u = model(xt)

    grad_u = torch.autograd.grad(
        u,
        xt,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True
    )[0]

    u_x = grad_u[:, 0:1]
    u_t = grad_u[:, 1:2]

    grad_u_x = torch.autograd.grad(
        u_x,
        xt,
        grad_outputs=torch.ones_like(u_x),
        retain_graph=True,
        create_graph=True
    )[0]

    u_xx = grad_u_x[:, 0:1]

    grad_u_xx = torch.autograd.grad(
        u_xx,
        xt,
        grad_outputs=torch.ones_like(u_xx),
        retain_graph=True,
        create_graph=True
    )[0]

    u_xxx = grad_u_xx[:, 0:1]

    c0, c1, c2 = model.coefficients()

    residual = u_t + c0 * u_x + c1 * u * u_x + c2 * u_xxx
    return residual


# =========================
# 5. 训练函数
# =========================

def sample_batch(X, y, n, device):
    idx = np.random.choice(X.shape[0], size=min(n, X.shape[0]), replace=False)
    xb = torch.tensor(X[idx], dtype=torch.float32, device=device)
    yb = torch.tensor(y[idx], dtype=torch.float32, device=device)
    return xb, yb

def train_kdv_pinn(
    data_path,
    n_iters=1000,
    print_every=100,
    n_data=2048,
    n_pde=1024,
    hidden_dim=64,
    n_layers=4,
    lr=1e-3,
    seed=42,
    w_data=10.0,
    w_pde=1.0,
):
    set_seed(seed)

    device = get_device()
    X, y, meta = load_kdv_npz(data_path)

    model = KDVDataPINN(hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []

    print("Using device:", device)
    print("Loaded data:", X.shape, y.shape)

    for it in range(1, n_iters + 1):
        xb, yb = sample_batch(X, y, n_data, device)
        xf, _ = sample_batch(X, y, n_pde, device)

        pred = model(xb)
        loss_data = torch.mean((pred - yb) ** 2)

        residual = pde_residual(model, xf)
        loss_pde = torch.mean(residual ** 2)

        loss = w_data * loss_data + w_pde * loss_pde

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it == 1 or it % print_every == 0 or it == n_iters:
            c0, c1, c2 = [float(v.detach().cpu()) for v in model.coefficients()]
            row = {
                "iter": it,
                "loss": float(loss.detach().cpu()),
                "data": float(loss_data.detach().cpu()),
                "pde": float(loss_pde.detach().cpu()),
                "c0": c0,
                "c1": c1,
                "c2": c2,
            }
            history.append(row)

            print(
                f"Iter {it:5d}/{n_iters} | "
                f"loss={row['loss']:.3e} | "
                f"data={row['data']:.3e} | "
                f"pde={row['pde']:.3e} | "
                f"c0={c0:.3e} | c1={c1:.3e} | c2={c2:.3e}"
            )

    return {
        "model": model,
        "history": history,
        "meta": meta,
        "device": device,
    }


# =========================
# 6. 预测与评估
# =========================

def predict_grid(result, nx=None, nt=None):
    model = result["model"]
    meta = result["meta"]
    device = result["device"]

    x = meta["x"] if nx is None else np.linspace(meta["x_min"], meta["x_max"], nx, dtype=np.float32)
    t = meta["t"] if nt is None else np.linspace(meta["t_min"], meta["t_max"], nt, dtype=np.float32)

    X_grid, T_grid = np.meshgrid(x, t, indexing="ij")

    xt = np.stack([X_grid.reshape(-1), T_grid.reshape(-1)], axis=1).astype(np.float32)

    xt_norm = xt.copy()
    xt_norm[:, 0] = 2.0 * (xt_norm[:, 0] - meta["x_min"]) / (meta["x_max"] - meta["x_min"]) - 1.0
    xt_norm[:, 1] = 2.0 * (xt_norm[:, 1] - meta["t_min"]) / (meta["t_max"] - meta["t_min"]) - 1.0

    model.eval()
    with torch.no_grad():
        u_norm = model(torch.tensor(xt_norm, dtype=torch.float32, device=device)).cpu().numpy()

    U_pred = u_norm.reshape(len(x), len(t)) * meta["u_std"] + meta["u_mean"]

    return X_grid, T_grid, U_pred

def evaluate(result):
    X_grid, T_grid, U_pred = predict_grid(result)
    U_true = result["meta"]["u"]

    rel_l2 = np.linalg.norm(U_pred - U_true) / (np.linalg.norm(U_true) + 1e-12)
    mae = np.mean(np.abs(U_pred - U_true))

    return {
        "relative_l2": float(rel_l2),
        "mae": float(mae),
    }


# =========================
# 7. 开始训练
# =========================

result = train_kdv_pinn(
    data_path=data_path,
    n_iters=1000,
    print_every=100,
    n_data=2048,
    n_pde=1024,
    hidden_dim=64,
    n_layers=4,
    lr=1e-3,
)

metrics = evaluate(result)
print("Evaluation:", metrics)


# =========================
# 8. 可视化结果
# =========================

X_grid, T_grid, U_pred = predict_grid(result)
U_true = result["meta"]["u"]

plt.figure(figsize=(7, 4))
plt.pcolormesh(T_grid, X_grid, U_pred, shading="auto", cmap="viridis")
plt.colorbar(label="u_pred(x,t)")
plt.xlabel("t")
plt.ylabel("x")
plt.title("KdV Data-PINN Prediction")
plt.show()

plt.figure(figsize=(7, 4))
plt.pcolormesh(T_grid, X_grid, U_true, shading="auto", cmap="viridis")
plt.colorbar(label="u_true(x,t)")
plt.xlabel("t")
plt.ylabel("x")
plt.title("Ground Truth")
plt.show()

plt.figure(figsize=(7, 4))
plt.pcolormesh(T_grid, X_grid, np.abs(U_pred - U_true), shading="auto", cmap="magma")
plt.colorbar(label="abs error")
plt.xlabel("t")
plt.ylabel("x")
plt.title("Absolute Error")
plt.show()

# 固定 t=1.0 的曲线对比
t_target = 1.0
t_idx = np.argmin(np.abs(result["meta"]["t"] - t_target))

plt.figure(figsize=(7, 4))
plt.plot(result["meta"]["x"], U_true[:, t_idx], label="true", linewidth=2)
plt.plot(result["meta"]["x"], U_pred[:, t_idx], "--", label="pred", linewidth=2)
plt.xlabel("x")
plt.ylabel("u")
plt.title(f"Curve Comparison at t={result['meta']['t'][t_idx]:.2f}")
plt.grid(alpha=0.3)
plt.legend()
plt.show()
