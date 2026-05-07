import json
import os

import numpy as np
import torch


def generate_sim_data(t_size=21, x_size=64, v_size=64):
    t = np.linspace(0, 6.25, t_size)
    x = np.linspace(0, 10, x_size)
    v = np.linspace(-5, 5, v_size)

    T, X, V = np.meshgrid(t, x, v, indexing="ij")
    E = np.sin(2 * np.pi * X / 10.0) * np.cos(T)
    f = np.exp(-V**2 / 2.0) * (1.0 + 0.1 * np.sin(2 * np.pi * X / 10.0) * np.cos(T))

    return {
        "t": t,
        "x": x,
        "v": v,
        "E": E.astype(np.float32),
        "f": f.astype(np.float32),
    }


def load_real_data(data_dir):
    E = np.load(os.path.join(data_dir, "E.npy"))
    f = np.load(os.path.join(data_dir, "f.npy"))
    t = np.linspace(0, 62.5, E.shape[0])
    x = np.linspace(0, 10, E.shape[1])
    v = np.linspace(-5, 5, E.shape[2])
    return {"t": t, "x": x, "v": v, "E": E.astype(np.float32), "f": f.astype(np.float32)}


def build_training_set(data, n_u=70000, seed=42):
    t = data["t"]
    x = data["x"]
    v = data["v"]
    E = data["E"]
    f = data["f"]

    T, X, V = np.meshgrid(t, x, v, indexing="ij")
    X_star = np.hstack((T.flatten()[:, None], X.flatten()[:, None], V.flatten()[:, None]))
    f_star = f.flatten()[:, None]

    total = X_star.shape[0]
    n_use = min(n_u, total)
    rng = np.random.default_rng(seed)
    idx = rng.choice(total, n_use, replace=False)

    X_u_train = X_star[idx, :]
    idx_on_txv = np.unravel_index(idx, f.shape)
    E_train = E[idx_on_txv[0], idx_on_txv[1], idx_on_txv[2]][:, None]
    f_train = f_star[idx, :]

    return X_u_train.astype(np.float32), E_train.astype(np.float32), f_train.astype(np.float32)


def build_prediction_grid(data, device):
    t = data["t"]
    x = data["x"]
    v = data["v"]
    T, X, V = np.meshgrid(t, x, v, indexing="ij")
    T_star = torch.tensor(T.flatten()[:, None], dtype=torch.float32, device=device)
    X_star = torch.tensor(X.flatten()[:, None], dtype=torch.float32, device=device)
    V_star = torch.tensor(V.flatten()[:, None], dtype=torch.float32, device=device)
    return T, X, V, T_star, X_star, V_star


def save_data_info(data, output_path, source):
    info = {
        "source": source,
        "shape_E": list(data["E"].shape),
        "shape_f": list(data["f"].shape),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)