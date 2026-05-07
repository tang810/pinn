from pathlib import Path

import numpy as np
import scipy.io


def _sample_points(n_points, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-7.5, 22.5, size=(n_points * 2, 1))
    y = rng.uniform(-10.0, 10.0, size=(n_points * 2, 1))
    keep = (x**2 + y**2).reshape(-1) >= 0.5**2
    x = x[keep][:n_points]
    y = y[keep][:n_points]
    if x.shape[0] < n_points:
        pad = n_points - x.shape[0]
        x_pad = rng.uniform(-7.5, 22.5, size=(pad, 1))
        y_pad = rng.uniform(-10.0, 10.0, size=(pad, 1))
        x = np.vstack([x, x_pad])
        y = np.vstack([y, y_pad])
    return np.hstack([x, y]).astype(np.float32)


def create_simul_mat(path, n_points=3000, n_steps=60, seed=42):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    x_star = _sample_points(n_points=n_points, seed=seed)
    x = x_star[:, 0:1]
    y = x_star[:, 1:2]
    t = np.linspace(0.0, 6.0, n_steps, dtype=np.float32).reshape(-1, 1)

    u = np.zeros((n_points, n_steps), dtype=np.float32)
    v = np.zeros((n_points, n_steps), dtype=np.float32)
    p = np.zeros((n_points, n_steps), dtype=np.float32)

    for i in range(n_steps):
        ti = t[i, 0]
        wake = np.exp(-((x - 0.8) ** 2 + (0.6 * y) ** 2) / 3.0)
        core = np.exp(-((x + 0.2) ** 2 + y**2) / 0.8)
        u[:, i] = (
            1.0
            - 0.42 * wake.reshape(-1) * (1.0 + 0.18 * np.sin(1.8 * ti))
            + 0.03 * np.sin(0.25 * x.reshape(-1) + 0.4 * ti)
        )
        v[:, i] = (
            0.15
            * np.exp(-((x - 1.0) ** 2 + y**2) / 4.5).reshape(-1)
            * np.sin(0.35 * y.reshape(-1) + 1.6 * ti)
        )
        p[:, i] = (
            0.35 * core.reshape(-1) * np.cos(1.2 * ti)
            - 0.015 * x.reshape(-1)
            + 0.02 * np.sin(0.2 * y.reshape(-1) + 0.7 * ti)
        )

    u_star = np.stack([u, v], axis=1).astype(np.float32)  # N x 2 x T
    scipy.io.savemat(
        str(path),
        {
            "U_star": u_star,
            "P_star": p.astype(np.float32),
            "t": t.astype(np.float32),
            "X_star": x_star.astype(np.float32),
        },
    )
    return path


def ensure_simul_data(path, seed=42):
    path = Path(path)
    if not path.exists():
        create_simul_mat(path=path, seed=seed)
    return path
