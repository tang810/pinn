import os
from typing import Dict

import numpy as np


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def _synthetic_airfoil_flow(n: int = 2048, seed: int = 42) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    alpha = rng.uniform(-5.0, 15.0, size=(n, 1))  # angle of attack
    reynolds = rng.uniform(1e5, 5e6, size=(n, 1))
    x = rng.uniform(0.0, 1.0, size=(n, 1))
    y = rng.uniform(-0.3, 0.3, size=(n, 1))

    features = np.hstack([alpha / 15.0, np.log10(reynolds) / 7.0, x, y]).astype(np.float32)
    # Lightweight surrogate targets: pressure, u, v
    p = (1.2 - x) * (1.0 + 0.02 * alpha) - 0.8 * y**2
    u = (1.0 + 0.05 * alpha) * (1.0 - 0.15 * y**2)
    v = 0.35 * y * (x - 0.5) + 0.01 * alpha
    targets = np.hstack([p, u, v]).astype(np.float32)
    return {"x": features, "y": targets}


def ensure_simul_dataset(path: str, seed: int = 42) -> None:
    if os.path.exists(path):
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    data = _synthetic_airfoil_flow(n=2048, seed=seed)
    np.savez(path, x=data["x"], y=data["y"])


def load_dataset(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {"x": data["x"].astype(np.float32), "y": data["y"].astype(np.float32)}

