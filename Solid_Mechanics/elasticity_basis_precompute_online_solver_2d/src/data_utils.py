import os

import numpy as np


def exact_ux(x, y):
    return np.cos(2 * np.pi * x) * np.sin(np.pi * y)


def exact_uy(x, y, q):
    return np.sin(np.pi * x) * (q / 4.0) * (y**4)


def generate_sim_data(n_samples=4096, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.random((n_samples, 1), dtype=np.float32)
    y = rng.random((n_samples, 1), dtype=np.float32)
    q = rng.uniform(0.1, 10.0, size=(n_samples, 1)).astype(np.float32)
    ux = exact_ux(x, y).astype(np.float32)
    uy = exact_uy(x, y, q).astype(np.float32)
    return {"x": x, "y": y, "q": q, "ux": ux, "uy": uy}


def load_real_data(npz_path):
    data = np.load(npz_path)
    required = ["x", "y", "q", "ux", "uy"]
    for k in required:
        if k not in data:
            raise ValueError(f"missing key: {k}")
    return {k: data[k].astype(np.float32) for k in required}


def save_sample_real_data(path, n_samples=1024, seed=7):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    d = generate_sim_data(n_samples=n_samples, seed=seed)
    np.savez(path, **d)


def to_xyq_and_u(data):
    x = data["x"]
    y = data["y"]
    q = data["q"]
    ux = data["ux"]
    uy = data["uy"]
    inp = np.hstack([x, y, q]).astype(np.float32)
    out = np.hstack([ux, uy]).astype(np.float32)
    return inp, out