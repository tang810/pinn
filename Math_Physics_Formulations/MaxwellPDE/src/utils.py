# src/utils.py
import os
import torch
import numpy as np


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def generate_training_data(
    N_int=1500,
    N_bc=450,
    device="cpu",
    save=False,
    save_path="data/training_data.npy",
):
    """
    Generate training data for Maxwell PINN

    Parameters
    ----------
    save : bool
        Whether to save data to disk
    save_path : str
        Path like 'data/training_data.npy'
    """

    # =========================
    # 内部点
    # =========================
    x_int = np.random.uniform(-1, 1, (N_int, 3))
    omega_int = np.random.uniform(0.5 * np.pi, 2 * np.pi, (N_int, 1))
    X_int = np.concatenate([x_int, omega_int], axis=1)

    # =========================
    # 边界点
    # =========================
    N_face = N_bc // 3
    X_bc = []

    for face in range(3):
        if face == 0:
            x = np.full((N_face, 1), -1.0)
            y = np.random.uniform(-1, 1, (N_face, 1))
            z = np.random.uniform(-1, 1, (N_face, 1))
        elif face == 1:
            x = np.full((N_face, 1), 1.0)
            y = np.random.uniform(-1, 1, (N_face, 1))
            z = np.random.uniform(-1, 1, (N_face, 1))
        else:
            x = np.random.uniform(-1, 1, (N_face, 1))
            y = np.random.uniform(-1, 1, (N_face, 1))
            z = np.full((N_face, 1), 1.0)

        omega = np.random.uniform(0.5 * np.pi, 2 * np.pi, (N_face, 1))
        X_bc.append(np.concatenate([x, y, z, omega], axis=1))

    # =========================
    # 合并
    # =========================
    X = np.concatenate([X_int, np.concatenate(X_bc)], axis=0)

    # =========================
    # 保存到 data 目录（关键）
    # =========================
    if save:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, X)
        print(f"[Data] Saved training data to {save_path}")

    # =========================
    # 转 torch
    # =========================
    X = torch.tensor(X, dtype=torch.float32, device=device)
    X.requires_grad_(True)

    return X
