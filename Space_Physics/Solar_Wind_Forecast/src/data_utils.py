import numpy as np
import pandas as pd
import torch

from .constants import FEATURES


def make_synthetic_hourly(n_hours=24 * 240, seed=0, alpha_true=0.02):
    """
    合成更“可预测”的小时数据：
    - V, B 都包含：慢周期 + 日周期 + 强AR(1)
    - E(t) 依赖 cross_norm(t) 和 cross_norm(t-12)
    - 少量违反物理约束的点（可用 physics loss 拉回）
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_hours, dtype=np.float64)

    daily = np.sin(2 * np.pi * t / 24.0)
    weekly = np.sin(2 * np.pi * t / (24.0 * 7.0))
    slow = np.sin(2 * np.pi * t / (24.0 * 60.0))

    def ar1(mu, sig, phi, noise):
        base = mu + sig * rng.normal(0, 1, n_hours)
        x = np.zeros(n_hours, dtype=np.float64)
        eps = rng.normal(0, noise / 10.0, n_hours)
        for i in range(1, n_hours):
            x[i] = phi * x[i - 1] + eps[i]
        return base + x

    # 速度（km/s）
    Vx = ar1(mu=-450, sig=15, phi=0.999, noise=0.6) + 30 * daily + 20 * weekly + 10 * slow
    Vy = ar1(mu=10, sig=8, phi=0.998, noise=0.4) + 12 * daily + 6 * weekly
    Vz = ar1(mu=0, sig=7, phi=0.998, noise=0.4) + 8 * daily - 5 * weekly

    # 磁场（nT）
    Bx = ar1(mu=0.0, sig=1.5, phi=0.995, noise=0.12) + 1.2 * daily + 0.8 * weekly
    By = ar1(mu=0.0, sig=1.5, phi=0.995, noise=0.12) - 1.0 * daily + 0.6 * weekly
    Bz = ar1(mu=0.0, sig=1.5, phi=0.995, noise=0.12) + 0.7 * daily + 0.9 * slow

    V = np.stack([Vx, Vy, Vz], axis=1)
    B = np.stack([Bx, By, Bz], axis=1)
    cross_norm = np.linalg.norm(np.cross(V, B), axis=1)

    lag = 12
    cross_lag = np.roll(cross_norm, lag)
    cross_lag[:lag] = cross_norm[:lag]

    noise = rng.normal(0, 0.1 / 10.0, n_hours)
    E = 0.85 * alpha_true * cross_norm + 0.15 * alpha_true * cross_lag + noise
    E = np.clip(E, -np.inf, alpha_true * cross_norm)

    violate = rng.random(n_hours) < 0.03
    E[violate] = alpha_true * cross_norm[violate] + np.abs(rng.normal(0, 0.3, violate.sum()))

    df = pd.DataFrame(
        np.column_stack([E, V, B]),
        columns=FEATURES,
        index=pd.date_range("1992-01-01", periods=n_hours, freq="h"),
    )
    return df


def normalize_z(df: pd.DataFrame):
    mu = df.mean()
    sig = df.std(ddof=0).replace(0, 1.0)
    df_n = (df - mu) / sig
    return df_n, mu, sig


def inverse_z_torch(x_norm: torch.Tensor, mu: torch.Tensor, sig: torch.Tensor):
    """
    x_norm: (batch, D) or (batch, seq, D)
    mu,sig: (D,)
    """
    return x_norm * sig + mu


def split_by_time(df_raw: pd.DataFrame, train_ratio=0.6, val_ratio=0.3):
    """按时间顺序切分 train/val/test。"""
    T = len(df_raw)
    i1 = int(T * train_ratio)
    i2 = int(T * (train_ratio + val_ratio))
    train_raw = df_raw.iloc[:i1]
    val_raw = df_raw.iloc[i1:i2]
    test_raw = df_raw.iloc[i2:]
    return train_raw, val_raw, test_raw


def fit_alpha_raw(df_raw: pd.DataFrame, eps=1e-6):
    """
    用原始单位拟合 alpha: median(|E| / ||VxB||)
    """
    arr = df_raw[FEATURES].to_numpy(dtype=np.float64)
    E = np.abs(arr[:, 0])
    V = arr[:, 1:4]
    B = arr[:, 4:7]
    cross_norm = np.linalg.norm(np.cross(V, B), axis=1)
    ratio = E / (cross_norm + eps)
    ratio = ratio[np.isfinite(ratio)]
    return float(np.median(ratio)) if ratio.size else 1.0
