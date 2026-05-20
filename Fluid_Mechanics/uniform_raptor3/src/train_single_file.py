"""
Notebook-friendly single-file train/simulation for uniform_raptor3.

This merged version does not import local src modules. It runs a lightweight
Raptor 3 quasi-1D nozzle and analytical plume simulation, then saves a .pt
checkpoint for eval_single_file.py.
"""

import os
import time
import random

import numpy as np
import torch
import matplotlib.pyplot as plt


DATA_HASH = ""
MODEL_HASH = "uniform_raptor3"
MODEL_PATH = os.path.expanduser(f"~/minio/Resource/{MODEL_HASH}/trained_model_uniform_raptor3.pt")

SEED = 42
NX_INT = 80
NX_EXT = 80
NY = 40


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


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Config:
    P_CHAMBER = 35e6
    T_CHAMBER = 3550.0
    P_AMBIENT = 101325.0
    T_AMBIENT = 288.0
    M_CH4 = 170.0
    M_LOX = 630.0
    M_TOTAL = 800.0
    OF_RATIO = 630.0 / 170.0
    GAMMA = 1.18
    R = 380.0
    N_SPECIES = 5
    SPECIES_NAMES = ["CH4", "O2", "CO2", "H2O", "N2"]
    L_PLUME = 8.0
    R_PLUME = 3.0

    @property
    def Y_CH4_IN(self):
        return self.M_CH4 / self.M_TOTAL

    @property
    def Y_O2_IN(self):
        return self.M_LOX / self.M_TOTAL


BASE_WALL = np.array([
    [0.00, 0.650], [0.40, 0.650], [0.75, 0.620], [0.92, 0.480],
    [1.00, 0.380], [1.12, 0.300], [1.28, 0.375], [1.60, 0.600],
    [2.05, 0.940], [2.70, 1.340], [3.50, 1.750],
], dtype=np.float64)
L_ENGINE = float(BASE_WALL[-1, 0])


def interp_wall(x, wall_pts):
    return np.interp(x, wall_pts[:, 0], wall_pts[:, 1])


def init_wall(cfg):
    a_star_target = (
        cfg.M_TOTAL
        / (
            cfg.P_CHAMBER
            * np.sqrt(cfg.GAMMA / (cfg.R * cfg.T_CHAMBER))
            * (2.0 / (cfg.GAMMA + 1.0)) ** ((cfg.GAMMA + 1.0) / (2.0 * (cfg.GAMMA - 1.0)))
        )
    )
    r_throat_base = np.min(BASE_WALL[:, 1])
    scale = np.sqrt(a_star_target / (np.pi * r_throat_base**2))
    wall = BASE_WALL.copy()
    wall[:, 1] *= scale
    return wall, scale


def area_mach_relation(mach, gamma):
    mach = max(float(mach), 1e-8)
    t1 = 2.0 / (gamma + 1.0)
    t2 = 1.0 + (gamma - 1.0) * mach**2 / 2.0
    return (1.0 / mach) * (t1 * t2) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))


def solve_mach_for_area(area_ratio, gamma, supersonic):
    lo, hi = (1.0001, 8.0) if supersonic else (1e-4, 0.9999)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        val = area_mach_relation(mid, gamma)
        if supersonic:
            if val < area_ratio:
                lo = mid
            else:
                hi = mid
        else:
            if val > area_ratio:
                lo = mid
            else:
                hi = mid
    return 0.5 * (lo + hi)


class Quasi1DNozzle:
    def __init__(self, cfg, wall, nx=80):
        self.cfg = cfg
        self.xc = np.linspace(0.01, L_ENGINE, nx)
        self.r_wall = interp_wall(self.xc, wall)
        self.A = np.pi * self.r_wall**2
        self.i_throat = int(np.argmin(self.A))
        self.x_throat = float(self.xc[self.i_throat])
        self.A_star = float(self.A[self.i_throat])
        self.A_ratio = np.maximum(self.A / self.A_star, 1.0)

    def solve(self):
        cfg = self.cfg
        self.M = np.zeros_like(self.xc)
        for i, ar in enumerate(self.A_ratio):
            if abs(ar - 1.0) < 1e-6:
                self.M[i] = 1.0
            else:
                self.M[i] = solve_mach_for_area(ar, cfg.GAMMA, i > self.i_throat)
        t_ratio = 1.0 / (1.0 + (cfg.GAMMA - 1.0) * self.M**2 / 2.0)
        self.T = cfg.T_CHAMBER * t_ratio
        self.P = cfg.P_CHAMBER * t_ratio ** (cfg.GAMMA / (cfg.GAMMA - 1.0))
        self.rho = self.P / (cfg.R * self.T)
        self.u = self.M * np.sqrt(cfg.GAMMA * cfg.R * self.T)
        self.Y = self.compute_species()
        return self

    def compute_species(self):
        cfg = self.cfg
        y = np.zeros((len(self.xc), cfg.N_SPECIES), dtype=np.float32)
        x_comb = self.xc[max(1, int(self.i_throat * 0.7))]
        y_ch4_react = cfg.Y_CH4_IN * (cfg.Y_O2_IN / 4.0)
        for i, x in enumerate(self.xc):
            frac = min(1.0, max(0.0, (x / max(x_comb, 1e-6)) ** 2))
            y[i, 0] = cfg.Y_CH4_IN - frac * y_ch4_react
            y[i, 1] = cfg.Y_O2_IN - frac * cfg.Y_O2_IN
            y[i, 2] = frac * y_ch4_react * 44.0 / 16.0
            y[i, 3] = frac * y_ch4_react * 36.0 / 16.0
            y[i, 4] = max(0.0, 1.0 - float(np.sum(y[i, :4])))
            y[i] /= max(float(np.sum(y[i])), 1e-8)
        return y

    def thrust(self):
        cfg = self.cfg
        mdot = self.rho[-1] * self.u[-1] * self.A[-1]
        f_mom = mdot * self.u[-1]
        f_pres = (self.P[-1] - cfg.P_AMBIENT) * self.A[-1]
        return f_mom + f_pres, f_mom, f_pres, mdot

    def isp(self):
        f_total, _, _, mdot = self.thrust()
        return f_total / (mdot * 9.81)


class PlumeModel:
    def __init__(self, cfg, nozzle, nx=80, ny=40):
        self.cfg = cfg
        self.nozzle = nozzle
        self.x = np.linspace(L_ENGINE, L_ENGINE + cfg.L_PLUME, nx)
        self.r = np.linspace(0.0, cfg.R_PLUME, ny)
        self.X, self.RR = np.meshgrid(self.x, self.r, indexing="ij")
        self.compute()

    def compute(self):
        cfg = self.cfg
        n = self.nozzle
        r_exit = n.r_wall[-1]
        d_exit = 2.0 * r_exit
        p_ratio = max(n.P[-1] / cfg.P_AMBIENT, 0.01)
        self.diamond_spacing = min(0.8 * d_exit * np.sqrt(max(n.M[-1] ** 2 - 1.0, 0.01)) * p_ratio ** (1.0 / (2 * cfg.GAMMA)), cfg.L_PLUME / 3)
        x_rel = self.X - L_ENGINE
        spread = r_exit + 0.10 * np.maximum(x_rel - 2.0 * d_exit, 0.0)
        wave = np.sin(2.0 * np.pi * x_rel / max(self.diamond_spacing, 1e-3))
        envelope = np.exp(-x_rel / max(cfg.L_PLUME * 0.45, 1e-6))
        core = np.exp(-(self.RR / np.maximum(spread, 1e-6)) ** 4)
        amp = 0.22 * wave * envelope
        self.T = cfg.T_AMBIENT + (n.T[-1] * (1.0 + amp) - cfg.T_AMBIENT) * core
        self.P = cfg.P_AMBIENT + (n.P[-1] * (1.0 + 1.4 * amp) - cfg.P_AMBIENT) * core
        self.Mach = n.M[-1] * (1.0 - 0.6 * amp) * core
        self.u = n.u[-1] * core


def show_train_result(nozzle, plume):
    f_total, f_mom, f_pres, mdot = nozzle.thrust()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].plot(nozzle.xc, nozzle.M)
    axes[0, 0].axvline(nozzle.x_throat, color="r", linestyle="--", alpha=0.5)
    axes[0, 0].set_title("Nozzle Mach")
    axes[0, 1].plot(nozzle.xc, nozzle.T, label="T")
    axes[0, 1].set_title("Temperature")
    axes[1, 0].imshow(plume.Mach.T, origin="lower", aspect="auto", extent=[plume.x.min(), plume.x.max(), plume.r.min(), plume.r.max()])
    axes[1, 0].set_title("Plume Mach")
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.02,
        0.95,
        f"Thrust: {f_total/1e6:.3f} MN\nIsp: {nozzle.isp():.1f} s\nmdot: {mdot:.1f} kg/s\nExit M: {nozzle.M[-1]:.2f}\nDiamond: {plume.diamond_spacing:.2f} m",
        va="top",
        family="monospace",
    )
    plt.tight_layout()
    plt.show()


def train(model_path=MODEL_PATH):
    set_seed(SEED)
    device = get_device()
    print(f"device={device}")
    t0 = time.time()
    cfg = Config()
    wall, scale = init_wall(cfg)
    nozzle = Quasi1DNozzle(cfg, wall, NX_INT).solve()
    plume = PlumeModel(cfg, nozzle, NX_EXT, NY)
    f_total, f_mom, f_pres, mdot = nozzle.thrust()

    checkpoint = {
        "project": "uniform_raptor3",
        "wall": wall.astype(np.float32),
        "scale": float(scale),
        "nozzle": {
            "x": nozzle.xc.astype(np.float32),
            "r_wall": nozzle.r_wall.astype(np.float32),
            "M": nozzle.M.astype(np.float32),
            "T": nozzle.T.astype(np.float32),
            "P": nozzle.P.astype(np.float32),
            "rho": nozzle.rho.astype(np.float32),
            "u": nozzle.u.astype(np.float32),
            "Y": nozzle.Y.astype(np.float32),
        },
        "plume": {
            "x": plume.x.astype(np.float32),
            "r": plume.r.astype(np.float32),
            "Mach": plume.Mach.astype(np.float32),
            "T": plume.T.astype(np.float32),
            "P": plume.P.astype(np.float32),
            "u": plume.u.astype(np.float32),
            "diamond_spacing": float(plume.diamond_spacing),
        },
        "metrics": {
            "thrust_N": float(f_total),
            "momentum_thrust_N": float(f_mom),
            "pressure_thrust_N": float(f_pres),
            "mdot_kg_s": float(mdot),
            "isp_s": float(nozzle.isp()),
            "exit_mach": float(nozzle.M[-1]),
            "exit_pressure_Pa": float(nozzle.P[-1]),
            "exit_temperature_K": float(nozzle.T[-1]),
        },
    }

    model_path = os.path.expanduser(model_path)
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    torch.save(checkpoint, model_path)
    print(f"Simulation finished in {time.time() - t0:.2f}s")
    print(f"Thrust={f_total/1e6:.3f} MN, Isp={nozzle.isp():.1f} s, Exit Mach={nozzle.M[-1]:.2f}")
    print(f"Model saved to: {model_path}")
    show_train_result(nozzle, plume)
    return checkpoint


train_result = train()
