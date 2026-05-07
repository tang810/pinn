# -*- coding: utf-8 -*-
"""
后验切片可视化（z = mid）
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # 后端设为非交互式
import matplotlib.pyplot as plt

def save_posterior_slices(pred_list, N_val_1d: int, lb: float, ub: float, outdir: str = "result"):
    os.makedirs(outdir, exist_ok=True)

    Nv = N_val_1d
    u_samps = pred_list[0].detach().cpu().numpy().squeeze()  # 形状可能是 (S, N^3, 1) 或 (S, N^3)
    f_samps = pred_list[1].detach().cpu().numpy().squeeze()

    if u_samps.ndim == 3:  # (S, N^3, 1) → (S, N^3)
        u_samps = u_samps[..., 0]
        f_samps = f_samps[..., 0]

    u_mean = u_samps.mean(axis=0).reshape(Nv, Nv, Nv)
    u_std  = u_samps.std(axis=0).reshape(Nv, Nv, Nv)
    f_mean = f_samps.mean(axis=0).reshape(Nv, Nv, Nv)
    f_std  = f_samps.std(axis=0).reshape(Nv, Nv, Nv)

    mid = Nv // 2

    plt.figure(figsize=(6,5))
    plt.title('u mean (z=mid)')
    plt.imshow(u_mean[:, :, mid].T, origin='lower', extent=[lb, ub, lb, ub], aspect='auto')
    plt.colorbar(); plt.xlabel('x'); plt.ylabel('y')
    plt.tight_layout(); plt.savefig(os.path.join(outdir, 'u_mean_zmid.png'), dpi=200); plt.close()

    plt.figure(figsize=(6,5))
    plt.title('u std (z=mid)')
    plt.imshow(u_std[:, :, mid].T, origin='lower', extent=[lb, ub, lb, ub], aspect='auto')
    plt.colorbar(); plt.xlabel('x'); plt.ylabel('y')
    plt.tight_layout(); plt.savefig(os.path.join(outdir, 'u_std_zmid.png'), dpi=200); plt.close()

    plt.figure(figsize=(6,5))
    plt.title('f mean (z=mid)')
    plt.imshow(f_mean[:, :, mid].T, origin='lower', extent=[lb, ub, lb, ub], aspect='auto')
    plt.colorbar(); plt.xlabel('x'); plt.ylabel('y')
    plt.tight_layout(); plt.savefig(os.path.join(outdir, 'f_mean_zmid.png'), dpi=200); plt.close()

    plt.figure(figsize=(6,5))
    plt.title('f std (z=mid)')
    plt.imshow(f_std[:, :, mid].T, origin='lower', extent=[lb, ub, lb, ub], aspect='auto')
    plt.colorbar(); plt.xlabel('x'); plt.ylabel('y')
    plt.tight_layout(); plt.savefig(os.path.join(outdir, 'f_std_zmid.png'), dpi=200); plt.close()

    print(f"[Plot] Saved posterior slice figures to '{outdir}/'")
