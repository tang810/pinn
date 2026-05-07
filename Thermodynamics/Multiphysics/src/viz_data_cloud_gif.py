import os
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import imageio.v2 as imageio


def plot_frame(xyz, T, title, out_png, vmin, vmax,
               s=18, alpha=0.5, elev=18, azim=-55, cmap="viridis"):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    fig = plt.figure(figsize=(11, 7), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)

    sc = ax.scatter(x, y, z, c=T, s=s, alpha=alpha,
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    depthshade=False, linewidths=0)

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title)

    # 固定坐标范围，避免每帧缩放导致“变换逻辑不对”
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))
    ax.set_zlim(np.min(z), np.max(z))

    cb = fig.colorbar(sc, ax=ax, shrink=0.72, pad=0.08)
    cb.set_label("Temperature (K)")
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_useOffset(False)
    fmt.set_powerlimits((-3, 3))
    cb.formatter = fmt
    cb.update_ticks()

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, required=True, help="x y z t T_true")
    ap.add_argument("--out_dir", type=str, default="results/viz_gt")
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--max_points", type=int, default=80000, help="downsample per frame (0=no limit)")
    ap.add_argument("--cleanup_frames", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    data = np.loadtxt(args.data_path)
    xyz = data[:, 0:3]
    t = data[:, 3]
    T = data[:, 4]

    t_unique = np.unique(t)
    Tmin, Tmax = float(np.min(T)), float(np.max(T))
    print(f"[INFO] points={xyz.shape[0]}, frames={len(t_unique)}, T_range=[{Tmin:.3f},{Tmax:.3f}]")

    frames = []
    paths = []

    for k, tk in enumerate(t_unique):
        mask = (t == tk)
        xyz_k = xyz[mask]
        T_k = T[mask]

        # 每帧可选降采样，避免太慢/太大
        if args.max_points and xyz_k.shape[0] > args.max_points:
            idx = np.random.choice(xyz_k.shape[0], args.max_points, replace=False)
            xyz_k = xyz_k[idx]
            T_k = T_k[idx]

        png = os.path.join(args.out_dir, f"gt_frame_{k:03d}.png")
        plot_frame(
            xyz_k, T_k,
            title=f"Ground Truth Temperature Point Cloud, t={tk:.4f}",
            out_png=png,
            vmin=Tmin, vmax=Tmax,
            s=18, alpha=0.5, elev=18, azim=-55
        )

        frames.append(imageio.imread(png))
        paths.append(png)

    out_gif = os.path.join(args.out_dir, "gt_pointcloud.gif")
    imageio.mimsave(out_gif, frames, fps=args.fps)
    print("[OK] Saved:", out_gif)

    if args.cleanup_frames:
        for p in paths:
            try:
                os.remove(p)
            except OSError:
                pass
        print("[OK] cleaned frames")


if __name__ == "__main__":
    main()