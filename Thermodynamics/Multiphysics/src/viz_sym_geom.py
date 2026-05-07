import os
import argparse
import numpy as np
import imageio.v2 as imageio
from matplotlib.ticker import ScalarFormatter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

from .config import define_arguments
from .data_loader import get_training_data
from .model import PINNsformer


def compute_thermal_stress_von_mises(T, E, nu, alpha, T0):
    """
    使用你训练损失里一致的热应力标量形式：
    sigma_th = E * alpha * (T - T0) / (1 - 2*nu)
    vonMises（简化）这里取 |sigma_th| 作为标量展示
    """
    sigma_th = E * alpha * (T - T0) / (1.0 - 2.0 * nu)
    return torch.abs(sigma_th)


def plot_scalar_3d(xyz, scalar, title, cbar_label, out_png,
                   s=120, alpha=0.55, elev=18, azim=-55,
                   cmap="viridis", vmin=None, vmax=None):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    v = scalar.reshape(-1)

    fig = plt.figure(figsize=(11, 7), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)

    sc = ax.scatter(
        x, y, z,
        c=v, s=s, alpha=alpha,
        cmap=cmap, vmin=vmin, vmax=vmax,
        depthshade=False, linewidths=0
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    # 轴范围固定，画出来更像“立体实体”
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))
    ax.set_zlim(np.min(z), np.max(z))

    cb = fig.colorbar(sc, ax=ax, shrink=0.72, pad=0.08)
    cb.set_label(cbar_label)

    # 关闭 colorbar 的 offset（+1e9 那种）
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_useOffset(False)
    fmt.set_powerlimits((-3, 3))
    cb.formatter = fmt
    cb.update_ticks()

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)

def save_gif_from_sequence(xyz0, scalar_seq, title_prefix, cbar_label, out_gif,
                           cmap="viridis", s=120, alpha=0.55, elev=18, azim=-55,
                           vmin=None, vmax=None, fps=6):
    """
    xyz0: (N,3) numpy
    scalar_seq: (N,T) numpy
    """
    frames = []
    T = scalar_seq.shape[1]

    # 固定颜色范围，避免每帧色条跳动（更像目标图）
    if vmin is None:
        vmin = float(np.min(scalar_seq))
    if vmax is None:
        vmax = float(np.max(scalar_seq))

    for k in range(T):
        png_path = out_gif.replace(".gif", f"_frame_{k:03d}.png")
        plot_scalar_3d(
            xyz0, scalar_seq[:, k],
            title=f"{title_prefix}, step={k:02d}",
            cbar_label=cbar_label,
            out_png=png_path,
            s=s, alpha=alpha, elev=elev, azim=azim,
            cmap=cmap, vmin=vmin, vmax=vmax
        )
        frames.append(imageio.imread(png_path))

    imageio.mimsave(out_gif, frames, fps=fps)
def main():
    # 复用 config.py 参数
    p = argparse.ArgumentParser()
    define_arguments(p)
    p.add_argument("--ckpt", type=str, default="results/pinnsformer_model.pth", help="model checkpoint path")
    p.add_argument("--out_dir", type=str, default="results/viz_sym", help="output directory")
    p.add_argument("--plot_on", type=str, default="bc", choices=["bc", "res", "ic"],
                   help="plot which points: bc boundary / res interior / ic initial")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(getattr(args, "device", "cpu"))
    print("Using device:", device)

    # 关键：强制使用 Sym 几何后端 + CSG
    args.geometry = "csg"
    args.geometry_backend = "sym"

    # 1) 取训练点（由 Sym 几何采样产生）
    res_seq, bc_seq, bc_normals, bc_tag, ic_seq, ic_T = get_training_data(args, device)

    # 选择画哪一类点
    if args.plot_on == "bc":
        pts_seq = bc_seq
        hole_ratio = float((bc_tag == 1).float().mean().item())
        print(f"[INFO] plot_on=bc, hole_ratio={hole_ratio:.3f}, normals={tuple(bc_normals.shape)}")
    elif args.plot_on == "res":
        pts_seq = res_seq
        print("[INFO] plot_on=res")
    else:
        pts_seq = ic_seq
        print("[INFO] plot_on=ic")

    # 2) 加载模型
    model = PINNsformer(d_model=args.d_model, d_hidden=args.d_hidden, N=args.N, heads=args.heads).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 3) 点云坐标（固定用 step=0 的 xyz）
    xyz0 = pts_seq[:, 0, 0:3].detach().cpu().numpy()

    # 4) 预测整个 time_steps 序列（一次前向就够）
    with torch.no_grad():
        T_pred_seq = model(pts_seq)[:, :, 0]  # (N, time_steps)

    T_np = T_pred_seq.detach().cpu().numpy()  # (N, T)

    # 5) t=0 的 png 输出
    T0_pred_np = T_np[:, 0]  # (N,)

    # 热应力序列（简化 von Mises 标量）
    E = float(args.E)
    nu = float(args.nu)
    alpha = float(args.alpha)
    T0ref = float(args.T0)

    stress_seq = np.abs(E * alpha * (T_np - T0ref) / (1.0 - 2.0 * nu))  # (N, T)
    stress0_np = stress_seq[:, 0]

    out_T = os.path.join(args.out_dir, f"sym_{args.plot_on}_T_pred_t0.png")
    out_S = os.path.join(args.out_dir, f"sym_{args.plot_on}_stress_t0.png")

    plot_scalar_3d(
        xyz0, T0_pred_np,
        title=f"Sym Geometry: PINNs Prediction, t=0.00 ({args.plot_on})",
        cbar_label="Temperature (K)",
        out_png=out_T,
        cmap="viridis"
    )

    plot_scalar_3d(
        xyz0, stress0_np,
        title=f"Sym Geometry: Thermal Stress, t=0.00 ({args.plot_on})",
        cbar_label="von Mises Stress (Pa)",
        out_png=out_S,
        cmap="inferno"
    )

    print("[OK] Saved:", out_T)
    print("[OK] Saved:", out_S)

    # 6) GIF 输出（沿 time_steps）
    out_gif_T = os.path.join(args.out_dir, f"sym_{args.plot_on}_T_pred.gif")
    out_gif_S = os.path.join(args.out_dir, f"sym_{args.plot_on}_stress.gif")

    save_gif_from_sequence(
        xyz0, T_np,
        title_prefix=f"Sym Geometry: PINNs Prediction (t sequence, {args.plot_on})",
        cbar_label="Temperature (K)",
        out_gif=out_gif_T,
        cmap="viridis",
        s=120, alpha=0.55, elev=18, azim=-55,
        fps=6
    )

    save_gif_from_sequence(
        xyz0, stress_seq,
        title_prefix=f"Sym Geometry: Thermal Stress (t sequence, {args.plot_on})",
        cbar_label="von Mises Stress (Pa)",
        out_gif=out_gif_S,
        cmap="inferno",
        s=120, alpha=0.55, elev=18, azim=-55,
        fps=6
    )

    print("[OK] Saved GIF:", out_gif_T)
    print("[OK] Saved GIF:", out_gif_S)


if __name__ == "__main__":
    main()