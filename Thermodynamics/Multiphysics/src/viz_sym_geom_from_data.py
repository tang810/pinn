import os
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

import imageio.v2 as imageio
import torch

from .config import define_arguments
from .data_loader import get_training_data
from .model import PINNsformer


def read_data_file(data_path):
    """
    data file format: x y z t T
    whitespace separated
    """
    data = np.loadtxt(data_path)
    x, y, z, t, T = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]
    xyz = np.stack([x, y, z], axis=1)
    return xyz, t, T


def build_knn(xyz_src):
    """
    优先用 scipy 的 cKDTree（快）；若没有 scipy 则退化到 sklearn
    """
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(xyz_src)
        return ("scipy", tree)
    except Exception:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(xyz_src)
        return ("sklearn", nn)


def query_knn(knn_obj, xyz_query):
    kind, obj = knn_obj
    if kind == "scipy":
        dist, idx = obj.query(xyz_query, k=1)
        return idx
    else:
        dist, idx = obj.kneighbors(xyz_query, n_neighbors=1, return_distance=True)
        return idx.reshape(-1)


def plot_scalar_3d(xyz, scalar, title, cbar_label, out_png,
                   s=1.2, alpha=0.10, elev=18, azim=-55,
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
        depthshade=True, linewidths=0
    )

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title)

    # 固定坐标范围，避免每帧缩放导致“变换逻辑不对”的观感
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))
    ax.set_zlim(np.min(z), np.max(z))

    cb = fig.colorbar(sc, ax=ax, shrink=0.72, pad=0.08)
    cb.set_label(cbar_label)

    fmt = ScalarFormatter(useMathText=True)
    fmt.set_useOffset(False)
    fmt.set_powerlimits((-3, 3))
    cb.formatter = fmt
    cb.update_ticks()

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)


#对比
def compute_thermal_stress_scalar(T, E, nu, alpha, T0):
    """
    与 loss 中一致的热应力标量形式（简化展示）：
    sigma_th = E * alpha * (T - T0) / (1 - 2*nu)
    这里取 abs 作为标量云图
    """
    sigma = E * alpha * (T - T0) / (1.0 - 2.0 * nu)
    return np.abs(sigma)


# def plot_gt_vs_pred_3d(xyz, T_gt, T_pred, tk, out_png,
#                        s=18, alpha=0.28, elev=18, azim=-55,
#                        cmap="viridis", vmin=None, vmax=None, dpi=160):
def fig_to_rgb_array(fig):
    """把 matplotlib Figure 转成 (H,W,3) uint8，用于直接写 gif。"""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return buf.reshape(h, w, 3)


def plot_gt_vs_pred_3d(
    xyz, T_gt, T_pred, tk, out_png,
    s=18, alpha=0.28, elev=18, azim=-55,
    cmap="viridis",
    vmin_gt=None, vmax_gt=None,
    vmin_pd=None, vmax_pd=None, 
    dpi=160,
):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    fig = plt.figure(figsize=(14, 7), dpi=dpi) # Increased from 13, 6


    ax1 = fig.add_subplot(121, projection="3d")
    ax1.view_init(elev=elev, azim=azim)
    sc1 = ax1.scatter(
        x, y, z, c=T_gt, s=s, alpha=alpha, cmap=cmap,
        vmin=vmin_gt, vmax=vmax_gt, depthshade=False, linewidths=0
    )
    ax1.set_title(f"Ground Truth, t={tk:.2f}")
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.view_init(elev=elev, azim=azim)
    sc2 = ax2.scatter(
        x, y, z, c=T_pred, s=s, alpha=alpha, cmap=cmap,
        vmin=vmin_pd, vmax=vmax_pd, depthshade=False, linewidths=0
    )
    ax2.set_title(f"PINNs Prediction, t={tk:.2f}")
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")

    for ax in (ax1, ax2):
        ax.set_xlim(np.min(x), np.max(x))
        ax.set_ylim(np.min(y), np.max(y))
        ax.set_zlim(np.min(z), np.max(z))

    cb1 = fig.colorbar(sc1, ax=ax1, shrink=0.72, pad=0.08)
    cb1.set_label("Temperature (K)")
    cb2 = fig.colorbar(sc2, ax=ax2, shrink=0.72, pad=0.08)
    cb2.set_label("Temperature (K)")

    plt.subplots_adjust(left=0.05, right=0.95, wspace=0.3)
    # plt.tight_layout() removed to prevent overlap
    plt.savefig(out_png)
    plt.close(fig)

# def build_time_sequence_for_points(xyz, tk, time_steps, step_size, device):
#     """
#     构造 (N, time_steps, 4) 的输入序列，让模型预测多个 step。
#     但可视化一般取 step=0 对应 tk。
#     """
#     # 在 build_time_sequence_for_points 内部，构造 base 之前加：
#     xmin, ymin, zmin = -1.0, -0.5, -0.3
#     xmax, ymax, zmax =  1.0,  0.5,  0.3
#     xyz01 = xyz.copy()
#     xyz01[:,0] = (xyz01[:,0]-xmin)/(xmax-xmin)
#     xyz01[:,1] = (xyz01[:,1]-ymin)/(ymax-ymin)
#     xyz01[:,2] = (xyz01[:,2]-zmin)/(zmax-zmin)
# # 然后用 xyz01 替代 xyz
#     N = xyz.shape[0]
#     base = np.hstack([xyz, np.full((N, 1), tk, dtype=np.float32)])  # (N,4)
#     seq = np.repeat(base[:, None, :], time_steps, axis=1)          # (N,T,4)
#     for i in range(time_steps):
#         seq[:, i, 3] = seq[:, i, 3] + i * step_size
#     return torch.tensor(seq, dtype=torch.float32, requires_grad=False).to(device)
def make_gif(frames, out_gif, fps=6, loop=0):
    if frames is None or len(frames) == 0:
        print(f"[WARN] No frames collected, skip gif: {out_gif}")
        return
    # loop=0 表示无限循环；loop=1 表示播放一次
    imageio.mimsave(out_gif, frames, fps=fps, loop=loop)

# def make_gif(frames, out_gif, fps=6):
#     # [新增] 空帧保护：避免 frames=[] 时 imageio 崩溃
#     if frames is None or len(frames) == 0:
#         print(f"[WARN] No frames collected, skip gif: {out_gif}")
#         return
#     imageio.mimsave(out_gif, frames, fps=fps)

def fig_to_rgb_array(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return buf.reshape(h, w, 3)
def build_time_sequence_for_points(
    xyz,
    tk,
    time_steps,
    step_size,
    device,
    input_norm="none",
    box_p1=None,
    box_p2=None,
    t_norm="none",
    tmin=None,
    tmax=None,
):
    """
    构造 (N, time_steps, 4) 输入序列:
      [x, y, z, t]
    并支持可选归一化：
      - input_norm="box01": 将 xyz 从 [box_p1, box_p2] 线性映射到 [0,1]
      - t_norm="minmax01":  将 tk 从 [tmin, tmax] 线性映射到 [0,1]
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    N = xyz.shape[0]

    # ---- xyz normalization ----
    xyz_use = xyz
    if input_norm == "box01":
        if box_p1 is None or box_p2 is None:
            raise ValueError("input_norm=box01 requires box_p1 and box_p2")
        p1 = np.array(box_p1, dtype=np.float32).reshape(1, 3)
        p2 = np.array(box_p2, dtype=np.float32).reshape(1, 3)
        denom = (p2 - p1)
        denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)
        xyz_use = (xyz - p1) / denom
        xyz_use = np.clip(xyz_use, 0.0, 1.0)

    # ---- time normalization ----
    tk_use = float(tk)
    if t_norm == "minmax01":
        if tmin is None or tmax is None:
            raise ValueError("t_norm=minmax01 requires tmin and tmax")
        denom = (float(tmax) - float(tmin))
        if abs(denom) < 1e-12:
            tk_use = 0.0
        else:
            tk_use = (tk_use - float(tmin)) / denom
        # clip to [0,1] to avoid out-of-range
        tk_use = float(np.clip(tk_use, 0.0, 1.0))

    # ---- base + sequence ----
    base = np.hstack([xyz_use, np.full((N, 1), tk_use, dtype=np.float32)])  # (N,4)
    seq = np.repeat(base[:, None, :], int(time_steps), axis=1)             # (N,T,4)
    for i in range(int(time_steps)):
        seq[:, i, 3] = seq[:, i, 3] + i * float(step_size)

    return torch.tensor(seq, dtype=torch.float32, requires_grad=False).to(device)


def run_viz(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # device 自动降级：cuda -> cpu
    _dev = getattr(args, "device", "cpu")
    if "cuda" in _dev and not torch.cuda.is_available():
        print(f"[WARN] CUDA requested but not available in this environment; falling back to CPU.")
        _dev = "cpu"
    device = torch.device(_dev)
    print("Using device:", device)

    # 默认使用 Sym + CSG，但若命令行已给出则尊重命令行
    if not hasattr(args, "geometry") or args.geometry is None:
        args.geometry = "csg"
    if not hasattr(args, "geometry_backend") or args.geometry_backend is None:
        args.geometry_backend = "sym"

    # 打印确认形状与参数
    print(f"[INFO] geometry={args.geometry}, backend={args.geometry_backend}, sym_shape={getattr(args,'sym_shape',None)}")
    if getattr(args, "sym_shape", None) == "box":
        print(f"[INFO] box_p1={getattr(args,'box_p1',None)}, box_p2={getattr(args,'box_p2',None)}")

    # 2) 先读 GT 数据文件（无论是否使用 sym，都需要 GT 数据）
    xyz_data, t_data, T_data = read_data_file(args.data_path_gt)
    t_unique = np.unique(t_data)
    print(f"[INFO] gt data points={xyz_data.shape[0]}, unique_t={len(t_unique)}")

    # 全局色条范围（GT）
    Tmin, Tmax = float(np.min(T_data)), float(np.max(T_data))
    print(f"[INFO] GT Temperature range: [{Tmin:.3f}, {Tmax:.3f}]")

    # 1) 从 Sym 几何采样得到点云; 若 sym 不可用则 fallback 到 GT 数据点
    xyz_sym = None
    try:
        # 先检测 physicsnemo.sym 是否可用（此行若报错则直接跳到 fallback）
        from physicsnemo.sym.geometry.primitives_3d import Box as _Box   # noqa: F401
        del _Box
        res_seq, bc_seq, bc_normals, bc_tag, ic_seq, ic_T = get_training_data(args, device)
        if args.plot_on == "bc":
            xyz_sym = bc_seq[:, 0, 0:3].detach().cpu().numpy()
            print(f"[INFO] sym bc points = {xyz_sym.shape[0]}")
        else:
            xyz_sym = res_seq[:, 0, 0:3].detach().cpu().numpy()
            print(f"[INFO] sym res points = {xyz_sym.shape[0]}")
    except Exception as e:
        print(f"[WARN] Sym geometry backend not available ({e}); using GT data points as point cloud.")
        # Fallback: 用第一帧的空间点作为"点云"（各帧空间分布相同）
        first_t = t_unique[0]
        mask0 = np.isclose(t_data, first_t, atol=float(getattr(args, 't_tol', 1e-8)), rtol=0.0)
        xyz_sym = xyz_data[mask0].astype(np.float32)
        if xyz_sym.shape[0] < 10:
            xyz_sym = xyz_data.astype(np.float32)
        print(f"[INFO] fallback point cloud: {xyz_sym.shape[0]} points")

    # 2.5) 加载模型（用于 Prediction）
    # pinnsformer_withsun.pt 的真实结构：d_model=64, d_hidden=512
    _d_model = 64
    _d_hidden = 512
    model = PINNsformer(d_model=_d_model, d_hidden=_d_hidden, N=args.N, heads=args.heads).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 帧缓存
    frames_gtpred = []
    frames_stress = []
    frame_paths = []

    # stress 色条范围（固定为首帧范围，避免每帧跳动）
    stress_vmin, stress_vmax = None, None

    # 兼容两种 plot_gt_vs_pred_3d 版本：
    #  - 旧版：plot_gt_vs_pred_3d(..., vmin=..., vmax=...)
    #  - 新版：plot_gt_vs_pred_3d(..., vmin_gt=..., vmax_gt=..., vmin_pd=..., vmax_pd=...)
    import inspect
    sig = inspect.signature(plot_gt_vs_pred_3d)
    has_separate = ("vmin_gt" in sig.parameters) and ("vmin_pd" in sig.parameters)

    for k, tk in enumerate(t_unique):
        # 用 isclose 修复浮点精度导致的“匹配不到该帧”
        mask = np.isclose(t_data, tk, atol=args.t_tol, rtol=0.0)
        xyz_src = xyz_data[mask]
        T_src = T_data[mask]

        if xyz_src.shape[0] < 10:
            print(f"[WARN] skip t={tk:.6g}: matched_points={xyz_src.shape[0]}")
            continue

        # --- GT: KNN 映射到 Sym 点云 ---
        knn = build_knn(xyz_src)
        idx = query_knn(knn, xyz_sym)
        T_gt = T_src[idx].astype(np.float32)  # (Nsym,)

        # --- Pred: 同一批 xyz_sym + tk 输入模型 ---
        # pts_seq = build_time_sequence_for_points(
        #     xyz_sym.astype(np.float32),
        #     float(tk),
        #     time_steps=int(args.time_steps),
        #     step_size=float(args.step_size),
        #     device=device,
        # )
        # box 参数（用于 input_norm=box01）
        box_p1 = getattr(args, "box_p1", None)
        box_p2 = getattr(args, "box_p2", None)

        # time min/max（用于 t_norm=minmax01）
        tmin_use = args.tmin if args.tmin is not None else float(np.min(t_data))
        tmax_use = args.tmax if args.tmax is not None else float(np.max(t_data))

        pts_seq = build_time_sequence_for_points(
            xyz_sym.astype(np.float32),
            float(tk),
            time_steps=int(args.time_steps),
            step_size=float(args.step_size),
            device=device,
            input_norm=args.input_norm,
            box_p1=box_p1,
            box_p2=box_p2,
            t_norm=args.t_norm,
            tmin=tmin_use,
            tmax=tmax_use,
)
        with torch.no_grad():
            T_pred_seq = model(pts_seq)[:, :, 0].detach().cpu().numpy()  # (Nsym, time_steps)
        T_pred = T_pred_seq[:, 0]  # (Nsym,) numpy (step=0 对应 tk)

        # ===== Pred scale diagnostics =====
        T_pred_np_raw = T_pred.reshape(-1)  # ✅T_pred 已经是 numpy
        print("[DIAG] T_pred(raw) stats:",
              f"min={T_pred_np_raw.min():.6g}, max={T_pred_np_raw.max():.6g},",
              f"mean={T_pred_np_raw.mean():.6g}, std={T_pred_np_raw.std():.6g}")
        # =================================

        # ===== optional inverse-normalization =====
        if args.pred_denorm:
            T_pred_np = T_pred_np_raw * (Tmax - Tmin) + Tmin
            print("[DIAG] pred_denorm enabled:",
                  f"min={T_pred_np.min():.3f}, max={T_pred_np.max():.3f}")
        else:
            T_pred_np = T_pred_np_raw
        # =========================================
                # --- 取出 raw 预测（未标定） ---
        #T_pred_raw = T_pred_np_raw.astype(np.float32)
        T_pred_raw = T_pred_np.astype(np.float32)
        T_gt = T_gt.reshape(-1).astype(np.float32)

        # ====== [NEW] 预测标定：T_cal = a*T_pred + b ======
        a = 1.0
        b = 0.0

        if args.pred_calib in ("global", "per_frame"):
            # 子采样，避免 120k 点拟合太慢
            n = T_gt.shape[0]
            m = min(int(args.calib_points), n)
            idx_cal = np.random.choice(n, m, replace=False)

            x = T_pred_raw[idx_cal]
            y = T_gt[idx_cal]

            # 最小二乘解 a,b
            # a = cov(x,y)/var(x), b = mean(y)-a*mean(x)
            vx = float(np.var(x)) + 1e-12
            a = float(np.cov(x, y, bias=True)[0, 1] / vx)
            b = float(np.mean(y) - a * np.mean(x))

            print(f"[CALIB] t={tk:.4f} a={a:.6g}, b={b:.6g}, "
                f"pred_raw_range=[{T_pred_raw.min():.6g},{T_pred_raw.max():.6g}] -> "
                f"pred_cal_range=[{(a*T_pred_raw+b).min():.3f},{(a*T_pred_raw+b).max():.3f}]")

        T_pred_cal = a * T_pred_raw + b
        # ================================================

        # 用标定后的 Pred 去画图/算 stress
        T_pred_use = T_pred_cal##################################################################################################
        

        # 颜色范围策略：全局共享范围，使 GIF 每帧的色条数值保持完全静止不变
        if args.cbar_mode == "shared":
            vmin_gt = vmin_pd = float(Tmin)
            vmax_gt = vmax_pd = float(Tmax)
        else:
            vmin_gt, vmax_gt = float(Tmin), float(Tmax)
            vmin_pd, vmax_pd = float(T_pred_use.min()), float(T_pred_use.max())
            vmin_pd, vmax_pd = float(T_pred_use.min()), float(T_pred_use.max())

        # --- 画“GT vs Pred”双子图 ---
        png_gtpred = os.path.join(args.out_dir, f"gt_vs_pred_{args.plot_on}_t_{k:03d}.png")

        if has_separate:
            # 新版：允许 GT 和 Pred 各自色条范围
        
            plot_gt_vs_pred_3d(
                xyz=xyz_sym,
                T_gt=T_gt,
                T_pred=T_pred_use,
                tk=float(tk),
                out_png=png_gtpred,
                s=float(args.point_size),
                alpha=float(args.point_alpha),
                elev=18,
                azim=-55,
                cmap="viridis",
                vmin_gt=vmin_gt,
                vmax_gt=vmax_gt,
                vmin_pd=vmin_pd,
                vmax_pd=vmax_pd,
                dpi=int(args.dpi),
            )
        else:
            # 旧版：只能共用一个 vmin/vmax（此时 separate 等价于 shared）
            plot_gt_vs_pred_3d(
                xyz_sym, T_gt, T_pred_use, float(tk), png_gtpred,
                #xyz_sym, T_gt, T_pred_np, float(tk), png_gtpred,#######################################################################
                s=float(args.point_size), alpha=float(args.point_alpha),
                elev=18, azim=-55, cmap="viridis",
                vmin=vmin_gt, vmax=vmax_gt,
                dpi=int(args.dpi),
            )

        frames_gtpred.append(imageio.imread(png_gtpred))
        frame_paths.append(png_gtpred)

        # --- Stress 图（按 Pred 温度推应力）---
        if args.make_stress:
            if not args.pred_denorm and args.pred_calib == "none":
                print("[WARN] stress uses raw pred scale; enable --pred_calib or correct denorm.")

            # 用最终用于画温度的 Pred（若开 denorm 则是 Kelvin）
            stress = compute_thermal_stress_scalar(
                T_pred_use,
                E=float(args.E), nu=float(args.nu), alpha=float(args.alpha), T0=float(args.T0)
            ).astype(np.float32)

            if stress_vmin is None:
                stress_vmin = 0.0
                stress_vmax = float(compute_thermal_stress_scalar(
                    np.array([float(Tmax)]), 
                    E=float(args.E), nu=float(args.nu), alpha=float(args.alpha), T0=float(args.T0)
                )[0])

            png_stress = os.path.join(args.out_dir, f"stress_{args.plot_on}_t_{k:03d}.png")
            plot_scalar_3d(
                xyz_sym, stress,
                title=f"Thermal Stress, t={float(tk):.2f}",
                cbar_label="von Mises Stress (Pa)",
                out_png=png_stress,
                s=float(args.point_size),
                alpha=float(args.point_alpha),
                cmap="inferno",
                vmin=stress_vmin, vmax=stress_vmax,
            )
            frames_stress.append(imageio.imread(png_stress))
            frame_paths.append(png_stress)

    # 输出 gif
    out_gif_gtpred = os.path.join(args.out_dir, f"gt_vs_pred_{args.plot_on}.gif")
    make_gif(frames_gtpred, out_gif_gtpred, fps=args.fps)
    print("[OK] Saved GIF:", out_gif_gtpred)

    if args.make_stress:
        out_gif_stress = os.path.join(args.out_dir, f"stress_{args.plot_on}.gif")
        make_gif(frames_stress, out_gif_stress, fps=args.fps)
        print("[OK] Saved GIF:", out_gif_stress)

    # 清理帧
    if args.cleanup_frames:
        for fp in frame_paths:
            try:
                os.remove(fp)
            except OSError:
                pass
        print("[OK] cleaned frames")
if __name__ == "__main__":
    import sys
    try:
        from .config import define_arguments
    except ImportError:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.config import define_arguments
        
    p = argparse.ArgumentParser()
    define_arguments(p)
    args = p.parse_args()
    run_viz(args)