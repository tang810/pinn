import os
import numpy as np

import matplotlib
matplotlib.use("Agg")  # 强制无界面后端，确保 savefig 正常写文件
import matplotlib.pyplot as plt


# ---------- SDF primitives ----------
def sdf_sphere(p, center=(0, 0, 0), r=1.2):
    c = np.array(center)[None, :]
    return np.linalg.norm(p - c, axis=1) - r


def sdf_box(p, p1=(-1, -1, -1), p2=(1, 1, 1)):
    # axis-aligned box SDF
    p1 = np.array(p1)
    p2 = np.array(p2)
    c = (p1 + p2) / 2.0
    h = (p2 - p1) / 2.0  # half-size
    q = np.abs(p - c[None, :]) - h[None, :]
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    return outside + inside


def sdf_cylinder_axis(p, r=0.5, h=2.0, axis="z"):
    # finite cylinder SDF, centered at origin, axis in {x,y,z}
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    if axis == "z":
        d0 = np.sqrt(x * x + y * y) - r
        d1 = np.abs(z) - h / 2
    elif axis == "x":
        d0 = np.sqrt(y * y + z * z) - r
        d1 = np.abs(x) - h / 2
    elif axis == "y":
        d0 = np.sqrt(x * x + z * z) - r
        d1 = np.abs(y) - h / 2
    else:
        raise ValueError("axis must be x/y/z")

    outside = np.sqrt(np.maximum(d0, 0.0) ** 2 + np.maximum(d1, 0.0) ** 2)
    inside = np.minimum(np.maximum(d0, d1), 0.0)
    return outside + inside


# ---------- CSG ops via SDF ----------
def sdf_union(a, b):         # A ∪ B
    return np.minimum(a, b)


def sdf_intersection(a, b):  # A ∩ B
    return np.maximum(a, b)


def sdf_subtract(a, b):      # A \ B
    return np.maximum(a, -b)


def sdf_geometry_components(p):
    # box & sphere intersection: s_bs = max(s_box, s_sph)
    s_box = sdf_box(p, (-1, -1, -1), (1, 1, 1))
    s_sph = sdf_sphere(p, (0, 0, 0), 1.2)
    s_bs = sdf_intersection(s_box, s_sph)

    # cylinders union: s_c = min(s_cz, s_cx, s_cy)
    s_cz = sdf_cylinder_axis(p, r=0.5, h=2.0, axis="z")
    s_cx = sdf_cylinder_axis(p, r=0.5, h=2.0, axis="x")
    s_cy = sdf_cylinder_axis(p, r=0.5, h=2.0, axis="y")
    s_c = sdf_union(sdf_union(s_cz, s_cx), s_cy)

    # final: (box ∩ sphere) \ cylinders  => s_geom = max(s_bs, -s_c)
    s_geom = sdf_subtract(s_bs, s_c)
    return s_geom, s_bs, s_c


def sdf_geometry(p):
    s_geom, _, _ = sdf_geometry_components(p)
    return s_geom


# ---------- adaptive sampling ----------
def sample_points(bound=1.05, n_interior=60000, n_boundary=80000,
                  hole_ratio=0.60, chunk=300000, max_rounds=50):
    """
    自适应采样（孔洞增强版）
    - interior: s_geom < 0
    - boundary 分两类：
      1) outer:  |s_geom| 最小（外表面）
      2) holes:  在 s_bs<0 前提下 |s_c| 最小（圆柱孔洞内表面）
    返回:
      interior, boundary, boundary_sgeom, boundary_tag (0 outer / 1 holes)
    """
    n_hole = int(round(n_boundary * hole_ratio))
    n_outer = n_boundary - n_hole

    interior_pool = []

    outer_p, outer_s = None, None          # s 为 s_geom
    hole_p, hole_s, hole_sc = None, None, None  # hole_s 为 s_geom；hole_sc 为 s_c（用于裁剪）

    total_candidate = 0

    for r in range(max_rounds):
        p = np.random.uniform(-bound, bound, size=(chunk, 3))
        s_geom, s_bs, s_c = sdf_geometry_components(p)
        total_candidate += chunk

        # ----- interior accumulate -----
        inside = p[s_geom < 0]
        if inside.size > 0:
            interior_pool.append(inside)

        # ----- outer boundary pool (min |s_geom|) -----
        abs_g = np.abs(s_geom)
        k_out = min(chunk, max(n_outer * 3, n_outer))
        k_out = max(k_out, 1)
        idx_out = np.argpartition(abs_g, k_out - 1)[:k_out]

        cand_out_p = p[idx_out]
        cand_out_s = s_geom[idx_out]

        if outer_p is None:
            outer_p, outer_s = cand_out_p, cand_out_s
        else:
            outer_p = np.vstack([outer_p, cand_out_p])
            outer_s = np.concatenate([outer_s, cand_out_s])

        if outer_p.shape[0] > n_outer:
            keep = np.argpartition(np.abs(outer_s), n_outer - 1)[:n_outer]
            outer_p, outer_s = outer_p[keep], outer_s[keep]

        # ----- hole boundary pool (min |s_c| inside s_bs<0) -----
        mask_hole = (s_bs < 0)
        if np.any(mask_hole):
            p_h = p[mask_hole]
            s_geom_h = s_geom[mask_hole]
            s_c_h = s_c[mask_hole]

            abs_c = np.abs(s_c_h)
            k_h = min(p_h.shape[0], max(n_hole * 8, n_hole))
            k_h = max(k_h, 1)
            idx_h = np.argpartition(abs_c, k_h - 1)[:k_h]

            cand_h_p = p_h[idx_h]
            cand_h_sgeom = s_geom_h[idx_h]
            cand_h_sc = s_c_h[idx_h]

            if hole_p is None:
                hole_p, hole_s, hole_sc = cand_h_p, cand_h_sgeom, cand_h_sc
            else:
                hole_p = np.vstack([hole_p, cand_h_p])
                hole_s = np.concatenate([hole_s, cand_h_sgeom])
                hole_sc = np.concatenate([hole_sc, cand_h_sc])

            # 关键：孔洞池裁剪用 |s_c|
            if hole_p.shape[0] > n_hole:
                keep = np.argpartition(np.abs(hole_sc), n_hole - 1)[:n_hole]
                hole_p, hole_s, hole_sc = hole_p[keep], hole_s[keep], hole_sc[keep]

        # ----- check done -----
        interior_now = np.vstack(interior_pool) if len(interior_pool) else np.empty((0, 3))
        hole_count = 0 if hole_p is None else hole_p.shape[0]
        print(f"[round {r+1}] total={total_candidate}, inside={interior_now.shape[0]}, outer={outer_p.shape[0]}, hole={hole_count}")

        if (interior_now.shape[0] >= n_interior) and (hole_p is not None) and (hole_p.shape[0] >= n_hole) and (outer_p.shape[0] >= n_outer):
            sel = np.random.choice(interior_now.shape[0], n_interior, replace=False)
            interior = interior_now[sel]

            boundary = np.vstack([outer_p, hole_p])
            boundary_s = np.concatenate([outer_s, hole_s])
            boundary_tag = np.concatenate([np.zeros(n_outer, dtype=int), np.ones(n_hole, dtype=int)])
            return interior, boundary, boundary_s, boundary_tag

    raise RuntimeError(
        f"Sampling failed after {max_rounds} rounds. "
        f"Try increasing max_rounds/chunk or reducing n_interior/n_boundary."
    )


# ---------- plotting ----------
def plot_boundary_two_layers(points, scalar, tag, title, out_path, outer_keep=20000):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    outer_idx = np.where(tag == 0)[0]
    hole_idx = np.where(tag == 1)[0]

    # 外表面点太多会遮挡：抽样保留一部分即可
    if outer_idx.size > outer_keep:
        outer_idx = np.random.choice(outer_idx, outer_keep, replace=False)

    # # 外表面：灰色、极透明、小点（只作为轮廓参考）
    # ax.scatter(points[outer_idx, 0], points[outer_idx, 1], points[outer_idx, 2],
    #            c="lightgray", s=0.6, alpha=0.03)

    # # 孔洞表面：彩色更突出
    # sc = ax.scatter(points[hole_idx, 0], points[hole_idx, 1], points[hole_idx, 2],
    #                 c=np.abs(scalar[hole_idx]), cmap="viridis",
    #                 s=8, alpha=0.90)
        # ---------- 建议：统一“小点 + 透明”，让点云更细腻 ----------
    # 1) 外表面：依然作为轮廓参考，但提高可见性（否则只剩孔洞粗点）
    ax.scatter(
        points[outer_idx, 0], points[outer_idx, 1], points[outer_idx, 2],
        c="lightgray",
        s=1.2,                 # 小点
        alpha=0.08,            # 比 0.03 更可见
        depthshade=True,
        linewidths=0,
        rasterized=True
    )

    # 2) 孔洞表面：仍然彩色突出，但把点缩小，避免“粗糙颗粒感”
    sc = ax.scatter(
        points[hole_idx, 0], points[hole_idx, 1], points[hole_idx, 2],
        c=np.abs(scalar[hole_idx]),
        cmap="viridis",
        s=2.2,                 # 关键：从 8 降到 ~2
        alpha=0.55,            # 关键：从 0.90 降到 ~0.5
        depthshade=True,
        linewidths=0,
        rasterized=True
    )

    fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1, label="|SDF| (hole points)")

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=18, azim=45)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)
    print("Saved:", out_path)


def plot_holes_only(boundary, boundary_tag, out_path):
    hole = (boundary_tag == 1)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # ax.scatter(boundary[hole, 0], boundary[hole, 1], boundary[hole, 2],
    #            c="crimson", s=8, alpha=0.90)
    ax.scatter(
    boundary[hole, 0], boundary[hole, 1], boundary[hole, 2],
    c="crimson",
    s=1.6,
    alpha=0.45,
    depthshade=True,
    linewidths=0,
    rasterized=True
    )

    ax.set_title("CSG Holes Only (cylinders)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])
    #新添加的部分，在view中添加坐标轴
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(-1.05, 1.05)
    ax.view_init(elev=18, azim=45)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)
    print("Saved:", out_path)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print("CWD =", os.getcwd())

    out_dir = os.path.join(script_dir, "geom_outputs_numpy")
    os.makedirs(out_dir, exist_ok=True)
    print("OUT_DIR =", out_dir)

    interior, boundary, boundary_s, boundary_tag = sample_points(
        bound=1.05,
        n_interior=300000,
        n_boundary=200000,
        hole_ratio=0.60,
        chunk=600000,
        max_rounds=50
    )

    plot_boundary_two_layers(
        boundary,
        boundary_s,
        boundary_tag,
        title="CSG Boundary (outer thinned + holes highlighted)",
        out_path=os.path.join(out_dir, "boundary.png"),
        outer_keep=20000
    )

    plot_holes_only(
        boundary,
        boundary_tag,
        out_path=os.path.join(out_dir, "holes_only.png")
    )

    print("Done.")


if __name__ == "__main__":
    main()