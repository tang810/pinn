# geometry_csg.py
"""
CSG geometry utilities for sampling interior/boundary points + normals.

Conventions
- Solid domain: s_geom(p) < 0
- Boundary:     s_geom(p) = 0
- Outward normal for the solid: n = ∇s_geom / ||∇s_geom||
- Tag: 0=outer surface, 1=hole surfaces
"""
from __future__ import annotations
import numpy as np

# ---------- SDF primitives ----------
def sdf_sphere(p: np.ndarray, center=(0.0, 0.0, 0.0), r=1.2) -> np.ndarray:
    c = np.array(center, dtype=float)[None, :]
    return np.linalg.norm(p - c, axis=1) - float(r)

def sdf_box(p: np.ndarray, p1=(-1.0, -1.0, -1.0), p2=(1.0, 1.0, 1.0)) -> np.ndarray:
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    c = (p1 + p2) / 2.0
    h = (p2 - p1) / 2.0
    q = np.abs(p - c[None, :]) - h[None, :]
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    return outside + inside

def sdf_cylinder_axis(p: np.ndarray, r=0.5, h=2.0, axis="z") -> np.ndarray:
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    r = float(r)
    h = float(h)
    if axis == "z":
        d0 = np.sqrt(x * x + y * y) - r
        d1 = np.abs(z) - h / 2.0
    elif axis == "x":
        d0 = np.sqrt(y * y + z * z) - r
        d1 = np.abs(x) - h / 2.0
    elif axis == "y":
        d0 = np.sqrt(x * x + z * z) - r
        d1 = np.abs(y) - h / 2.0
    else:
        raise ValueError("axis must be x/y/z")

    outside = np.sqrt(np.maximum(d0, 0.0) ** 2 + np.maximum(d1, 0.0) ** 2)
    inside = np.minimum(np.maximum(d0, d1), 0.0)
    return outside + inside

# ---------- CSG ops via SDF ----------
def sdf_union(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.minimum(a, b)

def sdf_intersection(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(a, b)

def sdf_subtract(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(a, -b)

# ---------- Geometry definition ----------
def sdf_geometry_components(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # (box ∩ sphere) \ cylinders
    s_box = sdf_box(p, (-1, -1, -1), (1, 1, 1))
    s_sph = sdf_sphere(p, (0, 0, 0), 1.2)
    s_bs = sdf_intersection(s_box, s_sph)

    s_cz = sdf_cylinder_axis(p, r=0.5, h=2.0, axis="z")
    s_cx = sdf_cylinder_axis(p, r=0.5, h=2.0, axis="x")
    s_cy = sdf_cylinder_axis(p, r=0.5, h=2.0, axis="y")
    s_c = sdf_union(sdf_union(s_cz, s_cx), s_cy)

    s_geom = sdf_subtract(s_bs, s_c)
    return s_geom, s_bs, s_c

def sdf_geometry(p: np.ndarray) -> np.ndarray:
    return sdf_geometry_components(p)[0]

# ---------- Gradients / normals (finite diff) ----------
def _finite_diff_grad(func, p: np.ndarray, eps: float) -> np.ndarray:
    eps = float(eps)
    g = np.zeros_like(p, dtype=float)
    for i in range(3):
        dp = np.zeros_like(p)
        dp[:, i] = eps
        f1 = func(p + dp)
        f2 = func(p - dp)
        g[:, i] = (f1 - f2) / (2.0 * eps)
    return g

def project_to_surface(p: np.ndarray, func=sdf_geometry, eps: float = 1e-3, iters: int = 2) -> np.ndarray:
    p = p.astype(float, copy=True)
    for _ in range(int(iters)):
        s = func(p)                        # (N,)
        g = _finite_diff_grad(func, p, eps=eps)  # (N,3)
        denom = np.sum(g * g, axis=1, keepdims=True) + 1e-12
        p = p - (s[:, None] / denom) * g
    return p

def normals_from_sdf(p_on_surface: np.ndarray, func=sdf_geometry, eps: float = 1e-3) -> np.ndarray:
    g = _finite_diff_grad(func, p_on_surface, eps=eps)
    nrm = np.linalg.norm(g, axis=1, keepdims=True) + 1e-12
    return g / nrm

# ---------- Sampling (ported from your geom_demo.py) ----------
def sample_points(bound=1.05, n_interior=60000, n_boundary=80000,
                  hole_ratio=0.60, chunk=300000, max_rounds=50):
    n_hole = int(round(n_boundary * hole_ratio))
    n_outer = n_boundary - n_hole

    interior_pool = []
    outer_p, outer_s = None, None
    hole_p, hole_s, hole_sc = None, None, None

    for _ in range(int(max_rounds)):
        p = np.random.uniform(-bound, bound, size=(int(chunk), 3))
        s_geom, s_bs, s_c = sdf_geometry_components(p)

        inside = p[s_geom < 0]
        if inside.size > 0:
            interior_pool.append(inside)

        # outer: min |s_geom|
        abs_g = np.abs(s_geom)
        k_out = min(int(chunk), max(n_outer * 3, n_outer))
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

        # holes: min |s_c| within s_bs<0
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
            if hole_p.shape[0] > n_hole:
                keep = np.argpartition(np.abs(hole_sc), n_hole - 1)[:n_hole]
                hole_p, hole_s, hole_sc = hole_p[keep], hole_s[keep], hole_sc[keep]

        interior_now = np.vstack(interior_pool) if len(interior_pool) else np.empty((0, 3))
        hole_count = 0 if hole_p is None else hole_p.shape[0]
        if (interior_now.shape[0] >= n_interior) and (hole_p is not None) and (hole_count >= n_hole) and (outer_p.shape[0] >= n_outer):
            sel = np.random.choice(interior_now.shape[0], n_interior, replace=False)
            interior = interior_now[sel]
            boundary = np.vstack([outer_p, hole_p])
            boundary_tag = np.concatenate([np.zeros(n_outer, dtype=int), np.ones(n_hole, dtype=int)])
            return interior, boundary, boundary_tag

    raise RuntimeError("CSG sampling failed. Increase max_rounds/chunk or reduce points.")

def sample_training_geometry(
    n_interior: int,
    n_boundary: int,
    bound: float = 1.05,
    hole_ratio: float = 0.60,
    chunk: int = 300000,
    max_rounds: int = 50,
    sdf_eps: float = 1e-3,
    proj_iters: int = 2,
    interior_margin: float = 0.0,
):
    interior, boundary, boundary_tag = sample_points(
        bound=bound,
        n_interior=n_interior,
        n_boundary=n_boundary,
        hole_ratio=hole_ratio,
        chunk=chunk,
        max_rounds=max_rounds,
    )

    if interior_margin > 0:
        s_in = sdf_geometry(interior)
        interior = interior[s_in < -float(interior_margin)]
        interior = interior[:n_interior] if interior.shape[0] >= n_interior else interior

    boundary = project_to_surface(boundary, func=sdf_geometry, eps=sdf_eps, iters=proj_iters)
    normals = normals_from_sdf(boundary, func=sdf_geometry, eps=sdf_eps)
    return interior, boundary, normals, boundary_tag