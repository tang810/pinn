import numpy as np
from physicsnemo.sym.geometry.primitives_3d import Box, Sphere, Cylinder

#demo_csg
def build_demo_csg():
    box = Box(point_1=(-1, -1, -1), point_2=(1, 1, 1))
    sphere = Sphere(center=(0, 0, 0), radius=1.2)
    cyl = Cylinder(center=(0, 0, 0), radius=0.5, height=2)
    cyl_x = cyl.rotate(angle=float(np.pi / 2.0), axis="x")
    cyl_y = cyl.rotate(angle=float(np.pi / 2.0), axis="y")

    all_cyl = cyl + cyl_x + cyl_y
    bs = box & sphere
    geo = bs - all_cyl
    return geo, bs, all_cyl
#生成box（长方体正方体形）
def build_box_geom(p1=(-1, -1, -1), p2=(1, 1, 1)):
    from physicsnemo.sym.geometry.primitives_3d import Box
    geo = Box(point_1=p1, point_2=p2)
    return geo

def _xyz(s):
    return np.hstack([s["x"], s["y"], s["z"]])

def _nrm(s):
    return np.hstack([s["normal_x"], s["normal_y"], s["normal_z"]])

# def sample_training_geometry_sym(n_interior: int, n_boundary: int, hole_tol: float = 1e-2):
#     geo, bs, all_cyl = build_demo_csg()

#     sb = geo.sample_boundary(nr_points=n_boundary)
#     xyz_b = _xyz(sb)
#     nrm_b = _nrm(sb)

#     # Sym SDF: inside is positive, outside is negative
#     sc = all_cyl.sdf({"x": sb["x"], "y": sb["y"], "z": sb["z"]}, params={})["sdf"].reshape(-1)
#     sbs = bs.sdf({"x": sb["x"], "y": sb["y"], "z": sb["z"]}, params={})["sdf"].reshape(-1)

#     # More robust hole detection: boundary points closer to cylinder surface than to bs boundary
#     abs_sc = np.abs(sc)
#     abs_bs = np.abs(sbs)
#     hole = (sbs > 0) & (abs_sc < abs_bs)

#     tag = hole.astype(np.int64)  # (N,)  1=hole, 0=outer

#     si = geo.sample_interior(nr_points=n_interior, compute_sdf_derivatives=True)
#     xyz_i = _xyz(si)

#     return xyz_i, xyz_b, nrm_b, tag
def sample_training_geometry_sym(n_interior, n_boundary, shape='csg', box_p1=(-1,-1,-1), box_p2=(1,1,1)):
    if shape == 'box':
        geo = build_box_geom(box_p1, box_p2)

        sb = geo.sample_boundary(nr_points=n_boundary)
        xyz_b = _xyz(sb)
        nrm_b = _nrm(sb)
        tag = np.zeros((xyz_b.shape[0],), dtype=np.int64)  # 全 outer

        si = geo.sample_interior(nr_points=n_interior, compute_sdf_derivatives=True)
        xyz_i = _xyz(si)

        return xyz_i, xyz_b, nrm_b, tag

    # shape == 'csg'：保持原来的 demo CSG 逻辑不变
    geo, bs, all_cyl = build_demo_csg()
    sb = geo.sample_boundary(nr_points=n_boundary)
    xyz_b = _xyz(sb)
    nrm_b = _nrm(sb)

    sc = all_cyl.sdf({"x": sb["x"], "y": sb["y"], "z": sb["z"]}, params={})["sdf"].reshape(-1)
    sbs = bs.sdf({"x": sb["x"], "y": sb["y"], "z": sb["z"]}, params={})["sdf"].reshape(-1)
    abs_sc = np.abs(sc)
    abs_bs = np.abs(sbs)
    hole = (sbs > 0) & (abs_sc < abs_bs)
    tag = hole.astype(np.int64)

    si = geo.sample_interior(nr_points=n_interior, compute_sdf_derivatives=True)
    xyz_i = _xyz(si)
    return xyz_i, xyz_b, nrm_b, tag
