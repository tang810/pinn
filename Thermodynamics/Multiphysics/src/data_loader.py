import numpy as np
import torch
import pandas as pd
from .utils import LHSample
# Optional: Local SDF CSG geometry sampling
try:
    from .geometry_csg import sample_training_geometry
except Exception:
    sample_training_geometry = None

# Optional: PhysicsNeMo Sym geometry sampling
try:
    from .geometry_sym_backend import sample_training_geometry_sym
except Exception:
    sample_training_geometry_sym = None

def make_time_sequence(xyz_t: torch.Tensor, time_steps: int, step_size: float) -> torch.Tensor:
    """
    Create a time sequence for each spatial point.
    xyz_t: tensor of shape (N, 4) containing [x, y, z, t]
    Returns: tensor of shape (N, time_steps, 4)
    """
    N = xyz_t.shape[0]
    seq = xyz_t.unsqueeze(1).repeat(1, time_steps, 1)
    for i in range(time_steps):
        seq[:, i, 3] = seq[:, i, 3] + i * step_size
    return seq

def sample_interior_points(num_points, sampler='lhs'):
    if sampler == 'grid':
        n = int(round(num_points ** (1/3)))
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
        z = np.linspace(0, 1, n)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        pts = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
        pts = pts[:num_points]
    else:
        pts = LHSample(3, [[0, 1], [0, 1], [0, 1]], num_points)
    t = np.random.rand(num_points, 1)
    pts = np.hstack([pts, t])
    return pts

def sample_initial_points(num_points, T0=273.15, sampler='lhs'):
    if sampler == 'grid':
        n = int(round(num_points ** (1/3)))
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
        z = np.linspace(0, 1, n)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        pts = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
        pts = pts[:num_points]
    else:
        pts = LHSample(3, [[0, 1], [0, 1], [0, 1]], num_points)
    t = np.zeros((num_points, 1))
    pts = np.hstack([pts, t])
    T = np.full((num_points, 1), T0)
    return pts, T

def sample_boundary_points(num_points, sampler='lhs'):
    # Sample points on the boundaries of the cube [0,1]^3
    pts = []
    # Distribute points equally among 6 faces
    per_face = num_points // 6
    faces = [
        ('x', 0.0), ('x', 1.0),
        ('y', 0.0), ('y', 1.0),
        ('z', 0.0), ('z', 1.0),
    ]
    for axis, val in faces:
        if sampler == 'grid':
            n = int(round(per_face ** 0.5))
            a = np.linspace(0, 1, n)
            b = np.linspace(0, 1, n)
            A, B = np.meshgrid(a, b, indexing='ij')
            A = A.ravel()[:per_face]
            B = B.ravel()[:per_face]
        else:
            AB = LHSample(2, [[0, 1], [0, 1]], per_face)
            A, B = AB[:, 0], AB[:, 1]

        x = np.zeros(per_face)
        y = np.zeros(per_face)
        z = np.zeros(per_face)
        if axis == 'x':
            x[:] = val
            y[:] = A
            z[:] = B
        elif axis == 'y':
            y[:] = val
            x[:] = A
            z[:] = B
        else:
            z[:] = val
            x[:] = A
            y[:] = B

        t = np.random.rand(per_face)
        pts_face = np.vstack([x, y, z, t]).T
        pts.append(pts_face)

    pts = np.vstack(pts)
    return pts
def sample_boundary_points_cube(num_points: int, sampler: str = 'lhs'):
    pts = []
    nrms = []

    per_face = num_points // 6
    faces = [
        ('x', 0.0, (-1.0, 0.0, 0.0)), ('x', 1.0, (1.0, 0.0, 0.0)),
        ('y', 0.0, (0.0, -1.0, 0.0)), ('y', 1.0, (0.0, 1.0, 0.0)),
        ('z', 0.0, (0.0, 0.0, -1.0)), ('z', 1.0, (0.0, 0.0, 1.0)),
    ]
    for axis, val, nrm in faces:
        if sampler == 'grid':
            n = int(round(per_face ** 0.5))
            a = np.linspace(0, 1, n)
            b = np.linspace(0, 1, n)
            A, B = np.meshgrid(a, b, indexing='ij')
            A = A.ravel()[:per_face]
            B = B.ravel()[:per_face]
        else:
            AB = LHSample(2, [[0, 1], [0, 1]], per_face)
            A, B = AB[:, 0], AB[:, 1]

        x = np.zeros(per_face); y = np.zeros(per_face); z = np.zeros(per_face)
        if axis == 'x':
            x[:] = val; y[:] = A; z[:] = B
        elif axis == 'y':
            y[:] = val; x[:] = A; z[:] = B
        else:
            z[:] = val; x[:] = A; y[:] = B

        t = np.random.rand(per_face)
        pts_face = np.vstack([x, y, z, t]).T
        pts.append(pts_face)
        nrms.append(np.tile(np.array(nrm, dtype=float)[None, :], (per_face, 1)))

    pts = np.vstack(pts)
    nrm = np.vstack(nrms)
    tag = np.zeros((pts.shape[0],), dtype=np.int64)  # cube: 全部视作 outer
    return pts, nrm, tag

def get_training_data(args, device):
    """
    Returns:
      res_seq: (Nres, T, 4)
      bc_seq: (Nbc,  T, 4)
      bc_normals: (Nbc, 3)
      bc_tag: (Nbc,) 0=outer, 1=holes
      ic_seq: (Nic,  T, 4)
      ic_T: (Nic, 1)
    """
    geometry = getattr(args, 'geometry', 'cube')

    if geometry == 'csg':
        backend = getattr(args, 'geometry_backend', 'sdf')

        n_int_total = max(int(args.res_points), int(args.ic_points))

        # if backend == 'sym':
        #     # PhysicsNeMo Sym backend
        #     if sample_training_geometry_sym is None:
        #         raise RuntimeError("geometry_sym_backend.py not available, cannot use --geometry_backend sym")

        #     interior_xyz, boundary_xyz, boundary_normals, boundary_tag = sample_training_geometry_sym(
        #         n_interior=n_int_total,
        #         n_boundary=int(args.bc_points),
        #     )
        # else:
        #     # Local SDF backend (geometry_csg.py)
        #     if sample_training_geometry is None:
        #         raise RuntimeError("geometry_csg.py not available, cannot use --geometry csg")
        if backend == 'sym':
            # PhysicsNeMo Sym backend
            if sample_training_geometry_sym is None:
                raise RuntimeError("geometry_sym_backend.py not available, cannot use --geometry_backend sym")

            # 关键：sym 后端要吃 sym_shape / box_p1 / box_p2
            sym_shape = getattr(args, 'sym_shape', 'csg')
            box_p1 = getattr(args, 'box_p1', (-1.0, -1.0, -1.0))
            box_p2 = getattr(args, 'box_p2', ( 1.0,  1.0,  1.0))

            interior_xyz, boundary_xyz, boundary_normals, boundary_tag = sample_training_geometry_sym(
                n_interior=n_int_total,
                n_boundary=int(args.bc_points),
                shape=sym_shape,
                box_p1=box_p1,
                box_p2=box_p2,
            )
        # else:
        #     # Local SDF backend (geometry_csg.py)
        #     if sample_training_geometry is None:
        #         raise RuntimeError("geometry_csg.py not available, cannot use --geometry csg")

        #     # SDF 后端不要传 shape/box_p1/box_p2（它不支持也不需要）
        #     interior_xyz, boundary_xyz, boundary_normals, boundary_tag = sample_training_geometry(
        #         n_interior=n_int_total,
        #         n_boundary=int(args.bc_points),
        #         bound=float(getattr(args, 'csg_bound', 1.05)),
        #         hole_ratio=float(getattr(args, 'csg_hole_ratio', 0.60)),
        #         chunk=int(getattr(args, 'csg_chunk', 300000)),
        #         max_rounds=int(getattr(args, 'csg_max_rounds', 50)),
        #         sdf_eps=float(getattr(args, 'sdf_eps', 1e-3)),
        #         proj_iters=int(getattr(args, 'proj_iters', 2)),
        #         interior_margin=float(getattr(args, 'interior_margin', 0.0)),
        #     )
        else:
            # Local SDF backend (geometry_csg.py)
            if sample_training_geometry is None:
                raise RuntimeError("geometry_csg.py not available, cannot use --geometry csg")

            interior_xyz, boundary_xyz, boundary_normals, boundary_tag = sample_training_geometry(
                n_interior=n_int_total,
                n_boundary=int(args.bc_points),
                bound=float(getattr(args, 'csg_bound', 1.05)),
                hole_ratio=float(getattr(args, 'csg_hole_ratio', 0.60)),
                chunk=int(getattr(args, 'csg_chunk', 300000)),
                max_rounds=int(getattr(args, 'csg_max_rounds', 50)),
                sdf_eps=float(getattr(args, 'sdf_eps', 1e-3)),
                proj_iters=int(getattr(args, 'proj_iters', 2)),
                interior_margin=float(getattr(args, 'interior_margin', 0.0)),
            )

            # #box数据checkpoint
            # sym_shape = getattr(args, 'sym_shape', 'csg')
            # box_p1 = tuple(getattr(args, 'box_p1', [-1.0, -1.0, -1.0]))
            # box_p2 = tuple(getattr(args, 'box_p2', [ 1.0,  1.0,  1.0]))
            # interior_xyz, boundary_xyz, boundary_normals, boundary_tag = sample_training_geometry(
            #     n_interior=n_int_total,
            #     n_boundary=int(args.bc_points),
            #     #新加入的功能
            #     shape=sym_shape,
            #     box_p1=box_p1,
            #     box_p2=box_p2,
            #     #
            #     bound=float(getattr(args, 'csg_bound', 1.05)),
            #     hole_ratio=float(getattr(args, 'csg_hole_ratio', 0.60)),
            #     chunk=int(getattr(args, 'csg_chunk', 300000)),
            #     max_rounds=int(getattr(args, 'csg_max_rounds', 50)),
            #     sdf_eps=float(getattr(args, 'sdf_eps', 1e-3)),
            #     proj_iters=int(getattr(args, 'proj_iters', 2)),
            #     interior_margin=float(getattr(args, 'interior_margin', 0.0)),
            # )

        # ensure tag is 1D
        boundary_tag = boundary_tag.reshape(-1)

        # residual points
        res_xyz = interior_xyz[: int(args.res_points)]
        res_t = np.random.rand(res_xyz.shape[0], 1)
        res_pts = np.hstack([res_xyz, res_t])

        # boundary points
        bc_t = np.random.rand(boundary_xyz.shape[0], 1)
        bc_pts = np.hstack([boundary_xyz, bc_t])

        # initial condition points
        ic_xyz = interior_xyz[-int(args.ic_points):]
        ic_t = np.zeros((ic_xyz.shape[0], 1))
        ic_pts = np.hstack([ic_xyz, ic_t])
        ic_T = np.full((ic_xyz.shape[0], 1), float(args.T0))

        res_tensor = torch.tensor(res_pts, dtype=torch.float32, requires_grad=True).to(device)
        bc_tensor = torch.tensor(bc_pts, dtype=torch.float32, requires_grad=True).to(device)
        ic_tensor = torch.tensor(ic_pts, dtype=torch.float32, requires_grad=True).to(device)
        ic_T_tensor = torch.tensor(ic_T, dtype=torch.float32).to(device)

        bc_normals_tensor = torch.tensor(boundary_normals, dtype=torch.float32).to(device)
        bc_tag_tensor = torch.tensor(boundary_tag, dtype=torch.long).to(device)

    else:
        # cube: 原逻辑 + normals/tag
        res_pts = sample_interior_points(args.res_points, sampler=args.sampler)
        res_tensor = torch.tensor(res_pts, dtype=torch.float32, requires_grad=True).to(device)

        bc_pts, bc_normals, bc_tag = sample_boundary_points_cube(args.bc_points, sampler=args.sampler)
        bc_tensor = torch.tensor(bc_pts, dtype=torch.float32, requires_grad=True).to(device)
        bc_normals_tensor = torch.tensor(bc_normals, dtype=torch.float32).to(device)
        bc_tag_tensor = torch.tensor(bc_tag, dtype=torch.long).to(device)

        ic_pts, ic_T = sample_initial_points(args.ic_points, T0=args.T0, sampler=args.sampler)
        ic_tensor = torch.tensor(ic_pts, dtype=torch.float32, requires_grad=True).to(device)
        ic_T_tensor = torch.tensor(ic_T, dtype=torch.float32).to(device)

    # Create time sequences
    res_seq = make_time_sequence(res_tensor, args.time_steps, args.step_size)
    bc_seq = make_time_sequence(bc_tensor, args.time_steps, args.step_size)
    ic_seq = make_time_sequence(ic_tensor, args.time_steps, args.step_size)

    return res_seq, bc_seq, bc_normals_tensor, bc_tag_tensor, ic_seq, ic_T_tensor
def get_evaluation_data(data_path):
    data = pd.read_csv(data_path, delimiter=' ', header=None)
    x = data.iloc[:, 0].values
    y = data.iloc[:, 1].values
    z = data.iloc[:, 2].values
    t = data.iloc[:, 3].values
    T = data.iloc[:, 4].values
    return x, y, z, t, T