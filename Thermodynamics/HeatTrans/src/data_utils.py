import os
import numpy as np


def nearest_index(points, target):
    distances = np.linalg.norm(points - target.reshape(1, -1), axis=1)
    return int(np.argmin(distances))


def compute_delta_t_by_space(raw_data, round_decimals=8):
    xyz = np.round(raw_data[:, 0:3].astype(np.float32), round_decimals)
    temp = raw_data[:, 4].astype(np.float32)

    unique_xyz, inverse = np.unique(xyz, axis=0, return_inverse=True)
    temp_min = np.full(unique_xyz.shape[0], np.inf, dtype=np.float32)
    temp_max = np.full(unique_xyz.shape[0], -np.inf, dtype=np.float32)
    np.minimum.at(temp_min, inverse, temp)
    np.maximum.at(temp_max, inverse, temp)

    return unique_xyz.astype(np.float32), (temp_max - temp_min).astype(np.float32)


def normalize_score(values):
    values = values.astype(np.float32)
    v_min = np.min(values)
    v_max = np.max(values)
    if v_max - v_min < 1e-12:
        return np.ones_like(values, dtype=np.float32)
    return (values - v_min) / (v_max - v_min)


def surface_mask_for_points(points, lb_xyz, ub_xyz, tol):
    return (
        np.isclose(points[:, 0], lb_xyz[0], atol=tol)
        | np.isclose(points[:, 0], ub_xyz[0], atol=tol)
        | np.isclose(points[:, 1], lb_xyz[1], atol=tol)
        | np.isclose(points[:, 1], ub_xyz[1], atol=tol)
        | np.isclose(points[:, 2], lb_xyz[2], atol=tol)
        | np.isclose(points[:, 2], ub_xyz[2], atol=tol)
    )


def choose_best_candidate(points, delta_t, candidate_indices, target, selected_keys, round_decimals=8):
    if len(candidate_indices) == 0:
        raise RuntimeError("No candidate points available for sensor selection.")

    candidate_points = points[candidate_indices]
    candidate_delta = normalize_score(delta_t[candidate_indices])

    domain_span = np.maximum(points.max(axis=0) - points.min(axis=0), 1e-6)
    normalized_distance = np.linalg.norm((candidate_points - target.reshape(1, -1)) / domain_span, axis=1)
    center_score = 1.0 / (1.0 + normalized_distance)

    score = 0.65 * candidate_delta + 0.35 * center_score
    order = np.argsort(score)[::-1]

    for local_idx in order:
        point = candidate_points[local_idx]
        key = tuple(np.round(point, round_decimals))
        if key not in selected_keys:
            return point.astype(np.float32)

    raise RuntimeError("All candidate points are already selected.")


def choose_surface_mirror_pair(points, delta_t, candidate_indices, lb_xyz, ub_xyz, selected_keys, round_decimals=8):
    if len(candidate_indices) == 0:
        raise RuntimeError("No side-surface candidates available for mirror-pair selection.")

    order = candidate_indices[np.argsort(delta_t[candidate_indices])[::-1]]
    surface_points = points[candidate_indices]

    for idx in order:
        point = points[idx]
        key = tuple(np.round(point, round_decimals))
        if key in selected_keys:
            continue

        reflected_target = np.array(
            [
                point[0],
                lb_xyz[1] + ub_xyz[1] - point[1],
                lb_xyz[2] + ub_xyz[2] - point[2],
            ],
            dtype=np.float32,
        )
        reflected_idx = nearest_index(surface_points, reflected_target)
        reflected_point = surface_points[reflected_idx]
        reflected_key = tuple(np.round(reflected_point, round_decimals))

        if reflected_key in selected_keys or reflected_key == key:
            continue

        return point.astype(np.float32), reflected_point.astype(np.float32)

    raise RuntimeError("Could not find a valid y/z mirrored surface sensor pair.")


def build_sensor_locations(raw_data, n_sensors, surface_tol, seed=1234, round_decimals=8):
    if n_sensors not in [3, 5]:
        raise ValueError("This surface layout currently supports only 3 or 5 sensors.")

    np.random.RandomState(seed)
    unique_xyz, delta_t = compute_delta_t_by_space(raw_data, round_decimals=round_decimals)

    lb_xyz = unique_xyz.min(axis=0)
    ub_xyz = unique_xyz.max(axis=0)
    center = 0.5 * (lb_xyz + ub_xyz)

    surface_mask = surface_mask_for_points(unique_xyz, lb_xyz, ub_xyz, tol=surface_tol)
    x0_mask = surface_mask & np.isclose(unique_xyz[:, 0], lb_xyz[0], atol=surface_tol)
    x1_mask = surface_mask & np.isclose(unique_xyz[:, 0], ub_xyz[0], atol=surface_tol)
    side_mask = surface_mask & (~x0_mask) & (~x1_mask)

    selected = []
    selected_keys = set()

    def add_point(point):
        key = tuple(np.round(point, round_decimals))
        if key in selected_keys:
            return False
        selected.append(point.astype(np.float32))
        selected_keys.add(key)
        return True

    source_target = np.array([ub_xyz[0], center[1], center[2]], dtype=np.float32)
    source_point = choose_best_candidate(unique_xyz, delta_t, np.where(x1_mask)[0], source_target, selected_keys, round_decimals)
    add_point(source_point)

    opposite_target = np.array([lb_xyz[0], center[1], center[2]], dtype=np.float32)
    opposite_point = choose_best_candidate(unique_xyz, delta_t, np.where(x0_mask)[0], opposite_target, selected_keys, round_decimals)
    add_point(opposite_point)

    if n_sensors == 3:
        side_indices = np.where(side_mask)[0]
        side_point = choose_best_candidate(unique_xyz, delta_t, side_indices, center, selected_keys, round_decimals)
        add_point(side_point)
    else:
        side_indices = np.where(side_mask)[0]
        pair_a, pair_b = choose_surface_mirror_pair(unique_xyz, delta_t, side_indices, lb_xyz, ub_xyz, selected_keys, round_decimals)
        add_point(pair_a)
        add_point(pair_b)

        remaining_surface_indices = np.where(surface_mask)[0]
        extra_point = choose_best_candidate(unique_xyz, delta_t, remaining_surface_indices, center, selected_keys, round_decimals)
        add_point(extra_point)

    if len(selected) != n_sensors:
        raise RuntimeError("Surface sensor selection failed to build the requested layout.")

    return np.array(selected, dtype=np.float32)


def build_sensor_training_data(raw_data, n_sensors, n_time_steps, surface_tol, seed=1234, round_decimals=8):
    raw_data = raw_data.astype(np.float32)

    sensor_xyz = build_sensor_locations(raw_data, n_sensors, surface_tol, seed=seed, round_decimals=round_decimals)
    unique_times = np.unique(np.round(raw_data[:, 3], round_decimals)).astype(np.float32)

    if n_time_steps > len(unique_times):
        raise ValueError(f"n_time_steps cannot exceed available time levels: {n_time_steps} > {len(unique_times)}")

    time_indices = np.linspace(0, len(unique_times) - 1, n_time_steps, dtype=int)
    selected_times = unique_times[time_indices]

    lookup = {}
    rounded_data = raw_data.copy()
    rounded_data[:, 0:4] = np.round(rounded_data[:, 0:4], round_decimals)

    for row_idx, row in enumerate(rounded_data):
        lookup[tuple(row[0:4])] = row_idx

    selected_indices = []
    missing = []

    for xyz in sensor_xyz:
        xyz_key = tuple(np.round(xyz, round_decimals))
        for t in selected_times:
            key = xyz_key + (float(np.round(t, round_decimals)),)
            row_idx = lookup.get(key)
            if row_idx is None:
                missing.append(key)
            else:
                selected_indices.append(row_idx)

    if missing:
        raise RuntimeError(f"Some sensor/time combinations are missing. Missing count: {len(missing)}")

    selected_indices = np.array(selected_indices, dtype=int)
    X_u = raw_data[selected_indices, 0:4].astype(np.float32)
    u = raw_data[selected_indices, 4:5].astype(np.float32)

    return X_u, u, sensor_xyz, selected_times


def sample_rows(data, n_samples, seed):
    if data.shape[0] <= n_samples:
        return data.astype(np.float32)
    rng = np.random.RandomState(seed)
    idx = rng.choice(data.shape[0], size=n_samples, replace=False)
    return data[idx, :].astype(np.float32)


def build_bc_ic_points(raw_data, lb, ub, n_bc_per_face, n_ic, surface_tol, seed=1234):
    X_all = raw_data[:, 0:4].astype(np.float32)

    masks = {
        "x0": np.isclose(X_all[:, 0], lb[0], atol=surface_tol),
        "x1": np.isclose(X_all[:, 0], ub[0], atol=surface_tol),
        "y0": np.isclose(X_all[:, 1], lb[1], atol=surface_tol),
        "y1": np.isclose(X_all[:, 1], ub[1], atol=surface_tol),
        "z0": np.isclose(X_all[:, 2], lb[2], atol=surface_tol),
        "z1": np.isclose(X_all[:, 2], ub[2], atol=surface_tol),
    }

    X_bc = {}
    for i, key in enumerate(["x0", "x1", "y0", "y1", "z0", "z1"]):
        X_bc[key] = sample_rows(X_all[masks[key]], n_bc_per_face, seed + i)

    ic_mask = np.isclose(X_all[:, 3], lb[3], atol=surface_tol)
    X_ic = sample_rows(X_all[ic_mask], n_ic, seed + 99)
    return X_bc, X_ic


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    return np.loadtxt(path).astype(np.float32)
