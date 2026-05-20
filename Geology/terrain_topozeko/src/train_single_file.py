"""
Notebook-friendly train file for terrain_topozeko.

Online Jupyter usage:
1. Upload one terrain .mat dataset as a public asset.
2. Fill DATA_HASH and MODEL_HASH.
3. Run this whole file/cell. It saves trained_model_terrain_topozeko.pt to
   ~/minio/Resource/{MODEL_HASH}/ for later public-asset upload.

Expected dataset file:
    example_data_Morteratsch_25m.mat
or:
    example_data_Galapagos_100m.mat

The .mat file must contain BED and SUR arrays.
"""

import os
import time

import numpy as np
import matplotlib.pyplot as plt


# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = "4724d61a9bd749aca8439277ccfc26cc"
MODEL_HASH = "4724d61a9bd749aca8439277ccfc26cc"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/trained_model_terrain_topozeko.pt")

PREFERRED_DATA_FILE = "example_data_Morteratsch_25m.mat"
VERTICAL_SCALE = 4.0


# =========================
# 2. Data utilities
# =========================

def resolve_mat_path(data_path):
    if not data_path:
        raise FileNotFoundError("DATA_PATH is empty. Fill DATA_HASH first.")

    data_path = os.path.expanduser(data_path)
    candidates = []
    if os.path.isfile(data_path):
        candidates.append(data_path)
    elif os.path.isdir(data_path):
        candidates.append(os.path.join(data_path, PREFERRED_DATA_FILE))
        candidates.append(os.path.join(data_path, "example_data_Galapagos_100m.mat"))
        candidates.append(os.path.join(data_path, "example_data_Morteratsch_25m.mat"))

    for path in candidates:
        if os.path.isfile(path):
            return path

    raise FileNotFoundError(
        "Could not find terrain .mat data. Upload one of these files into the data asset folder: "
        "example_data_Morteratsch_25m.mat or example_data_Galapagos_100m.mat. "
        f"Current DATA_PATH={data_path!r}"
    )


def load_bed_sur(mat_path):
    try:
        import h5py

        with h5py.File(mat_path, "r") as f:
            if "BED" not in f or "SUR" not in f:
                raise KeyError("The .mat file must contain BED and SUR arrays.")
            bed = np.asarray(f["BED"][:], dtype=np.float32)
            sur = np.asarray(f["SUR"][:], dtype=np.float32)
    except (OSError, ImportError, ModuleNotFoundError):
        try:
            from scipy.io import loadmat
        except Exception as exc:
            raise ImportError("Reading non-HDF5 .mat files requires scipy.") from exc

        mat = loadmat(mat_path)
        if "BED" not in mat or "SUR" not in mat:
            raise KeyError("The .mat file must contain BED and SUR arrays.")
        bed = np.asarray(mat["BED"], dtype=np.float32)
        sur = np.asarray(mat["SUR"], dtype=np.float32)

    bed = np.squeeze(bed)
    sur = np.squeeze(sur)
    if bed.shape != sur.shape:
        raise ValueError(f"BED and SUR must have the same shape, got {bed.shape} and {sur.shape}.")
    if bed.ndim != 2:
        raise ValueError(f"BED and SUR must be 2D arrays, got BED shape {bed.shape}.")
    return bed, sur


def build_topography_state(bed, sur):
    thickness = sur - bed
    thickness = np.where(np.isfinite(thickness), thickness, 0.0)
    thickness = np.maximum(thickness, 0.0)
    surface_mask = np.where(thickness > 0, 1.0, 0.0).astype(np.float32)
    relief = sur - np.nanmin(bed)

    return {
        "bed": bed.astype(np.float32),
        "surface": sur.astype(np.float32),
        "thickness": thickness.astype(np.float32),
        "surface_mask": surface_mask,
        "relief": relief.astype(np.float32),
        "stats": {
            "bed_min": float(np.nanmin(bed)),
            "bed_max": float(np.nanmax(bed)),
            "surface_min": float(np.nanmin(sur)),
            "surface_max": float(np.nanmax(sur)),
            "thickness_min": float(np.nanmin(thickness)),
            "thickness_max": float(np.nanmax(thickness)),
            "thickness_mean": float(np.nanmean(thickness)),
            "valid_fraction": float(np.mean(surface_mask)),
        },
    }


def save_state(state, model_path, mat_path):
    model_path = os.path.expanduser(model_path)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    payload = {
        "project": np.array("terrain_topozeko"),
        "mat_path": np.array(mat_path),
        "bed": state["bed"],
        "surface": state["surface"],
        "thickness": state["thickness"],
        "surface_mask": state["surface_mask"],
        "relief": state["relief"],
        "stats_keys": np.array(list(state["stats"].keys())),
        "stats_values": np.array(list(state["stats"].values()), dtype=np.float64),
    }
    with open(model_path, "wb") as f:
        np.savez_compressed(f, **payload)
    return model_path


# =========================
# 3. Visualization
# =========================

def show_result(state):
    bed = state["bed"]
    sur = state["surface"]
    thickness = state["thickness"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    im0 = axes[0].imshow(bed, cmap="terrain")
    axes[0].set_title("BED elevation")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(sur, cmap="terrain")
    axes[1].set_title("SUR elevation")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(thickness, cmap="viridis")
    axes[2].set_title("Ice thickness")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()

    y = np.arange(bed.shape[0])
    x = np.arange(bed.shape[1])
    xx, yy = np.meshgrid(x, y)
    step = max(1, int(max(bed.shape) / 120))

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        xx[::step, ::step],
        yy[::step, ::step],
        sur[::step, ::step] / VERTICAL_SCALE,
        cmap="terrain",
        linewidth=0,
        antialiased=True,
        alpha=0.95,
    )
    ax.set_title("3D surface topography")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel(f"elevation / {VERTICAL_SCALE:g}")
    plt.tight_layout()
    plt.show()


# =========================
# 4. Train entry
# =========================

def train(data_path=DATA_PATH, model_path=MODEL_PATH):
    t0 = time.time()
    mat_path = resolve_mat_path(data_path)
    bed, sur = load_bed_sur(mat_path)
    state = build_topography_state(bed, sur)
    saved_path = save_state(state, model_path, mat_path)

    print("Training finished")
    print("dataset:", mat_path)
    print("model saved to:", saved_path)
    print("shape:", bed.shape)
    print("elapsed:", f"{time.time() - t0:.1f}s")
    print("stats:")
    for key, value in state["stats"].items():
        print(f"  {key}: {value:.6g}")

    show_result(state)
    return state


train_result = train()
