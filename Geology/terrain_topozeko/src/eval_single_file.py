"""
Notebook-friendly eval file for terrain_topozeko.

Online Jupyter usage:
1. Upload the trained .pt model as a public asset.
2. Upload or keep the terrain .mat dataset as a public asset.
3. Fill MODEL_HASH and DATA_HASH, then run this whole file/cell.

This file loads the model package and part of the dataset, prints final metrics,
and shows figures with plt.show(). It does not save images.
"""

import os

import numpy as np
import matplotlib.pyplot as plt


# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = "4724d61a9bd749aca8439277ccfc26cc"
MODEL_HASH = "719358cc9f3142b58930413e3051f6a7"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/trained_model_terrain_topozeko.pt")

PREFERRED_DATA_FILE = "example_data_Morteratsch_25m.mat"
VERTICAL_SCALE = 4.0


# =========================
# 2. Data and model utilities
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
    return bed, sur


def load_model_package(model_path):
    model_path = os.path.expanduser(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"MODEL_PATH does not exist: {model_path}")
    package = np.load(model_path, allow_pickle=True)
    stats = {
        str(k): float(v)
        for k, v in zip(package["stats_keys"].tolist(), package["stats_values"].tolist())
    }
    return {
        "model_path": model_path,
        "mat_path": str(package["mat_path"].tolist()),
        "bed": package["bed"].astype(np.float32),
        "surface": package["surface"].astype(np.float32),
        "thickness": package["thickness"].astype(np.float32),
        "surface_mask": package["surface_mask"].astype(np.float32),
        "relief": package["relief"].astype(np.float32),
        "stats": stats,
    }


def summarize_against_dataset(package, data_path):
    mat_path = resolve_mat_path(data_path)
    bed, sur = load_bed_sur(mat_path)
    thickness = np.maximum(sur - bed, 0.0)

    n0 = min(thickness.shape[0], package["thickness"].shape[0])
    n1 = min(thickness.shape[1], package["thickness"].shape[1])
    ref = thickness[:n0, :n1]
    pred = package["thickness"][:n0, :n1]
    err = pred - ref
    return {
        "dataset_path": mat_path,
        "shape": tuple(ref.shape),
        "thickness_rmse": float(np.sqrt(np.mean(err**2))),
        "thickness_mae": float(np.mean(np.abs(err))),
        "thickness_relative_l2": float(np.linalg.norm(err) / (np.linalg.norm(ref) + 1e-12)),
    }


# =========================
# 3. Visualization
# =========================

def show_eval_result(package):
    bed = package["bed"]
    sur = package["surface"]
    thickness = package["thickness"]
    relief = package["relief"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    im0 = axes[0, 0].imshow(bed, cmap="terrain")
    axes[0, 0].set_title("Model BED")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(sur, cmap="terrain")
    axes[0, 1].set_title("Model SUR")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[1, 0].imshow(thickness, cmap="viridis")
    axes[1, 0].set_title("Model thickness")
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    im3 = axes[1, 1].imshow(relief, cmap="magma")
    axes[1, 1].set_title("Model relief")
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()

    y = np.arange(sur.shape[0])
    x = np.arange(sur.shape[1])
    xx, yy = np.meshgrid(x, y)
    step = max(1, int(max(sur.shape) / 120))

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        xx[::step, ::step],
        yy[::step, ::step],
        sur[::step, ::step] / VERTICAL_SCALE,
        cmap="terrain",
        linewidth=0,
        antialiased=True,
    )
    ax.set_title("Eval 3D surface topography")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel(f"elevation / {VERTICAL_SCALE:g}")
    plt.tight_layout()
    plt.show()


# =========================
# 4. Eval entry
# =========================

def evaluate(model_path=MODEL_PATH, data_path=DATA_PATH):
    package = load_model_package(model_path)
    metrics = summarize_against_dataset(package, data_path)

    print("Eval result")
    print("model_path:", model_path)
    print("data_path:", metrics["dataset_path"])
    print("shape:", metrics["shape"])
    for key, value in package["stats"].items():
        print(f"{key}: {value:.6g}")
    for key in ("thickness_rmse", "thickness_mae", "thickness_relative_l2"):
        print(f"{key}: {metrics[key]:.6e}")

    show_eval_result(package)
    return {"package": package, "metrics": metrics}


eval_result = evaluate()
