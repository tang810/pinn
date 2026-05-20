"""
Notebook-friendly single-file eval for uniform_raptor3.

Loads the .pt checkpoint produced by train_single_file.py and shows summary
metrics/plots without importing local src modules.
"""

import os

import numpy as np
import torch
import matplotlib.pyplot as plt


MODEL_HASH = "uniform_raptor3"
MODEL_PATH = os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/trained_model_uniform_raptor3.pt")


def get_device():
    """Detect device in priority: GPU (cuda) -> NPU (npu/ascend) -> CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_npu
        if torch_npu.npu.is_available():
            return torch.device("npu")
    except (ImportError, AttributeError):
        pass
    return torch.device("cpu")


def load_torch(path, device="cpu"):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def project_dir():
    try:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        return os.path.abspath(os.path.join(os.getcwd(), "Fluid_Mechanics", "uniform_raptor3"))


def model_candidates(model_path):
    expanded = os.path.expanduser(model_path) if model_path else ""
    return [
        expanded,
        os.path.join(expanded, "trained_model_uniform_raptor3.pt") if expanded else "",
        os.path.expanduser(f"~/public/Resource/{MODEL_HASH}/trained_model_uniform_raptor3.pt"),
        os.path.expanduser(f"~/minio/Resource/{MODEL_HASH}/trained_model_uniform_raptor3.pt"),
        os.path.expanduser(f"~/Resource/{MODEL_HASH}/trained_model_uniform_raptor3.pt"),
        f"/Resource/{MODEL_HASH}/trained_model_uniform_raptor3.pt",
        os.path.join(project_dir(), "model", "trained_model_uniform_raptor3.pt"),
        os.path.join(project_dir(), "trained_model_uniform_raptor3.pt"),
    ]


def resolve_model_path(model_path):
    tried = model_candidates(model_path)
    for path in tried:
        if path and os.path.isfile(path):
            return path
    raise FileNotFoundError("Model checkpoint not found. Tried: " + ", ".join(tried))


def show_eval_result(ckpt):
    nozzle = ckpt["nozzle"]
    plume = ckpt["plume"]
    metrics = ckpt["metrics"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].plot(nozzle["x"], nozzle["M"])
    axes[0, 0].set_title("Nozzle Mach")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(nozzle["x"], np.asarray(nozzle["P"]) / 1e6, label="P MPa")
    axes[0, 1].plot(nozzle["x"], np.asarray(nozzle["T"]) / 1000.0, label="T / 1000 K")
    axes[0, 1].set_title("Nozzle P/T")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    im = axes[1, 0].imshow(
        np.asarray(plume["Mach"]).T,
        origin="lower",
        aspect="auto",
        extent=[float(np.min(plume["x"])), float(np.max(plume["x"])), float(np.min(plume["r"])), float(np.max(plume["r"]))],
    )
    axes[1, 0].set_title("Plume Mach")
    fig.colorbar(im, ax=axes[1, 0])
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.02,
        0.95,
        f"Thrust: {metrics['thrust_N']/1e6:.3f} MN\nIsp: {metrics['isp_s']:.1f} s\nmdot: {metrics['mdot_kg_s']:.1f} kg/s\nExit Mach: {metrics['exit_mach']:.2f}\nExit P: {metrics['exit_pressure_Pa']/1e3:.1f} kPa\nDiamond: {plume['diamond_spacing']:.2f} m",
        va="top",
        family="monospace",
    )
    plt.tight_layout()
    plt.show()


def evaluate(model_path=MODEL_PATH):
    device = get_device()
    print(f"device={device}")
    resolved = resolve_model_path(model_path)
    ckpt = load_torch(resolved, device)
    metrics = ckpt["metrics"]
    print("Eval result")
    print(f"model_path: {resolved}")
    print(f"thrust: {metrics['thrust_N']/1e6:.3f} MN")
    print(f"isp: {metrics['isp_s']:.1f} s")
    print(f"mdot: {metrics['mdot_kg_s']:.1f} kg/s")
    print(f"exit_mach: {metrics['exit_mach']:.3f}")
    show_eval_result(ckpt)
    return ckpt


eval_result = evaluate()
