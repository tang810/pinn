import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _ensure_dirs(args):
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)


def _make_simul_data(path, seed=42, n=220):
    rng = np.random.default_rng(seed)
    x = np.linspace(-3.0, 8.0, n)
    y = np.linspace(-2.0, 2.0, n // 2)
    xx, yy = np.meshgrid(x, y)
    phase = 0.6
    u = 1.0 - 0.55 * np.exp(-((xx - 0.8) ** 2 + (1.4 * yy) ** 2) / 2.2)
    v = 0.25 * np.exp(-((xx - 1.2) ** 2 + yy**2) / 4.0) * np.sin(yy * 3.0 + phase)
    p = 0.45 * np.exp(-((xx + 0.2) ** 2 + yy**2) / 0.7) - 0.03 * xx
    np.savez(path, x=xx, y=yy, u=u, v=v, p=p)
    return xx, yy, u, v, p


def _load_or_create_data(args):
    data_path = Path(args.data_path)
    if args.data == "simul":
        xx, yy, u, v, p = _make_simul_data(data_path, seed=args.seed)
        return data_path, xx, yy, u, v, p
    if not data_path.exists():
        raise FileNotFoundError(f"Real data not found: {data_path}")
    d = np.load(data_path)
    return data_path, d["x"], d["y"], d["u"], d["v"], d["p"]


def _save_quick_results(result_dir, xx, yy, u, v, p):
    result_dir = Path(result_dir)
    speed = np.sqrt(u**2 + v**2)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=160)
    for ax, val, title in [
        (axes[0], speed, "Speed"),
        (axes[1], p, "Pressure"),
        (axes[2], np.abs(np.gradient(speed, axis=1)), "Speed Grad |dx|"),
    ]:
        im = ax.imshow(
            val,
            cmap="jet",
            origin="lower",
            extent=[xx.min(), xx.max(), yy.min(), yy.max()],
            aspect="auto",
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle("DEV-ACDM4Science Quick Result", fontsize=13)
    fig.tight_layout()
    fig.savefig(result_dir / "quick_result.png")
    plt.close(fig)

    np.savez(result_dir / "quick_result_fields.npz", x=xx, y=yy, u=u, v=v, p=p, speed=speed)


def run_quick(args):
    _ensure_dirs(args)
    data_path, xx, yy, u, v, p = _load_or_create_data(args)
    _save_quick_results(args.result_dir, xx, yy, u, v, p)

    summary = {
        "mode": "quick",
        "task": args.task,
        "data": args.data,
        "data_path": str(data_path),
        "result_image": str(Path(args.result_dir) / "quick_result.png"),
        "result_fields": str(Path(args.result_dir) / "quick_result_fields.npz"),
    }
    (Path(args.result_dir) / "quick_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[quick] Result image saved to {Path(args.result_dir) / 'quick_result.png'}")


def _build_original_command(args):
    scripts = {
        "acdm_cylinder": Path("src/scripts/run_acdm.py"),
        "acdm_foilslice": Path("src/scripts/run_acdm_foilslice.py"),
        "ddpm": Path("src/scripts/run_ddpm.py"),
    }
    script = scripts[args.task]
    if args.task == "ddpm":
        return [sys.executable, str(script)]
    return [sys.executable, str(script), f"--data-path={args.data_path}"]


def run_train(args):
    _ensure_dirs(args)
    if args.delegate_original:
        cmd = _build_original_command(args)
        print("[train] Delegating to original script:", " ".join(cmd))
        completed = subprocess.run(cmd, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"Original script failed with code {completed.returncode}")
    else:
        # Lightweight fallback that preserves standardized workflow.
        print("[train] delegate_original is False, running standardized lightweight training surrogate.")
        data_path, xx, yy, u, v, p = _load_or_create_data(args)
        # tiny denoise-like smoothing as placeholder training output
        k = 0.85
        u2 = k * u + (1 - k) * np.mean(u)
        v2 = k * v + (1 - k) * np.mean(v)
        p2 = k * p + (1 - k) * np.mean(p)
        np.savez(Path(args.model_dir) / "train_model_surrogate.npz", u=u2, v=v2, p=p2)
        _save_quick_results(args.result_dir, xx, yy, u2, v2, p2)
        (Path(args.result_dir) / "train_summary.json").write_text(
            json.dumps(
                {
                    "mode": "train",
                    "data": args.data,
                    "data_path": str(data_path),
                    "model": str(Path(args.model_dir) / "train_model_surrogate.npz"),
                    "result_image": str(Path(args.result_dir) / "quick_result.png"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    print("[train] Done.")

