import contextlib
import importlib
import os
import sys
from pathlib import Path


_VARIANT_DIR = {
    "mcf": "01MCFNets",
    "ev_mcf": "01ev_MCFNets",
}


@contextlib.contextmanager
def legacy_import_context(project_root: Path, variant: str):
    legacy_dir = project_root / "src" / "legacy" / _VARIANT_DIR[variant]
    if not legacy_dir.exists():
        raise FileNotFoundError(f"legacy variant folder not found: {legacy_dir}")

    old_cwd = Path.cwd()
    old_path = list(sys.path)
    try:
        os.chdir(str(legacy_dir))
        sys.path.insert(0, str(legacy_dir))
        yield legacy_dir
    finally:
        os.chdir(str(old_cwd))
        sys.path[:] = old_path


def import_legacy_modules(project_root: Path, variant: str):
    with legacy_import_context(project_root, variant):
        cavity_data = importlib.import_module("cavity_data")
        pinn_solver = importlib.import_module("pinn_solver")
    return cavity_data, pinn_solver


def find_checkpoint(legacy_dir: Path):
    candidates = sorted(legacy_dir.glob("results/**/*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def default_eval_data(legacy_dir: Path, re: int, pr: float, ri: float, variant: str):
    if variant == "mcf":
        return legacy_dir / "data" / f"cavity_Re{re}Pr{pr}Ri{ri}_256.mat"
    return legacy_dir / "data" / f"cavity_Re{re}Pr{pr}Ri{ri}_256.mat"
