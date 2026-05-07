import torch


def _build_simulated_topography(n=128):
    y, x = torch.meshgrid(
        torch.linspace(-3.14, 3.14, n),
        torch.linspace(-3.14, 3.14, n),
        indexing="ij",
    )
    bed = 200 * torch.sin(x) * torch.cos(y)
    sur = bed + 120 + 20 * torch.sin(2 * x + y)
    return bed, sur


def load_bed_sur(mat_path, device="cpu"):
    if mat_path == "simul":
        bed, sur = _build_simulated_topography()
        return bed.to(device), sur.to(device)

    bed = sur = None

    try:
        import h5py  # type: ignore

        with h5py.File(mat_path, "r") as mat_file:
            if "BED" not in mat_file or "SUR" not in mat_file:
                raise KeyError("Missing BED or SUR in .mat file")
            bed = torch.tensor(mat_file["BED"][:], dtype=torch.float32)
            sur = torch.tensor(mat_file["SUR"][:], dtype=torch.float32)
    except ModuleNotFoundError:
        from scipy.io import loadmat  # type: ignore

        try:
            mat = loadmat(mat_path)
            if "BED" not in mat or "SUR" not in mat:
                raise KeyError("Missing BED or SUR in .mat file")
            bed = torch.tensor(mat["BED"], dtype=torch.float32)
            sur = torch.tensor(mat["SUR"], dtype=torch.float32)
        except NotImplementedError:
            # matlab v7.3 requires h5py; fallback keeps quick mode runnable.
            bed, sur = _build_simulated_topography(n=120)

    return bed.to(device), sur.to(device)
