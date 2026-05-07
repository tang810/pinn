import os
import time

import numpy as np
import torch

from .data_loader import load_csv, validate_element
from .interpolation import create_idw_map, create_kriging_like_map, save_heatmap_png, save_summary


def train_model(args):
    # For this project, "train" means building full interpolation artifacts.
    df = load_csv(args.data_csv)
    element = validate_element(df, args.element)

    idw_map, idw_vals = create_idw_map(df, element, grid_size=args.grid_size)
    kr_map, kr_vals = create_kriging_like_map(df, element, grid_size=args.grid_size)

    if idw_map is not None:
        idw_path = os.path.join(args.output_dir, f"{element}_idw_map.html")
        idw_map.save(idw_path)
    else:
        idw_path = os.path.join(args.output_dir, f"{element}_idw_map.png")
        save_heatmap_png(idw_vals, idw_path, f"{element} IDW")

    if kr_map is not None:
        kr_path = os.path.join(args.output_dir, f"{element}_kriging_map.html")
        kr_map.save(kr_path)
    else:
        kr_path = os.path.join(args.output_dir, f"{element}_kriging_map.png")
        save_heatmap_png(kr_vals, kr_path, f"{element} Kriging-like")

    summary = {
        "element": element,
        "rows": int(len(df)),
        "idw_min": float(np.nanmin(idw_vals)),
        "idw_max": float(np.nanmax(idw_vals)),
        "kriging_min": float(np.nanmin(kr_vals)),
        "kriging_max": float(np.nanmax(kr_vals)),
        "timestamp": time.time(),
    }
    save_summary(summary, os.path.join(args.output_dir, f"{element}_summary.json"))

    ckpt = {
        "element": element,
        "data_csv": args.data_csv,
        "grid_size": args.grid_size,
        "output_files": [idw_path, kr_path],
    }
    torch.save(ckpt, os.path.join(args.model_dir, "geochatbot_state.pt"))
    print(f"Saved: {idw_path}")
    print(f"Saved: {kr_path}")
    print(f"Saved: {os.path.join(args.model_dir, 'geochatbot_state.pt')}")
