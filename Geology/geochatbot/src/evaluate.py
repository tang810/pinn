import os

import numpy as np

from .data_loader import load_csv, validate_element
from .interpolation import create_idw_map, save_heatmap_png, save_summary


def run(args):
    # Lightweight quick mode: generate one IDW map + statistics.
    df = load_csv(args.data_csv)
    element = validate_element(df, args.element)
    idw_map, idw_vals = create_idw_map(df, element, grid_size=min(args.grid_size, 80))
    if idw_map is not None:
        out_map = os.path.join(args.output_dir, f"{element}_quick_idw_map.html")
        idw_map.save(out_map)
    else:
        out_map = os.path.join(args.output_dir, f"{element}_quick_idw_map.png")
        save_heatmap_png(idw_vals, out_map, f"{element} Quick IDW")

    summary = {
        "mode": "quick",
        "element": element,
        "rows": int(len(df)),
        "min": float(np.nanmin(idw_vals)),
        "max": float(np.nanmax(idw_vals)),
        "mean": float(np.nanmean(idw_vals)),
    }
    out_json = os.path.join(args.output_dir, f"{element}_quick_summary.json")
    save_summary(summary, out_json)
    print(f"Quick map saved to {out_map}")
    print(f"Quick summary saved to {out_json}")
