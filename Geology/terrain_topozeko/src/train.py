import os

import torch

from .data_loader import load_bed_sur
from .topozeko import topozeko


def train_model(args):
    # This project is a visualization pipeline, so "train" keeps behavior by
    # generating outputs and persisting a lightweight run state.
    bed, sur = load_bed_sur(args.data_path, device=args.device)
    out_path = os.path.join(args.output_dir, args.plot_name)
    topozeko(bed, sur, output_path=out_path, d2=args.d2)

    ckpt_path = os.path.join(args.model_dir, "topozeko_state.pt")
    torch.save(
        {
            "data_path": args.data_path,
            "plot_name": args.plot_name,
            "shape": list(bed.shape),
        },
        ckpt_path,
    )
    print(f"Saved visualization to {out_path}")
    print(f"Saved state to {ckpt_path}")

    if args.save_legacy_name and os.path.basename(out_path) != "topozeko_plot.png":
        legacy = os.path.join(args.output_dir, "topozeko_plot.png")
        topozeko(bed, sur, output_path=legacy, d2=False)
        print(f"Also saved legacy figure to {legacy}")
