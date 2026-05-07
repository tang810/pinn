import os

from .data_loader import load_bed_sur
from .topozeko import topozeko


def run(args):
    bed, sur = load_bed_sur(args.data_path, device=args.device)
    out_path = os.path.join(args.output_dir, args.plot_name)
    topozeko(bed, sur, output_path=out_path, d2=args.d2)
    print(f"Quick visualization saved to {out_path}")
