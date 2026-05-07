import os

import torch

from .data_loader import create_visualization_grid
from .model import FNN
from .physics import transform_inverse
from .utils import plot_complex_field, resolve_checkpoint, select_device


def run(args):
    device = select_device(args.device)
    print(f"Using device: {device}")

    ckpt_path = resolve_checkpoint(args)
    if not ckpt_path:
        raise FileNotFoundError(
            "Checkpoint not found. Please run train first, or pass --ckpt explicitly."
        )

    ckpt = torch.load(ckpt_path, map_location=device)
    task = ckpt.get("task", args.task)

    net = FNN(num_layers=args.num_layers, num_nodes=args.hidden_dim).to(device)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    vis_points, xx, yy = create_visualization_grid(args.length, args.dpml, args.viz_resolution, device)
    with torch.no_grad():
        outputs = net(vis_points)
        radius = None
        if task == "inverse":
            radius = float(ckpt.get("R_param", args.target_r_init))
            outputs = transform_inverse(vis_points, outputs, radius)

        re_field = outputs[:, 0].reshape(xx.shape).detach().cpu().numpy()
        im_field = outputs[:, 1].reshape(xx.shape).detach().cpu().numpy()

    file_name = "final_Re_Im.png" if task == "inverse" else "pytorchES.png"
    save_path = os.path.join(args.output_dir, file_name)
    plot_complex_field(
        xx,
        yy,
        re_field,
        im_field,
        save_path,
        "Real Part of Electric Field",
        "Imaginary Part of Electric Field",
        radius=radius,
    )

    print(f"Quick visualization saved to {save_path}")
    if task == "inverse":
        print(f"Recovered params: R = {ckpt.get('R_param', 'N/A')}, eps = {ckpt.get('eps_param', 'N/A')}")
