import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data_utils import generate_sim_data, load_real_data, to_xyq_and_u
from .model import ElasticPINN
from .plotting import compute_metrics, plot_quick_result


def _load_data(args):
    if args.data == "real":
        try:
            return load_real_data(args.data_path)
        except Exception as e:
            print(f"[warn] failed to load real data: {e}; fallback to simul")
    return generate_sim_data(n_samples=args.n_samples, seed=args.seed)


def _build_loader(inputs, targets, batch_size, device):
    x = torch.tensor(inputs, dtype=torch.float32, device=device)
    y = torch.tensor(targets, dtype=torch.float32, device=device)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def train_mode(args):
    data = _load_data(args)
    inputs, targets = to_xyq_and_u(data)
    loader = _build_loader(inputs, targets, args.batch_size, args.device)

    model = ElasticPINN(hidden=args.hidden, depth=args.depth).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    hist = []
    for ep in range(args.epochs):
        total = 0.0
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += loss.item()
        avg = total / max(len(loader), 1)
        hist.append(avg)
        if ep % max(args.epochs // 10, 1) == 0:
            print(f"epoch {ep}/{args.epochs} loss={avg:.6e}")

    os.makedirs(args.model_dir, exist_ok=True)
    ckpt = os.path.join(args.model_dir, "elastic_basis_model.pt")
    torch.save(model.state_dict(), ckpt)

    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, "train_log.txt"), "w", encoding="utf-8") as f:
        for v in hist:
            f.write(f"{v:.8e}\n")
    print(f"[train] model saved: {ckpt}")


def quick_mode(args):
    data = _load_data(args)
    inputs, targets = to_xyq_and_u(data)

    model = ElasticPINN(hidden=args.hidden, depth=args.depth).to(args.device)
    ckpt = os.path.join(args.model_dir, "elastic_basis_model.pt")

    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=args.device))
        print(f"[quick] loaded model: {ckpt}")
    else:
        print("[quick] model missing, warmup training...")
        warm = args.quick_epochs
        loader = _build_loader(inputs, targets, args.batch_size, args.device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()
        for _ in range(warm):
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
        os.makedirs(args.model_dir, exist_ok=True)
        torch.save(model.state_dict(), ckpt)

    model.eval()
    with torch.no_grad():
        x = torch.tensor(inputs, dtype=torch.float32, device=args.device)
        pred = model(x).cpu().numpy()

    metrics = compute_metrics(targets, pred)
    os.makedirs(args.results_dir, exist_ok=True)

    # plot ux for a fixed q and y slice (first 256 points sorted by x)
    ux_true = targets[:, 0]
    ux_pred = pred[:, 0]
    x_axis = inputs[:, 0]
    idx = np.argsort(x_axis)[: min(256, len(x_axis))]
    plot_quick_result(
        x_axis[idx],
        ux_true[idx],
        ux_pred[idx],
        os.path.join(args.results_dir, "quick_result.png"),
    )

    with open(os.path.join(args.results_dir, "quick_metrics.txt"), "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}={v:.8e}\n")

    print(f"[quick] mae={metrics['mae']:.3e}, rmse={metrics['rmse']:.3e}")