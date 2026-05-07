from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .losses import physics_residual_loss
from .model import AeroNet

@dataclass
class TrainOutputs:
    model: torch.nn.Module
    loss_hist: np.ndarray

def train_pinn(X_t: torch.Tensor, tensors: Dict[str, torch.Tensor],
               device: torch.device,
               epochs: int = 5000,
               lr: float = 1e-3,
               reg_weight: float = 1e-5,
               ckpt_path: Optional[str] = None,
               resume: bool = True,
               verbose_every: int = 500) -> TrainOutputs:
    """训练 PINN，并可选保存/恢复权重。"""
    net = AeroNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    start_ep = 0
    if resume and ckpt_path:
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            net.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["optimizer"])
            start_ep = int(ckpt.get("epoch", 0))
        except FileNotFoundError:
            pass

    loss_hist = []

    for ep in range(start_ep, epochs):
        opt.zero_grad()

        out = physics_residual_loss(
            net, X_t,
            tensors["u"], tensors["v"], tensors["w"],
            tensors["p"], tensors["q"], tensors["r"],
            tensors["phi"], tensors["theta"],
            tensors["u_dot"], tensors["v_dot"], tensors["w_dot"],
            tensors["p_dot"], tensors["q_dot"], tensors["r_dot"],
            reg_weight=reg_weight
        )
        out.total.backward()
        opt.step()

        loss_hist.append(float(out.total.detach().cpu().item()))

        if verbose_every and ((ep + 1) % verbose_every == 0 or ep == start_ep):
            print(f"Epoch {ep+1}/{epochs}: total={loss_hist[-1]:.3e}, "
                  f"trans={float(out.loss_trans.detach().cpu()):.3e}, "
                  f"rot={float(out.loss_rot.detach().cpu()):.3e}")

        if ckpt_path and ((ep + 1) % max(1, verbose_every) == 0 or ep == epochs - 1):
            torch.save(
                {"epoch": ep + 1, "model": net.state_dict(), "optimizer": opt.state_dict()},
                ckpt_path
            )

    return TrainOutputs(model=net, loss_hist=np.asarray(loss_hist, dtype=float))
