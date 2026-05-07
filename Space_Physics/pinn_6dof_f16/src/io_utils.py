from __future__ import annotations
from typing import Optional

import os
import torch

def save_model(net: torch.nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": net.state_dict()}, path)

def load_model(net: torch.nn.Module, path: str, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(path, map_location=device)
    net.load_state_dict(ckpt["model"])
    net.to(device)
    net.eval()
    return net
