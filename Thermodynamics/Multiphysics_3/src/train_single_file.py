"""
Notebook-friendly train file for Multiphysics_3 (PINNsformer inverse).

Online Jupyter usage:
1. Upload data_7246.txt as a public asset.
2. Fill DATA_HASH and MODEL_HASH.
3. Run this whole file/cell. It saves trained_model_multiphysics3.pt to
   ~/minio/Resource/{MODEL_HASH}/ for later public-asset upload.

Physics: PINNsformer for fully-coupled thermo-elastic inverse problem.
         Input (x,y,z,t) → Output (T, u, v, w).
         Learns unknown physical parameters (e.g. thermal conductivity k).

Note: imports from src/ for complex physics and data pipelines.
"""

import os
import time

import torch
import numpy as np
import matplotlib.pyplot as plt

from src import train as train_mod
from src import config



def get_device():
    """Detect device in priority: GPU (cuda) -> NPU (npu/ascend) -> CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_npu
        if torch_npu.npu.is_available():
            return torch.device("npu")
    except (ImportError, AttributeError):
        pass
    return torch.device("cpu")

# =========================
# 1. Parameters to edit
# =========================

DATA_HASH = "eed4f868ae734343867928fed865eb6b"
MODEL_HASH = "eed4f868ae734343867928fed865eb6b"

DATA_PATH = os.path.expanduser(f"~/public/Resource/{DATA_HASH}")
MODEL_PATH = os.path.expanduser(f"~/minio/Resource/{MODEL_HASH}/trained_model_multiphysics3.pt")


# =========================
# 2. Train entry
# =========================

class Args:
    pass


def train(data_path=DATA_PATH, model_path=MODEL_PATH, epochs=200, lr=1e-4):
    t0 = time.time()
    device = get_device()
    print(f"device={device}")

    args = Args()
    for k, v in config.__dict__.items():
        if k.isupper() and not k.startswith("_"):
            setattr(args, k.lower(), v)
    data_path = os.path.expanduser(data_path) if data_path else config.data_path
    model_path = os.path.expanduser(model_path)
    args.data_path = data_path
    args.model_save_path = model_path
    args.epochs = epochs
    args.learning_rate = lr
    args.device = str(device)
    args.inverse_mode = True
    args.infer_params = ['k']
    args.resume_epoch = 0
    args.pretrained_model_path = None
    args.batch_size = getattr(args, 'batch_size', 200)
    args.data_batch_size = getattr(args, 'data_batch_size', 400)
    args.time_steps = getattr(args, 'time_steps', 5)
    args.step_size = getattr(args, 'step_size', 2e-2)
    args.sampler = getattr(args, 'sampler', 'lhs')

    print(f"data={data_path}"); print(f"model={model_path}")
    print(f"epochs={epochs}, inverse_mode={args.inverse_mode}, infer={args.infer_params}")

    train_mod.run(args)

    print("Training finished"); print(f"elapsed: {time.time() - t0:.1f}s")
    return {"model_path": model_path}


train_result = train()
