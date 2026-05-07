import os

import numpy as np
import torch
import torch.optim as optim

from .data_loader import load_dataset
from .model import MagneticMomentNet
from .physics import magnetic_field_loss
from .utils import select_device, setup_seed


def train_model(args):
    setup_seed(args.seed)
    device = select_device(args.device)
    print(f"Using device: {device}")

    dataset = load_dataset(args.mat_path, device)
    model_inputs = dataset["model_inputs"]
    print(f"inputs.shape: {model_inputs.shape}")

    model = MagneticMomentNet(
        input_size=args.input_size,
        output_size=args.output_size,
        hidden_dims=args.hidden_dims,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    history = []

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        predicted_moments = model(model_inputs)
        history.append(predicted_moments.detach().cpu().numpy())

        loss = magnetic_field_loss(predicted_moments, model_inputs)
        loss.backward()
        optimizer.step()

        if epoch % args.log_every == 0:
            print(f"Epoch [{epoch + 1}], Loss: {loss.item():.6f}")

    history_path = os.path.join(args.output_dir, "magnetic_moments_history.npy")
    np.save(history_path, np.array(history))
    print(f"Saved history to {history_path}")

    ckpt_path = os.path.join(args.model_dir, "final_model.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": args.epochs - 1,
            "loss": float(loss.item()),
            "history_path": history_path,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint to {ckpt_path}")
