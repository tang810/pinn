import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models import KdV_PINN
from src.utils import plot_loss_curve


def build_dataloader(data, device="cpu", batch_size=4):
    x = torch.tensor(data["x"], dtype=torch.float32).to(device).reshape(-1, 1)
    y = torch.tensor(data["ref"], dtype=torch.float32).to(device).reshape(-1, 1)
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def run_pinn_training(model, dataloader, device, model_dir, epochs=10, lr=1e-4):
    import torch.nn as nn
    import torch.optim as optim

    os.makedirs(model_dir, exist_ok=True)
    model_file = os.path.join(model_dir, "kdv_pinn_model.pt")

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    all_losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for x, ref in dataloader:
            x, ref = x.to(device), ref.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, ref)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / max(len(dataloader), 1)
        all_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), model_file)
    print(f"model saved: {model_file}")
    return {"loss_history": all_losses}


def run_quick_inference(data, device="cpu", model_dir="model", quick_epochs=30):
    os.makedirs(model_dir, exist_ok=True)
    model_file = os.path.join(model_dir, "kdv_pinn_model.pt")

    model = KdV_PINN(n_input=1, n_output=1, n_hidden=50, n_layers=3).to(device)

    x = torch.tensor(data["x"], dtype=torch.float32).to(device)
    n_samples, n_x = x.shape
    x_flat = x.reshape(-1, 1)

    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file, map_location=device))
    elif quick_epochs > 0:
        # lightweight warmup to keep quick mode runnable even without checkpoint
        dl = build_dataloader(data, device=device, batch_size=8)
        run_pinn_training(model, dl, device, model_dir, epochs=quick_epochs, lr=1e-3)

    model.eval()
    with torch.no_grad():
        pred_flat = model(x_flat)
    pred = pred_flat.cpu().numpy().reshape(n_samples, n_x)
    return pred


def export_training_results(log, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    loss_history = log.get("loss_history", []) if isinstance(log, dict) else []
    with open(os.path.join(output_dir, "train_log.txt"), "w", encoding="utf-8") as f:
        for l in loss_history:
            f.write(f"{l}\n")
    if loss_history:
        plot_loss_curve(loss_history, output_dir=output_dir)


def export_all_results(pred, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    np.savetxt(os.path.join(output_dir, "pred_results.txt"), pred)