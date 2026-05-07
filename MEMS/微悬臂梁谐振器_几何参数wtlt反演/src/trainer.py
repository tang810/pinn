
# ================================
# File: src/trainer.py
# ================================
from typing import Dict, Any, Callable
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange

from .models import PINNInverseMLP
from .losses import pinn_loss
from .data_utils import build_dataloaders

# Train Loop with tqdm + Scheduler (+ 可选 EarlyStopping)
def train(model, dataloader, optimizer, inverse_func, scalers, constants,
          device='cpu', num_epochs=100,
          scheduler_patience=10, earlystop_patience=20):
    
    model.train()
    loss_history = {'total': [], 'data': [], 'pde': []}
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                  patience=scheduler_patience, verbose=True)

    best_loss = float('inf')
    wait = 0

    pbar = trange(1, num_epochs + 1, desc="Training", dynamic_ncols=True)

    for epoch in pbar:
        total_loss, total_data, total_phys = 0.0, 0.0, 0.0

        for batch in dataloader:
            X = batch['X'].to(device)
            Y_target = batch['Y_target'].to(device)
            Y_phys = batch['Y_phys'].to(device)
            phi = batch['phi']  # list of [T_i]
            omega_batch = batch['omega']
            y_batch = batch['y']

            optimizer.zero_grad(set_to_none=True)
            pred_wtlt = model(X)
            loss_total, loss_data, loss_phys = pinn_loss(pred_wtlt, Y_target, Y_phys, inverse_func, scalers, constants,
                              omega_batch, y_batch, phi, device)
            loss_total.backward()
            optimizer.step()

            total_loss += loss_total.item()
            total_data += loss_data
            total_phys += loss_phys

        avg_total = total_loss / len(dataloader)
        avg_data = total_data / len(dataloader)
        avg_phys = total_phys / len(dataloader)

        loss_history['total'].append(avg_total)
        loss_history['data'].append(avg_data)
        loss_history['pde'].append(avg_phys)

        # 更新进度条
        pbar.set_description(f"Epoch {epoch:03d} | Total: {avg_total:.4f} | Data: {avg_data:.4f} | PDE: {avg_phys:.4f}")

        # 学习率调度
        scheduler.step(avg_total)

        # 可选早停（保留接口，默认关闭）
        # if avg_total < best_loss - 1e-5:
        #     best_loss = avg_total
        #     wait = 0
        # else:
        #     wait += 1
        #     if wait >= earlystop_patience:
        #         print(f"\nEarly stopping triggered at epoch {epoch}.")
        #         break

    return loss_history

def run_training(X_feat, Y_target, Y_phys, phi, omega_norm_all, y_norm_all,
                 inverse_func: Callable, scalers: Dict[str, Any], constants: Dict[str, Any],
                 hidden_dim=128, hidden_layers=4, num_epochs=100, batch_size=32, lr=1e-3, device='cpu'):

    model = PINNInverseMLP(input_dim=8, output_dim=2, hidden_dim=hidden_dim, hidden_layers=hidden_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader, val_loader = build_dataloaders(
        X_feat, Y_target, Y_phys, phi, omega_norm_all, y_norm_all, batch_size=batch_size
    )

    print("Starting training...")
    loss_history = train(model, train_loader, optimizer, inverse_func, scalers, constants, device=device, num_epochs=num_epochs)
    print("Training complete.")

    return model, train_loader, val_loader, loss_history
