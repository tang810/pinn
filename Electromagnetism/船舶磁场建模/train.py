import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
from torch.utils.data import TensorDataset, random_split, DataLoader
from tqdm import tqdm

from ReadData import ReadData
from MagneticCalculate import forward_problem
from net import DNN

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x


def Loss_pde(model, H_batch, ship_batch):
    out = model(H_batch)
    L, W, θ = ship_batch[:,2], ship_batch[:,3], ship_batch[:,4]
    x, y = ship_batch[:,0], ship_batch[:,1]
    z = H_batch[:,3]
    m_pred = out[:,5:]
    H_pred = forward_problem(x, y, z, m_pred, theta=θ, Ln=L, Wn=W)
    Hx_p, Hy_p, Hz_p = H_pred[:,0:1], H_pred[:,1:2], H_pred[:,2:3]
    Hx, Hy, Hz    = H_batch[:,0:1],    H_batch[:,1:2],    H_batch[:,2:3]
    return ((Hx - Hx_p)**2).mean() + ((Hy - Hy_p)**2).mean() + ((Hz - Hz_p)**2).mean()


def Loss_data(model, H_batch, ship_batch):
    return nn.MSELoss()(model(H_batch), ship_batch)


def test(model, H_norm, ship_norm, H_mean, H_std, ship_mean, ship_std):
    model.eval()
    with torch.no_grad():
        out_norm = model(H_norm.to(device))
    # 反归一化，避免除以零
    ship_std_safe = ship_std.clone()
    ship_std_safe[ship_std_safe == 0] = 1.0
    out = out_norm.cpu() * ship_std_safe + ship_mean
    true = ship_norm.cpu() * ship_std_safe + ship_mean
    err = to_numpy(out - true)
    denom = np.linalg.norm(to_numpy(true), axis=1)
    denom[denom == 0] = 1.0
    l2 = np.linalg.norm(err, axis=1) / denom
    return l2


if __name__ == "__main__":
    start = time.time()
    df = ReadData('./data')
    Hx, Hy, Hz = df["Hx"].values, df["Hy"].values, df["Hz"].values
    z = df["z"].values
    H_np = np.stack([Hx, Hy, Hz, z], axis=1)
    ship_np = df.iloc[:, 4:].values

    # 转为张量
    H_t    = torch.tensor(H_np,    dtype=torch.float64)
    ship_t = torch.tensor(ship_np, dtype=torch.float64)

    # 归一化，加 eps 防止 std=0
    eps = 1e-6
    H_mean,   H_std    = H_t.mean(0),   H_t.std(0)   + eps
    ship_mean,ship_std = ship_t.mean(0), ship_t.std(0) + eps
    H_norm    = (H_t - H_mean) / H_std
    ship_norm = (ship_t - ship_mean) / ship_std

    ds = TensorDataset(H_norm, ship_norm)
    n_train = int(0.8 * len(ds))
    n_test  = len(ds) - n_train
    train_ds, test_ds = random_split(ds, [n_train, n_test])
    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
    H_test, ship_test = map(torch.stack, zip(*list(test_ds)))

    # 模型与优化器
    layers = [4, 64, 128, 512, 512, 256, 53]
    model = DNN(layers).to(device)
    model._initialize_weights()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.5, patience=5)
    epochs = 800
    α, β = 1.0, 1.0
    save_dir = './Result'
    os.makedirs(save_dir, exist_ok=True)

    epoch_bar = tqdm(total=epochs, desc="Training", unit="epoch")
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0

        for H_batch, ship_batch in train_loader:
            H_batch, ship_batch = H_batch.to(device), ship_batch.to(device)
            optimizer.zero_grad()
            lp = Loss_pde(model, H_batch, ship_batch)
            ld = Loss_data(model, H_batch, ship_batch)
            loss = α * lp + β * ld

            # 检查 nan
            if torch.isnan(loss):
                print(f"NaN detected at epoch {epoch}, skipping backprop")
                continue

            loss.backward()
            nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        l2 = test(model, H_test, ship_test, H_mean, H_std, ship_mean, ship_std).mean()

        scheduler.step(avg_loss)
        epoch_bar.set_postfix({
            'avg_loss': f"{avg_loss:.4e}",
            'test_L2':  f"{l2:.4e}",
            'lr':       f"{optimizer.param_groups[0]['lr']:.1e}"
        })
        epoch_bar.update(1)
        if epoch % 50 == 0:
            # 保存模型
            torch.save(model.state_dict(), f"{save_dir}/epoch{epoch:03d}.pth")
            print(f"Model saved at epoch {epoch}")

    epoch_bar.close()
    print(f"Training completed in {time.time()-start:.2f}s")
