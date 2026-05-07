import torch.optim as optim
from tqdm import tqdm
from .config import n_iter, lr

def train_model(model, dataset, n_iter=n_iter, lr=lr):
    """训练模型"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    pbar = tqdm(range(n_iter))
    for it in pbar:
    # 直接使用数据集
        batch = dataset[0]
        # 前向传播
        l1, lx, ly, lz, loss = model(batch)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 每2000步更新学习率
        if it % 2000 == 0 and it > 0:
            scheduler.step()  
        # 记录损失
        if it % 10 == 0:
            model.loss_log.append(loss.item())
            model.loss_res_log.append(l1.item())  
            model.loss_x_log.append(lx.item())
            model.loss_y_log.append(ly.item())
            model.loss_z_log.append(lz.item())
                     
            # 更新进度条
            pbar.set_postfix({
                'Loss': f"{loss.item():.2f}",
                'l1': f"{l1.item():.2e}",
                'lx': f"{lx.item():.2f}",
                'ly': f"{ly.item():.2f}",
                'lz': f"{lz.item():.2f}"
            })