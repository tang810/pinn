import torch
import matplotlib.pyplot as plt
import time

def train_model(model, dataset, n_iter=10000, lr=1e-3, lr_decay_step=2000, lr_gamma=0.95, 
                output_interval=100, intermediate_dir="intermediate", save_intermediate=True):
    """
    训练模型 - 支持中间结果输出和监控
    
    参数:
        model: PINN模型
        dataset: 数据集
        n_iter: 训练迭代次数 (对应epochs)
        lr: 学习率
        lr_decay_step: 学习率衰减步长
        lr_gamma: 学习率衰减因子
        output_interval: 中间输出间隔（默认100）
        intermediate_dir: 中间结果保存目录
        save_intermediate: 是否保存中间结果
    """
    print("开始训练模型 (固定 A=4)...")
    
    # 初始化优化器和调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_gamma)
    
    # 记录相关变量
    loss_record = []
    start_time = time.time()
    
    # 训练循环
    for epoch in range(1, n_iter + 1):
        optimizer.zero_grad()
        
        # 获取批次数据并计算损失
        model.train()
        batch = dataset[0]
        loss, l1, lx, ly, lz = model(batch)
        
        # 记录总损失
        loss_record.append(loss.item())
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # 每output_interval轮输出详细信息
        if epoch % output_interval == 0 or epoch == 1:
            print(f"Epoch {epoch}: Physics={l1.item():.3e}, BoundX={lx.item():.3e}, BoundY={ly.item():.3e}, BoundZ={lz.item():.3e}, Total={loss.item():.3e}")
    
    # 计算总训练时间
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds.")
    
    # 将loss记录存储到模型中（为了兼容性）
    model.loss_log = loss_record
    
    return loss_record


def plot_training_loss(loss_record, save_path=None, title="PINN Training Loss"):
    """
    绘制训练损失曲线
    
    参数:
        loss_record: 损失记录列表
        save_path: 保存路径
        title: 图表标题
    """
    plt.figure(figsize=(10, 6))
    
    # 创建epoch列表 (从1开始)
    epochs = list(range(1, len(loss_record) + 1))
    
    # 绘制损失曲线
    plt.plot(epochs, loss_record, 'b-', linewidth=2)
    
    # 设置标签和标题
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 如果loss值跨度很大，使用对数刻度
    if len(loss_record) > 0 and max(loss_record) / min(loss_record) > 100:
        plt.yscale('log')
        plt.ylabel('Loss (log scale)', fontsize=12)
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"损失曲线已保存到: {save_path}")
    
    # 显示图表
    plt.show()
    
    # 打印统计信息
    if len(loss_record) > 0:
        print(f"📈 Training completed: {len(loss_record)} epochs")
        print(f"   - Initial Loss: {loss_record[0]:.3e}")
        print(f"   - Final Loss: {loss_record[-1]:.3e}")
        print(f"   - Min Loss: {min(loss_record):.3e}")
        if len(loss_record) > 1:
            reduction = (loss_record[0] - loss_record[-1]) / loss_record[0] * 100
            print(f"   - Loss reduction: {reduction:.2f}%")