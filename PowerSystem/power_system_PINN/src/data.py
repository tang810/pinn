import torch

def generate_training_data(num_samples=10000, num_nodes=10, t_max=1.0):
    """Generate training data points"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t = torch.rand(num_samples) * t_max
    x = torch.randint(0, num_nodes, (num_samples,), dtype=torch.float32)
    return t.to(device), x.to(device)

def generate_test_data(num_nodes=10, t_max=10.0, num_time_steps=200):
    """Generate test data grid"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t = torch.linspace(0, t_max, num_time_steps, device=device)
    x = torch.linspace(0, num_nodes - 1, num_nodes, device=device)
    t_grid, x_grid = torch.meshgrid(t, x, indexing='ij')
    t_flat = t_grid.flatten()
    x_flat = x_grid.flatten()
    return t, x, t_flat, x_flat, t_grid.cpu().detach().numpy(), x_grid.cpu().detach().numpy()


def save_model(model, save_path):
    """
    保存模型权重到指定路径
    :param model: 训练好的 PINN 模型
    :param save_path: 保存路径，如 "../model/pinn_model.pt"
    """
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存到: {save_path}")