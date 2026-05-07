# src/train_utils.py（新增）
import torch

def save_model(model, save_path):
    """
    保存模型权重到指定路径
    :param model: 训练好的 PINN 模型
    :param save_path: 保存路径，如 "../model/pinn_model.pt"
    """
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存到: {save_path}")