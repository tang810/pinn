"""
简化版数据管理器
只保留模型保存和加载功能
"""

import torch
import numpy as np
import os
import json
from datetime import datetime


class ModelManager:
    """模型管理器 - 只负责保存和加载模型"""
    
    def __init__(self, base_dir='model'):
        self.base_dir = base_dir
        # 确保目录存在
        os.makedirs(f'{base_dir}/best_models', exist_ok=True)

    def _load_checkpoint(self, filepath, device):
        try:
            return torch.load(filepath, map_location=device, weights_only=True)
        except TypeError:
            return torch.load(filepath, map_location=device)
    
    def save_best_model(self, model, eps, r, metrics=None, device='cpu'):
        """保存最佳模型"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'{self.base_dir}/best_models/best_model_{timestamp}.pt'
        
        # 确保eps和r是标量值
        eps_val = eps.item() if hasattr(eps, 'item') else float(eps)
        r_val = r.item() if hasattr(r, 'item') else float(r)
        
        save_dict = {
            'model_state_dict': model.state_dict(),
            'eps': eps_val,
            'r': r_val,
            'metrics': metrics or {},
            'timestamp': timestamp,
            'device': str(device)
        }
        
        torch.save(save_dict, filename)
        print(f"模型已保存: {filename}")
        print(f"   参数: ε={eps_val:.4f}, r={r_val:.4f}")
        return filename
    
    def load_best_model(self, filename, model, device='cpu'):
        """加载最佳模型"""
        filepath = os.path.join(self.base_dir, "best_models", filename)
        if os.path.exists(filepath):
            try:
                # 加载到指定设备
                save_dict = self._load_checkpoint(filepath, device)
                
                # 加载模型状态
                model.load_state_dict(save_dict['model_state_dict'])
                model.to(device)
                
                print(f"模型加载成功: {filename}")
                print(f"   参数: ε={save_dict['eps']:.4f}, r={save_dict['r']:.4f}")
                
                return save_dict
            except Exception as e:
                print(f"模型加载失败: {e}")
                return None
        else:
            print(f"模型文件不存在: {filepath}")
            return None
