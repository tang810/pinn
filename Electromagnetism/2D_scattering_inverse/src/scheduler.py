class DynamicWeightScheduler:
    """
    动态权重策略 (改进版)
    """
    def __init__(self, total_epochs=3000):
        self.total_epochs = total_epochs
        
    def get_pde_weight(self, epoch):
        # 动态调整：初期强调数据拟合，后期强调物理约束
        if epoch < 1000:
            return 0.8
        elif epoch < 2000:
            return 1.2
        else:
            return 1.5
    
    def get_radius_weight(self, epoch):
        # 半径约束权重逐渐增加
        return min(0.3, epoch / 3000)