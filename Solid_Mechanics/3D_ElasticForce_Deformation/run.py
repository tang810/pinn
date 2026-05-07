from scripts.utils import load_data, load_test_data, save_model, load_model
from scripts.MultiFNN import MultiFNN
from scripts.config import device
from scripts.visualization import plot_displacement, plot_stress, loss_curve, evaluate_physics_metrics
from scripts.train import train_model


def run():
    """主函数：加载数据、训练模型并可视化结果"""
    # 加载数据
    dataset = load_data()
    
    # 初始化模型
    print("初始化模型...")
    layers = [3, 20, 20, 20, 20, 1]
    model = MultiFNN(layers).to(device)

    # 询问是否训练模型
    train_flag = input("是否训练模型? (y/n): ").lower()
    if train_flag == 'y':
        print("开始训练模型...")
        # 训练模型
        train_model(model, dataset, n_iter=8000)
        # 询问是否保存模型
        save_flag = input("是否保存模型? (y/n): ").lower()
        if save_flag == 'y':
            save_model(model, './model/pinn_elasticity_model.pt')
    elif train_flag == 'n':
        print("加载已有模型...")
        # 询问是否加载已有模型
        load_model(model, './model/pinn_elasticity_model.pt', device=device)
    else:
        print("退出程序")
        return
    
    # 加载测试数据
    test_x = load_test_data()
    print("开始预测...")
    # 预测位移
    plot_displacement(test_x, model, batch_size=dataset.batch_size)
    # 计算应力
    print("计算应力...")
    plot_stress(test_x, model, batch_size=dataset.batch_size)
    #绘制损失曲线
    loss_curve(model)
    #物理损失
    evaluate_physics_metrics(model)
# 执行主函数
if __name__ == "__main__":
    run()
