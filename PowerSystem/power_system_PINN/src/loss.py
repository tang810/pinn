import torch

def compute_loss(model, t, x, num_nodes=10):
    """Calculate loss function following three physical constraints of power systems"""
    # 获取设备信息（从模型参数或输入张量中获取）
    device = next(model.parameters()).device  # 从模型参数获取设备
    
    delta_f, delta_P_e, delta_P_m, df_dt, delta_theta = model.compute_derivatives(t, x)
    
    # 1. 发电机转子运动方程损失
    eq1_loss = torch.mean(torch.square(
        model.M * df_dt - (delta_P_m - delta_P_e)
    ))
    
    # 2. 功率平衡约束损失（修复形状不匹配问题）
    eq2_loss = torch.tensor(0.0, device=device)  # 现在device已定义
    batch_size = t.shape[0]

    for i in range(batch_size):
        node = x[i]
        if 0 < node < num_nodes - 1:  # 非边界节点
            prev_mask = (x == (node - 1)) & (torch.abs(t - t[i]) < 1e-6)
            next_mask = (x == (node + 1)) & (torch.abs(t - t[i]) < 1e-6)

            if torch.any(prev_mask) and torch.any(next_mask):
                # 计算相邻节点功率传输
                delta_P_prev = (delta_theta[i] - delta_theta[prev_mask].mean()) / model.X_line
                delta_P_next = (delta_theta[next_mask].mean() - delta_theta[i]) / model.X_line
                
                # 确保power_balance是标量
                power_balance = (delta_P_e[i] - (delta_P_next + delta_P_prev)).item()
                eq2_loss += torch.square(torch.tensor(power_balance, device=device))
    
    eq2_loss = eq2_loss / batch_size if batch_size > 0 else eq2_loss
    
    # 3. 相邻节点有功功率传输方程损失
    eq3_loss = torch.tensor(0.0, device=device)
    for i in range(batch_size):
        node = x[i]
        if node < num_nodes - 1:  # 非最后一个节点
            neighbor_mask = (x == (node + 1)) & (torch.abs(t - t[i]) < 1e-6)
            if torch.any(neighbor_mask):
                delta_P_transfer = (delta_theta[i] - delta_theta[neighbor_mask].mean()) / model.X_line
                # 确保累加的是标量
                eq3_loss += torch.square(delta_P_transfer).item()
    
    eq3_loss = torch.tensor(eq3_loss, device=device) / batch_size if batch_size > 0 else eq3_loss
    
    # 边界条件损失
    boundary_loss = torch.tensor(0.0, device=device)
    initial_mask = (t < 0.01)  # 初始时刻约束
    if torch.any(initial_mask):
        boundary_loss += torch.mean(torch.square(delta_f[initial_mask]))
    
    disturbance_mask = (t > 0.09) & (t < 0.11) & (x < 0.1)  # 扰动条件
    if torch.any(disturbance_mask):
        boundary_loss += torch.mean(torch.square(delta_P_m[disturbance_mask] - 0.1))
    
    end_mask = (x > num_nodes - 1.1) & (x < num_nodes - 0.9)  # 末端节点约束
    if torch.any(end_mask):
        df_dx = torch.autograd.grad(
            delta_f, x,
            grad_outputs=torch.ones_like(delta_f),
            create_graph=True,
            retain_graph=True
        )[0]
        boundary_loss += torch.mean(torch.square(df_dx[end_mask]))
    
    # 总损失
    total_loss = (
            1.0 * eq1_loss +
            1.0 * eq2_loss +
            1.0 * eq3_loss +
            2.0 * boundary_loss
    )
    
    return total_loss, eq1_loss, eq2_loss, eq3_loss, boundary_loss
    