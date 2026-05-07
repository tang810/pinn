
# ================================
# File: src/losses.py
# ================================
import torch
import torch.nn as nn

# ====================
# Loss 构建（数据项）
# ====================
def data_loss(pred_wtlt_norm, true_wtlt_norm):
    """
    数据项损失函数（监督回归损失）
    """
    return nn.MSELoss()(pred_wtlt_norm, true_wtlt_norm)

def pde_loss(pred_wtlt_norm, true_wtlt_norm, true_phys,
              inverse_func, scalers, constants,
              omega_batch, y_batch, phi, device='cpu'):
    """
    PDE残差损失（根据真实物理参数和预测物理参数推导的 R1/R2 差值）
    """
    B = pred_wtlt_norm.shape[0]

    # 获取预测物理参数（dict -> Tensor）
    with torch.no_grad():
        pred_wt = pred_wtlt_norm[:, 0].detach().cpu().numpy()
        pred_lt = pred_wtlt_norm[:, 1].detach().cpu().numpy()
        pred_phys_dict = inverse_func(pred_wt, pred_lt, scalers, constants)

    # 解构真实值
    true_M    = true_phys[:, 0]
    true_dkt  = true_phys[:, 1]
    true_dk3t = true_phys[:, 2]
    true_c    = true_phys[:, 3]

    # 转为 tensor（预测值）
    pred_M    = torch.tensor(pred_phys_dict['M_norm'],    dtype=torch.float32, device=device)
    pred_dkt  = torch.tensor(pred_phys_dict['dkt_norm'],  dtype=torch.float32, device=device)
    pred_dk3t = torch.tensor(pred_phys_dict['dk3t_norm'], dtype=torch.float32, device=device)
    pred_c    = torch.tensor(pred_phys_dict['c_norm'],    dtype=torch.float32, device=device)

    loss_R1_list = []
    loss_R2_list = []

    for i in range(B):
        omega_i = torch.tensor(omega_batch[i], dtype=torch.float32, device=device)
        y_i     = torch.tensor(y_batch[i],     dtype=torch.float32, device=device)
        phi_i   = torch.tensor(phi[i],         dtype=torch.float32, device=device)

        # F1_true - F1_pred
        w2y_true = -true_M[i] * omega_i**2 * y_i
        w2y_pred = -pred_M[i] * omega_i**2 * y_i

        lin_y_true = true_dkt[i] * y_i
        lin_y_pred = pred_dkt[i] * y_i

        nonlin_y_true = true_dk3t[i] * (y_i**3) * 3.0 / 4.0
        nonlin_y_pred = pred_dk3t[i] * (y_i**3) * 3.0 / 4.0

        drive_cos = torch.cos(phi_i)
        drive_sin = torch.sin(phi_i)

        F1_true = w2y_true + lin_y_true + nonlin_y_true - drive_cos
        F1_pred = w2y_pred + lin_y_pred + nonlin_y_pred - drive_cos

        F2_true = true_c[i] * omega_i * y_i - drive_sin
        F2_pred = pred_c[i] * omega_i * y_i - drive_sin

        R1 = F1_true - F1_pred
        R2 = F2_true - F2_pred

        loss_R1_list.append(torch.mean(R1**2))
        loss_R2_list.append(torch.mean(R2**2))

    return 0.5 * (torch.stack(loss_R1_list).mean() + torch.stack(loss_R2_list).mean())

# ====================
# Loss 构建（总损失）
# ====================
def pinn_loss(pred_wtlt_norm, true_wtlt_norm, true_phys_norm, inverse_func, scalers, constants,
                             omega_batch, y_batch, phi, device,lambda_phys=5.0):
    loss_data = data_loss(pred_wtlt_norm, true_wtlt_norm)
    loss_phys = lambda_phys*pde_loss(pred_wtlt_norm,  true_wtlt_norm,true_phys_norm, inverse_func, scalers, constants,omega_batch, y_batch, phi, device)
    loss_total = loss_data + loss_phys
    return loss_total, loss_data.item(), loss_phys.item()