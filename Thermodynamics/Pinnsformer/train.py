import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from torch.optim import LBFGS, Adam
from tqdm import tqdm
import os
from scripts.utils import *
from model.pinn import PINNs
from scripts.Pinnsformer import PINNsformer
from tqdm import tqdm

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
LOAD_MODEL = True  # True 表示加载已有模型
MODEL_PATH = './model/3d_pinnsformer.pt'
SAVE_VIZ_PATH = './viz/animation.gif'
seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
step_size = 1e-4
filename = './data/data_rbc.txt'
x_star, y_star, z_star, t_star, T_star = loading_evaluate_data(filename)
N = x_star.shape[0]
idx = np.random.choice(N, 5, replace=False)
idx = [y for x in idx for y in
    [x // 41 * 41 + 1, x // 41 * 41 + 2, x // 41 * 41 + 3, x // 41 * 41 + 4, x // 41 * 41 + 5,
        x // 41 * 41 + 6, x // 41 * 41 + 7, x // 41 * 41 + 8, x // 41 * 41 + 9, x // 41 * 41 + 10, x // 41 * 41 + 11,
        x // 41 * 41 + 12, x // 41 * 41 + 13, x // 41 * 41 + 14, x // 41 * 41 + 15, x // 41 * 41 + 16,
        x // 41 * 41 + 17, x // 41 * 41 + 18, x // 41 * 41 + 19, x // 41 * 41 + 20, x // 41 * 41 + 21,
        x // 41 * 41 + 22, x // 41 * 41 + 23, x // 41 * 41 + 24, x // 41 * 41 + 25]]

x_train = x_star[idx, :]
y_train = y_star[idx, :]
t_train = t_star[idx, :]
z_train = z_star[idx, :]
T_train = T_star[idx, :]

x_star = np.expand_dims(np.tile(x_star[:], (5)), -1)
y_star = np.expand_dims(np.tile(y_star[:], (5)), -1)
z_star = np.expand_dims(np.tile(z_star[:], (5)), -1)
t_star = make_time_sequence(t_star, num_step=5, step=step_size)
x_star = torch.tensor(x_star, dtype=torch.float32, requires_grad=True).to(device)
y_star = torch.tensor(y_star, dtype=torch.float32, requires_grad=True).to(device)
z_star = torch.tensor(z_star, dtype=torch.float32, requires_grad=True).to(device)
t_star = torch.tensor(t_star, dtype=torch.float32, requires_grad=True).to(device)

x_train = np.expand_dims(np.tile(x_train[:], (5)), -1)
y_train = np.expand_dims(np.tile(y_train[:], (5)), -1)
z_train = np.expand_dims(np.tile(z_train[:], (5)), -1)
t_train = make_time_sequence(t_train, num_step=5, step=step_size)
x_train = torch.tensor(x_train, dtype=torch.float32, requires_grad=True).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32, requires_grad=True).to(device)
t_train = torch.tensor(t_train, dtype=torch.float32, requires_grad=True).to(device)
z_train = torch.tensor(z_train, dtype=torch.float32, requires_grad=True).to(device)
T_train = torch.tensor(T_train, dtype=torch.float32, requires_grad=True).to(device)

# Train PINNsformer
res, x_lb, x_ub, y_lb, y_ub, z_lb, z_ub, t_lb, t_ub = get_data([0, 1], [0, 1], [0, 1], [0, 0.4], 4,
                                                            4, 4, 41)

res = make_time_sequence(res, num_step=5, step=step_size)
x_lb = make_time_sequence(x_lb, num_step=5, step=step_size)
x_ub = make_time_sequence(x_ub, num_step=5, step=step_size)
y_lb = make_time_sequence(y_lb, num_step=5, step=step_size)
y_ub = make_time_sequence(y_ub, num_step=5, step=step_size)
z_lb = make_time_sequence(z_lb, num_step=5, step=step_size)
z_ub = make_time_sequence(z_ub, num_step=5, step=step_size)
t_lb = make_time_sequence(t_lb, num_step=5, step=step_size)
t_ub = make_time_sequence(t_ub, num_step=5, step=step_size)

res = torch.tensor(res, dtype=torch.float32, requires_grad=True).to(device)
x_lb = torch.tensor(x_lb, dtype=torch.float32, requires_grad=True).to(device)
x_ub = torch.tensor(x_ub, dtype=torch.float32, requires_grad=True).to(device)
y_lb = torch.tensor(y_lb, dtype=torch.float32, requires_grad=True).to(device)
y_ub = torch.tensor(y_ub, dtype=torch.float32, requires_grad=True).to(device)
z_lb = torch.tensor(z_lb, dtype=torch.float32, requires_grad=True).to(device)
z_ub = torch.tensor(z_ub, dtype=torch.float32, requires_grad=True).to(device)
t_lb = torch.tensor(t_lb, dtype=torch.float32, requires_grad=True).to(device)
t_ub = torch.tensor(t_ub, dtype=torch.float32, requires_grad=True).to(device)

x_res, y_res, z_res, t_res = res[:, :, 0:1], res[:, :, 1:2], res[:, :, 2:3], res[:, :, 3:4]
x_left, y_left, z_left, t_left = x_lb[:, :, 0:1], x_lb[:, :, 1:2], x_lb[:, :, 2:3], x_lb[:, :, 3:4]
x_right, y_right, z_right, t_right = x_ub[:, :, 0:1], x_ub[:, :, 1:2], x_ub[:, :, 2:3], x_ub[:, :, 3:4]
x_front, y_front, z_front, t_front = y_lb[:, :, 0:1], y_lb[:, :, 1:2], y_lb[:, :, 2:3], y_lb[:, :, 3:4]
x_back, y_back, z_back, t_back = y_ub[:, :, 0:1], y_ub[:, :, 1:2], y_ub[:, :, 2:3], y_ub[:, :, 3:4]
x_bottom, y_bottom, z_bottom, t_bottom = z_lb[:, :, 0:1], z_lb[:, :, 1:2], z_lb[:, :, 2:3], z_lb[:, :, 3:4]
x_top, y_top, z_top, t_top = z_ub[:, :, 0:1], z_ub[:, :, 1:2], z_ub[:, :, 2:3], z_ub[:, :, 3:4]
x_lb, y_lb, z_lb, t_lb = t_lb[:, :, 0:1], t_lb[:, :, 1:2], t_lb[:, :, 2:3], t_lb[:, :, 3:4]


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


model = PINNsformer(d_out=1, d_hidden=512, d_model=32, N=1, heads=2).to(device)

if LOAD_MODEL and os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"✅ Loaded model from {MODEL_PATH}")
else:
    model.apply(init_weights)
    print("🔁 Initialized new model.")
    n_params = get_n_params(model)


# ------------------ 训练函数 ------------------ #   
def train_model(model, optim, num_iters=1000):
    loss_track = []
    pbar = tqdm(range(num_iters))
    pi = torch.tensor(np.pi, dtype=torch.float32, requires_grad=False).to(device)

    for i in pbar:
        def closure():
            pred_res = model(x_res, y_res, z_res, t_res)

            pred_left = model(x_left, y_left, z_left, t_left)
            pred_right = model(x_right, y_right, z_right, t_right)
            pred_front = model(x_front, y_front, z_front, t_front)
            pred_back = model(x_back, y_back, z_back, t_back)
            pred_bottom = model(x_bottom, y_bottom, z_bottom, t_bottom)
            pred_top = model(x_top, y_top, z_top, t_top)
            pred_lb = model(x_lb, y_lb, z_lb, t_lb)

            pred_data = model(x_train, y_train, z_train, t_train)

            d_x = torch.autograd.grad(pred_data, x_train, grad_outputs=torch.ones_like(pred_data), retain_graph=True,
                                    create_graph=True)[0]

            u_x = torch.autograd.grad(pred_res, x_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True,
                                    create_graph=True)[0]

            u_y = torch.autograd.grad(pred_res, y_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True,
                                    create_graph=True)[0]
            u_z = torch.autograd.grad(pred_res, z_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True,
                                    create_graph=True)[0]

            u_xx = \
                torch.autograd.grad(u_x, x_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True,
                                    create_graph=True)[0]
            u_yy = \
                torch.autograd.grad(u_y, y_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True,
                                    create_graph=True)[0]
            u_zz = \
                torch.autograd.grad(u_z, z_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True,
                                    create_graph=True)[0]

            u_t = torch.autograd.grad(pred_res, t_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True,
                                    create_graph=True)[0]

            ub_x1 = torch.autograd.grad(pred_right, x_right, grad_outputs=torch.ones_like(pred_right), retain_graph=True,
                                        create_graph=True)[0]
            ub_x0 = torch.autograd.grad(pred_left, x_left, grad_outputs=torch.ones_like(pred_left), retain_graph=True,
                                        create_graph=True)[0]
            ub_y0 = torch.autograd.grad(pred_front, y_front, grad_outputs=torch.ones_like(pred_front), retain_graph=True,
                                        create_graph=True)[0]
            ub_y1 = torch.autograd.grad(pred_back, y_back, grad_outputs=torch.ones_like(pred_back), retain_graph=True,
                                        create_graph=True)[0]
            ub_z0 = torch.autograd.grad(pred_bottom, z_bottom, grad_outputs=torch.ones_like(pred_bottom), retain_graph=True,
                                        create_graph=True)[0]
            ub_z1 = torch.autograd.grad(pred_top, z_top, grad_outputs=torch.ones_like(pred_top), retain_graph=True,
                                        create_graph=True)[0]

            k = 167  # 热导率系数
            rho = 1
            C_p = 1
            eps = 0.1
            simga = 5.67e-8
            T_amb = 3  # 环境温度

            loss_res = torch.mean((u_t - k / (rho * C_p) * (u_xx + u_yy + u_zz)) ** 2)

            loss_bc1 = torch.mean((-k * ub_x1 - 1361) ** 2)  # 太阳热源条件
            loss_bc2 = torch.mean((eps * simga * (T_amb ** 4 - pred_left ** 4) + k * ub_x0) ** 2)
            loss_bc3 = torch.mean((eps * simga * (T_amb ** 4 - pred_front ** 4) + k * ub_y0) ** 2)
            loss_bc4 = torch.mean((eps * simga * (T_amb ** 4 - pred_back ** 4) - k * ub_y1) ** 2)
            loss_bc5 = torch.mean((eps * simga * (T_amb ** 4 - pred_bottom ** 4) + k * ub_z0) ** 2)
            loss_bc6 = torch.mean((eps * simga * (T_amb ** 4 - pred_top ** 4) - k * ub_z1) ** 2)
            loss_bc7 = torch.mean((eps * simga * (T_amb ** 4 - pred_right ** 4) - k * ub_x1) ** 2)

            # loss_bc = loss_bc2 + loss_bc3 + loss_bc4 + loss_bc5 + loss_bc6

            loss_bc = loss_bc2 + loss_bc3 + loss_bc4 + loss_bc5 + loss_bc6 + loss_bc7 # 该数据集不考虑太阳热源 只仿真了辐射条件

            loss_ic = torch.mean((pred_lb[:, 0] - 273) ** 2)

            loss_data = torch.mean((pred_data[:, 0] - T_train) ** 2)

            loss_track.append([loss_res.item(), loss_bc.item(), loss_ic.item(), loss_data.item()])

            loss = loss_res + loss_bc + loss_ic + 10 * loss_data # 去掉 loss_data 即无监督
            optim.zero_grad()
            loss.backward()
            return loss


        optim.step(closure)
        total_loss = np.sum(loss_track[-1])
        pbar.set_postfix({
            'PDE': f"{loss_track[-1][0]:.2e}",
            'BC': f"{loss_track[-1][1]:.2e}",
            'IC': f"{loss_track[-1][2]:.2e}",
            'Data': f"{loss_track[-1][3]:.2e}",
            'Total': f"{total_loss:.2e}"
        })
        torch.save(model.state_dict(), MODEL_PATH)
        
    loss_df = pd.DataFrame(loss_track, columns=['PDE', 'BC', 'IC', 'Data'])
    loss_df.to_csv('./data/loss_curve.csv', index=False)
    return model
# ------------------ 训练 or 加载 ------------------ #
if not LOAD_MODEL:
    optimizer = LBFGS(model.parameters(), line_search_fn='strong_wolfe')
    model = train_model(model, optimizer)

# ------------------ 推理 & 可视化 ------------------ #
model.eval()
with torch.no_grad():
    T_pred = model(x_star, y_star, z_star, t_star)
    T_pred = T_pred.cpu().detach().numpy()[:, 0]

rl1 = np.sum(np.abs(T_star - T_pred)) / np.sum(np.abs(T_star))
rl2 = np.linalg.norm(T_star - T_pred, 2) / np.linalg.norm(T_star, 2)
print(f'relative L1 error: {rl1:.4f}')
print(f'relative L2 error: {rl2:.4f}')

# 可视化函数调用
plot_temperature_animation(
    x_star[:, 0].detach().cpu().numpy().reshape(-1),
    y_star[:, 0].detach().cpu().numpy().reshape(-1),
    z_star[:, 0].detach().cpu().numpy().reshape(-1),
    t_star[:, 0].detach().cpu().numpy().reshape(-1),
    T_star.flatten(),
    x_star[:, 0].detach().cpu().numpy().reshape(-1),
    y_star[:, 0].detach().cpu().numpy().reshape(-1),
    z_star[:, 0].detach().cpu().numpy().reshape(-1),
    T_pred,
    save_path=SAVE_VIZ_PATH
)

# 从 CSV 读取 loss 数据
loss_df = pd.read_csv('./data/loss_curve.csv')

# 转为 numpy array（符合 plot_loss_curve 输入要求）
loss_track = loss_df.values  # shape: (N, 4)

# 绘图
plot_loss_curve(loss_track, save_path='./viz/loss_curve.png')


plot_error_animation(
    x_star[:, 0].detach().cpu().numpy().reshape(-1),
    y_star[:, 0].detach().cpu().numpy().reshape(-1),
    z_star[:, 0].detach().cpu().numpy().reshape(-1),
    t_star[:, 0].detach().cpu().numpy().reshape(-1),
    T_star.flatten(),
    T_pred,
    save_path='./viz/error_animation.gif'
)

print(f"🎞️ 可视化结果已保存到 {SAVE_VIZ_PATH}")