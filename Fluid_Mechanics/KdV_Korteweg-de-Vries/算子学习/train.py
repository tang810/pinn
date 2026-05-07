
from sympy import im
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from smt.sampling_methods import LHS
import sys
import os
# 将父级路径添加到 sys.path 中
parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_path)
from net import PINNsformer,DNN
from utils import make_sequence
from datetime import datetime
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(device)
np.random.seed(1234)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


def ref_sol(rho, h1, a, X):
    h2 = 1
    # h1 = 4.13
    g = 9.18
    # rho = 0.977
    h = h2/h1
    # a = -0.77
    c0 = np.sqrt((g * h2 * (1-rho)) / ((h2/h1) + rho))
    c1 = -3/2 * c0 * ((rho-h**2) / (rho * h2 + h*h2))
    c2 = 1/6 * c0 * ((rho * h2**2 * h1 + h1**2 * h2) / (rho * h1 + h2))
    c = c0 + a*c1 / 3
    lam = np.sqrt(12*c2 / (a*c1))
    # X = np.linspace(-20,20,100)
    # # import pdb
    # # pdb.set_trace()
    sol = a/(np.cosh(X/lam)**2)
    return sol


def log(fname, s):
    f = open(fname, 'a')
    f.write(str(datetime.now()) + ': ' + s + '\n')
    f.close()


def gradients(outputs, inputs):
    grad_outputs = torch.ones_like(outputs)
    result = torch.autograd.grad(outputs, inputs, grad_outputs=grad_outputs, create_graph=True)
    return result


# def loss_equation(x, model):
#     uu = model(x)
#     u_g = gradients(uu, x)[0]
#     u_t = u_g[:, :1]
#     u_x = u_g[:, 1:2]
#     u_xg = gradients(u_x, x)[0]
#     u_xx = u_xg[:, 1:2]
#     u_xx_g = gradients(u_xx, x)[0]
#     u_xxx = u_xx_g[:, 1:2]

#     equation = u_t + uu*u_x + 0.025*u_xxx
#     loss = torch.mean(equation**2)
#     # import pdb
#     # pdb.set_trace()
#     return loss

# def loss_equation(x, rho, h1, a, model):                   # rho = rho2/rho1
def loss_equation(rho, h1, a, x, model):                   # rho = rho2/rho1
    # uu = model(x, a)
    uu = model(rho, h1, a, x)
    g = 9.81
    h2 = 1
    h = h2/h1
    c0 = torch.sqrt((g * h2 * (1-rho)) / ((h2/h1) + rho))
    c1 = -3/2 * c0 * ((rho-h**2) / (rho * h2 + h*h2))
    c2 = 1/6 * c0 * ((rho * h2**2 * h1 + h1**2 * h2) / (rho * h1 + h2))
    c = c0 + a*c1 / 3

    # gradient
    u_x = gradients(uu, x)[0]
    u_xx = gradients(u_x, x)[0]
    u_xxx = gradients(u_xx, x)[0]
    equation = (-c + c0) * u_x + c1 * uu * u_x + c2 * u_xxx
    loss = torch.mean(10*equation**2)
    # import pdb
    # pdb.set_trace()
    return loss


# def loss_IC(x, model):
#     uu = model(x)
#     u0 = torch.cos(torch.pi*x[:, 1]).view(-1, 1)
#     loss = torch.mean((uu-u0)**2)
#     # import pdb
#     # pdb.set_trace()
#     return loss

def loss_Data(rho, h1, a, x, model, ref):
    # uu = model(x, aa)
    uu = model(rho, h1, a, x)
    # ref = ref_sol(rho, h1, a, x)
    # ref = torch.tensor(ref,dtype=torch.float32).to(device)
    loss = torch.mean((uu-ref)**2)
    # fig, axes = plt.subplots(1, 10, figsize=(30, 4))

    # for i, ax in enumerate(axes):
    #     ax.plot(x[i].detach().cpu().numpy(), ref[i].detach().cpu().numpy())  # 格式化为三位小数
    #     ax.plot(x[i].detach().cpu().numpy(), aa[i].detach().cpu().numpy(), alpha=0.5)
    #     ax.legend()
    #     ax.set_title(f'Subplot {i + 1}')

    # # plt.plot(x[0].detach().cpu().numpy(), ref[0].detach().cpu().numpy())
    # # plt.plot(x[0].detach().cpu().numpy(), aa[0].detach().cpu().numpy())
    # plt.savefig("duiqi.png")
    # import pdb
    # pdb.set_trace()
    return loss


def loss_BC(rho, h1, a, x, model):
    # uu = model(x, a)
    uu = model(rho, h1, a, x)
    # u_x = gradients(uu, x)[0]
    loss = torch.mean((100*uu)**2)
    # loss2 = torch.mean((u_x)**2)
    import pdb
    pdb.set_trace()
    return loss     # +loss2


if __name__ == "__main__":
    h2 = 1
    g = 9.18
    # adjustable parameters
    # h1 = 4.13
    # a = -0.36
    # rho = 0.977
    #########################
    # num_h1 = 128
    # num_rho = 128
    # num_a = 128
    num_h1 = 1000
    num_rho = 1000
    num_a = 1000
    len_parameters = 5000
    len_x = 3000
    id_h1 = np.random.choice(len_parameters, num_h1)
    id_rho = np.random.choice(len_parameters, num_rho)
    id_a = np.random.choice(len_parameters, num_a)
    min_x = -40
    max_x = 40
    # import pdb
    # pdb.set_trace()

    h1 = np.linspace(4, 10, len_parameters).reshape(-1, 1)
    h1 = np.expand_dims(np.tile(h1[id_h1], len_x), -1)
    rho = np.linspace(0.9, 0.99, len_parameters).reshape(-1, 1)
    rho = np.expand_dims(np.tile(rho[id_rho], len_x), -1)

    a = np.linspace(-0.4, -0.05, len_parameters).reshape(-1, 1)
    a = np.expand_dims(np.tile(a[id_a], len_x), -1)
    print(a.shape)
    h = h2/h1
    x = make_sequence(num_x=num_a, step=len_x, min_x=min_x, max_x=max_x)
    x_lbc = make_sequence(num_x=num_a, step=int(len_x/2),
                          min_x=min_x, max_x=min_x)
    x_rbc = make_sequence(num_x=num_a, step=int(len_x/2),
                          min_x=max_x, max_x=max_x)
    x_bc = np.hstack((x_lbc, x_rbc))
    print(x_bc.shape)
    c0 = np.sqrt((g * h2 * (1-rho)) / ((h2/h1) + rho))
    c1 = -3/2 * c0 * ((rho-h**2) / (rho * h2 + h*h2))
    c2 = 1/6 * c0 * ((rho * h2**2 * h1 + h1**2 * h2) / (rho * h1 + h2))
    c = c0 + a*c1 / 3
    lam = np.sqrt(12*c2 / (a*c1))
    ref = ref_sol(rho, h1, a, x)

    # # 创建 3D 图形
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # # 绘制散点图
    # ax.scatter(h1[:, 0], rho[:,0], a[:,0], c=a[:,0], cmap='viridis')

    # # 设置标签
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.show()
    # plt.savefig("rhoha.png")
    # import pdb
    # pdb.set_trace()

    x_int_train = torch.tensor(x,
                               requires_grad=True,
                               dtype=torch.float32).to(device)
    x_bc_train = torch.tensor(x_bc,
                              requires_grad=True,
                              dtype=torch.float32).to(device)
    rho = torch.tensor(rho, dtype=torch.float32).to(device)
    h1 = torch.tensor(h1, dtype=torch.float32).to(device)
    a = torch.tensor(a, dtype=torch.float32).to(device)
    ref = torch.tensor(ref, dtype=torch.float32).to(device)
    dataset = TensorDataset(x_int_train, x_bc_train, rho, h1, a, ref)
    # dataset = TensorDataset(x_int_train, x_bc_train, a, ref)
    # batch_size = 8
    batch_size = 4
    dataloader = DataLoader(dataset,
                            batch_size,
                            shuffle=True)

    # plt.plot(ref[0,:,:])
    # plt.savefig("ref0.png")
    # import pdb
    # pdb.set_trace()
    model = PINNsformer(d_out=1, d_model=513, d_hidden=64, N=1, heads=3).to(device)
    # checkpoint_path_noboundary = './model_parameters/PINNsformer_num_h1=1000/model_194_noboundary.pth'
    # model.load_state_dict(torch.load(checkpoint_path_noboundary, weights_only=True))
    checkpoint_path = './model_parameters/PINNsformer_num_h1=1000/model_1500.pth'
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    # checkpoint_path = './model_parameters/PINNsformer_num_h1=1000/final_model_263.pth'
    # model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    # model.apply(init_weights)
    # checkpoint_path = './model_parameters/PINNsformer_51264_different3value/model_31000.pth'
    # model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # optimizer = torch.optim.LBFGS(model.parameters(),
    #                               lr=1,
    #                               line_search_fn='strong_wolfe')

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=0.9,
                                                           patience=1000)
    epochs = 10000
    # save_path = './model_parameters/PINNsformer_25631_different_rho'
    save_path = f'./model_parameters/PINNsformer_num_h1={num_h1}'
    # PINNsformer_different_a'
    os.makedirs(save_path, exist_ok=True)
    tic = time.time()

    # for epoch in range(263+1, epochs+263+1):
    for epoch in range(1, epochs+1):
        loss_avebatch = []
        for _,(x_int_batch, x_bc_batch, rho_batch, h1_batch, a_batch, ref_batch) in enumerate(dataloader):
            # import pdb
            # pdb.set_trace()
            def closure():
                optimizer.zero_grad()
                # 计算不同损失项
                # loss_pde = loss_equation(rho, h1, a, x_int_train, model)
                # loss_data = loss_Data(rho, h1, a, x_int_train, model, ref)
                # loss_bc = loss_BC(rho, h1, a, x_bc_train, model)
                loss_pde = loss_equation(rho_batch, h1_batch, a_batch, x_int_batch, model)
                loss_data = loss_Data(rho_batch, h1_batch, a_batch, x_int_batch, model, ref_batch)
                loss_bc = loss_BC(rho_batch, h1_batch, a_batch, x_bc_batch, model)
                # loss_pde = loss_equation(x_int_train, rho, h1, a, model)
                # loss_data = loss_Data(x_int_train, a, model, ref)
                # loss_bc = loss_BC(x_bc_train, a, model)
                # loss = 0.1 * loss_pde + 1 * loss_bc + 10 * loss_data
                ## no boundary loss
                loss = 0.1 * loss_pde + 0 * loss_bc + 10 * loss_data
                loss.backward()
                print(f"epoch {epoch}, lr {optimizer.param_groups[0]['lr']}, loss_pde: {loss_pde:.8f},loss_bc: {loss_bc:.8f}, loss_data: {loss_data:.8f}")
                return loss
            loss = optimizer.step(closure)
            # optimizer.step()
            scheduler.step(loss)
            loss_avebatch.append(loss.item())
            # import pdb
            # pdb.set_trace()
            #     # 打印损失信息
            print(f"epoch {epoch}, lr {optimizer.param_groups[0]['lr']}, loss:{loss:.8f}")
            #     # 记录日志
            log_string = f"epoch {epoch}: loss {loss:.8f}"
            log(f'./loss/loss_num_h1={num_h1}_{optimizer.param_groups[0]["lr"]:.6f}_noboundary.log', log_string)

    #     # 反向传播
    #     loss.backward()
    # #     # 打印梯度检查信息
    # #     # for name, param in model.named_parameters():
    # #     #     if param.grad is not None:  # 确保梯度存在
    # #     #         print(f"epoch {epoch}, parameter: {name}, gradient: {param.grad.norm():.8f}")

    # #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # #     # return loss

    #     # 更新参数
    #     optimizer.step()
    #     # optimizer.step(closure)
    #     # scheduler.step(loss)

        if epoch % 2 == 0:
            torch.save(model.state_dict(),
                       os.path.join(save_path,
                                    f'model_{epoch}_noboundary.pth'))
        if epoch > 10000:
        # if np.mean(loss_avebatch) < 1e-5:
        #     print(f"The loss is {np.mean(loss_avebatch)}, stop training")  
            torch.save(model.state_dict(),
                       os.path.join(save_path,
                                    f'final_model_{epoch}_noboundary.pth'))
            break
    toc = time.time()
    print(f'Total training time: {toc - tic}')
    # import pdb
    # pdb.set_trace()









    # # X = np.linspace(-20,20,1000)

    # N_f = 4000
    # N_b = 200
    # # N_ic = 300
    # x_min = min(X)
    # x_max = max(X)
    

    # # # 边界采样点
    # x_bc_train = np.concatenate([x_min * np.ones([N_b]),
    #                       x_max * np.ones([N_b])],
    #                      axis=0).reshape([-1, 1])
    
    
    # # # # 内部采样点
    # # xlimits = np.array([[-20, 20]])
    # # sampling = LHS(xlimits=xlimits)
    # # x_int_train = sampling(N_f)
    # x_int_train = X.reshape(-1,1)

    # x_bc_train = torch.tensor(x_bc_train,
    #                           requires_grad=True,
    #                           dtype=torch.float32).to(device)
    # x_int_train = torch.tensor(x_int_train,
    #                            requires_grad=True,
    #                            dtype=torch.float32).to(device)
    # rho = torch.tensor(rho, dtype=torch.float32).to(device)
    # h1 = torch.tensor(h1, dtype=torch.float32).to(device)
    # a = torch.tensor(a, dtype=torch.float32).to(device)

    # # import pdb
    # # # pdb.set_trace()
    # # layers = [1, 20, 20, 20, 1]
    # # model = DNN(layers).to(device)
    # model = PINNsformer(d_out=3, d_model=128, d_hidden=32, N=1, heads=2).to(device)
    # # checkpoint_path = './model_parameters/rho=0.977,h1=4.13,a=-0.77_32_8_1/model_100.pth'
    # # model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    # # checkpoint_path = './model_parameters/adam/model_9990.pth'
    # # model.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=device))
    # # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # optimizer = torch.optim.LBFGS(model.parameters(),
    #                               lr=0.1,
    #                               line_search_fn='strong_wolfe')
    # # # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    # # #                                                        mode='min',
    # # #                                                        factor=0.95,
    # # #                                                        patience=10)
    # # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    # #                                             step_size=10000,
    # #                                             gamma=0.5)
    # epochs = 5000
    # # save_path = './model_parameters/DNNrho=0.977,h1=4.13,a=-0.22_32_8_1'
    # save_path = './model_parameters/PINNsformer_LBFGS'
    # os.makedirs(save_path, exist_ok=True)
    # # # filepath = './model_parameters/model_Adam0_001_wall.pth'
    # tic = time.time()
    # for epoch in range(epochs):
    #     # def closure():
    #     optimizer.zero_grad()

    #     # 计算不同损失项
    #     loss_pde = loss_equation(x_int_train, model, rho, h1, a)
    #     loss_data = loss_Data(x_int_train, model, rho, h1, a)
    #     # loss_ic = loss_IC(x_ic_train, model)
    #     loss_bc = loss_BC(x_bc_train, model)
    #     # import pdb
    #     # pdb.set_trace()
    # #     # 总损失
    #     loss = 1 * loss_pde + 1 * loss_bc + 10* loss_data

    #     # 打印损失信息
    #     print(f"epoch {epoch}, lr {optimizer.param_groups[0]['lr']}, loss_pde: {loss_pde:.8f},loss_bc: {loss_bc:.8f}, loss_data: {loss_data:.8f}")

    #     # 记录日志
    #     log_string = f"epoch {epoch}: loss_pde {loss_pde:.8f}, loss_bc {loss_bc:.8f}"
    #     log(f'./loss/loss_{optimizer.param_groups[0]["lr"]:.6f}.log', log_string)

    #     # 反向传播
    #     loss.backward()
    # #     # 打印梯度检查信息
    # #     # for name, param in model.named_parameters():
    # #     #     if param.grad is not None:  # 确保梯度存在
    # #     #         print(f"epoch {epoch}, parameter: {name}, gradient: {param.grad.norm():.8f}")

    # #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # #     # return loss

    #     # 更新参数
    #     optimizer.step()
    #     # optimizer.step(closure)
    #     # scheduler.step(loss)

    #     if epoch % 10 == 0:
    #         torch.save(model.state_dict(),
    #                    os.path.join(save_path,
    #                                 f'model_{epoch}.pth'))
    # toc = time.time()
    # print(f'Total training time: {toc - tic}')
