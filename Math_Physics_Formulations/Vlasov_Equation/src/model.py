import os
from collections import OrderedDict

import torch
import torch.nn as nn


class DNN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        layer_list = []
        for i in range(len(layers) - 2):
            layer_list.append((f"layer_{i}", nn.Linear(layers[i], layers[i + 1])))
            layer_list.append((f"activation_{i}", nn.Tanh()))
        layer_list.append((f"layer_{len(layers) - 2}", nn.Linear(layers[-2], layers[-1])))
        self.layers = nn.Sequential(OrderedDict(layer_list))

    def forward(self, x):
        return self.layers(x)


class PhysicsInformedNN:
    def __init__(self, X, E, f, layers, device):
        self.device = device
        self.t = torch.tensor(X[:, 0:1], requires_grad=True, dtype=torch.float32, device=device)
        self.x = torch.tensor(X[:, 1:2], requires_grad=True, dtype=torch.float32, device=device)
        self.v = torch.tensor(X[:, 2:3], requires_grad=True, dtype=torch.float32, device=device)
        self.E = torch.tensor(E, dtype=torch.float32, device=device)
        self.f = torch.tensor(f, dtype=torch.float32, device=device)

        self.lambda_1 = nn.Parameter(torch.tensor([0.0], dtype=torch.float32, device=device))
        self.lambda_2 = nn.Parameter(torch.tensor([0.0], dtype=torch.float32, device=device))

        self.dnn = DNN(layers).to(device)
        self.dnn.register_parameter("lambda_1", self.lambda_1)
        self.dnn.register_parameter("lambda_2", self.lambda_2)
        self.optimizer = torch.optim.Adam(self.dnn.parameters())

    def net_f(self, t, x, v):
        result = self.dnn(torch.cat([t, x, v], dim=1))
        f = result[:, 0:1]
        E = result[:, 1:2]
        return f, E

    def net_g(self, t, x, v):
        f, E = self.net_f(t, x, v)
        f_t = torch.autograd.grad(f, t, grad_outputs=torch.ones_like(f), retain_graph=True, create_graph=True)[0]
        f_x = torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f), retain_graph=True, create_graph=True)[0]
        f_v = torch.autograd.grad(f, v, grad_outputs=torch.ones_like(f), retain_graph=True, create_graph=True)[0]
        g = f_t + self.lambda_1 * v * f_x + self.lambda_2 * E * f_v
        return g

    def train(self, n_iter, model_dir, log_every=100, save_every=20000):
        os.makedirs(model_dir, exist_ok=True)
        log_path = os.path.join(model_dir, "train_loss.txt")
        self.dnn.train()
        for epoch in range(n_iter):
            f_pred, E_pred = self.net_f(self.t, self.x, self.v)
            g_pred = self.net_g(self.t, self.x, self.v)

            loss_f = torch.mean((self.f - f_pred) ** 2)
            loss_E = torch.mean((self.E - E_pred) ** 2)
            loss_g = torch.mean(g_pred**2)
            loss = loss_g + loss_f + loss_E

            with open(log_path, "a", encoding="utf-8") as fp:
                fp.write(f"{loss.item():.6e}\n")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch % log_every == 0:
                print(
                    f"Epoch: {epoch}, Loss: {loss.item():.6e}, "
                    f"Lambda_1: {self.lambda_1.item():.6f}, Lambda_2: {self.lambda_2.item():.6f}"
                )

            if epoch % save_every == 0:
                torch.save(self.dnn.state_dict(), os.path.join(model_dir, f"PINN_{epoch}.pth"))

        torch.save(self.dnn.state_dict(), os.path.join(model_dir, "PINN_final.pth"))

    def load_model(self, model_path):
        self.dnn.load_state_dict(torch.load(model_path, map_location=self.device))

    def predict(self, T_star, X_star, V_star):
        self.dnn.eval()
        f, E = self.net_f(T_star, X_star, V_star)
        return f.detach().cpu().numpy(), E.detach().cpu().numpy()