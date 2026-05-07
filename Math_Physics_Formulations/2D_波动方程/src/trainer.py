import torch
import numpy as np
import time
import matplotlib.pyplot as plt

from src.pinn_model import lhs_sampling, torch_tensor_grad, torch_tensor_nograd, PINNLosses

class PINNTrainer:
    def __init__(self, model, domain_bounds, N_i=1000, N_b=1000, N_f=20000, device="cpu"):
        """
        参数:
        - model: PINN 模型 (MLP)
        - domain_bounds: (lb, ub)  domain lower & upper bounds (numpy array)
        - N_i, N_b, N_f: 初值、边界、域内采样点数量
        """
        self.model = model.to(device)
        self.device = device
        self.lb, self.ub = domain_bounds

        self.N_i = N_i
        self.N_b = N_b
        self.N_f = N_f

        self.loss_fn = PINNLosses(self.model, device)

    def prepare_data(self, x_range, y_range, t_range, u_initial_fn):
        # Initial Condition Data
        XY, T = self._prepare_spatial_temporal_grid(x_range, y_range, t_range)
        X_ic = np.hstack((XY, np.zeros((XY.shape[0], 1))))
        u_ic = u_initial_fn(XY[:, 0], XY[:, 1])

        idx = np.random.choice(X_ic.shape[0], self.N_i, replace=False)
        self.X_i = torch_tensor_grad(X_ic[idx], self.device)
        self.Y_i = torch_tensor_nograd(u_ic[idx].reshape(-1, 1), self.device)

        # Boundary Data
        self.X_b = self._prepare_boundary_points()

        # Domain Points
        X_f_np = lhs_sampling(self.N_f, self.lb, self.ub)
        self.X_f = torch_tensor_grad(X_f_np, self.device)

    def _prepare_spatial_temporal_grid(self, x_range, y_range, t_range):
        x = np.linspace(x_range[0], x_range[1], 65)
        y = np.linspace(y_range[0], y_range[1], 65)
        X, Y = np.meshgrid(x, y)
        XY = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))
        T = np.linspace(t_range[0], t_range[1], int((t_range[1] - t_range[0]) * 1 / (6 / 30**2)) + 1)
        return XY, T

    def _prepare_boundary_points(self):
        X_left = lhs_sampling(self.N_b, self.lb, self.ub)
        X_left[:, 0:1] = self.lb[0]
        X_right = lhs_sampling(self.N_b, self.lb, self.ub)
        X_right[:, 0:1] = self.ub[0]
        X_bottom = lhs_sampling(self.N_b, self.lb, self.ub)
        X_bottom[:, 1:2] = self.lb[1]
        X_top = lhs_sampling(self.N_b, self.lb, self.ub)
        X_top[:, 1:2] = self.ub[1]
        X_b_np = np.vstack((X_right, X_top, X_left, X_bottom))
        np.random.shuffle(X_b_np)
        return torch_tensor_grad(X_b_np, self.device)

    def train(self, epochs=1000, lr=1e-3, lr_decay_step=5000, lr_gamma=0.9):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_gamma)

        loss_record = []
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()

            initial_loss = self.loss_fn.reconstruction_loss(self.X_i, self.Y_i) + self.loss_fn.initial_velocity_loss(self.X_i)
            boundary_loss = self.loss_fn.boundary_loss(self.X_b)
            domain_loss = self.loss_fn.pde_loss(self.X_f)

            total_loss = initial_loss + boundary_loss + domain_loss
            loss_record.append(total_loss.item())

            total_loss.backward()
            optimizer.step()
            scheduler.step()

            if epoch % 100 == 0 or epoch == 1:
                print(f"Epoch {epoch}: Init={initial_loss.item():.3e}, Bound={boundary_loss.item():.3e}, Domain={domain_loss.item():.3e}, Total={total_loss.item():.3e}")

        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds.")

        return loss_record

    def plot_loss(self, loss_record, save_path=None):
        plt.figure()
        plt.plot(loss_record)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        if save_path:
            plt.savefig(save_path)
        plt.show()
