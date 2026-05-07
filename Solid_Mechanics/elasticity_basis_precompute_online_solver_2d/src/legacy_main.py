import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad as torch_grad 
from torch.optim.lr_scheduler import LambdaLR
from pyDOE2 import lhs
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import random 

# === Global Constants and Data ===
pi_val = np.pi
E_scaled = 4.0 / 3.0
nu = 1.0 / 3.0
lam_const_val = E_scaled * nu / ((1 - 2 * nu) * (1 + nu))
mu_const_val = E_scaled / (2 * (1 + nu))

xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0

q_train = torch.linspace(0.1, 10.0, 100) 
q_test = torch.linspace(0.1, 10.0, 20)  

DTYPE = torch.float64
torch.set_default_dtype(DTYPE)

lam_tensor = None
mu_tensor = None

# Default loss weights for offline precomputation
OFFLINE_PINN_LOSS_WEIGHTS = {'eq': 1.0, 'const': 1.0, 'bc_u': 1.0, 'bc_s': 1.0}

# === Exact Solution & Plotting Helper Functions ===
def u_x_ext(x, y): 
    x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
    y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y
    return torch.tensor(np.cos(2 * pi_val * x_np) * np.sin(pi_val * y_np), dtype=DTYPE)

def u_y_ext(x, y, Q_val): 
    x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
    y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y
    return torch.tensor(np.sin(pi_val * x_np) * (Q_val / 4.0) * (y_np ** 4), dtype=DTYPE)

def u_x_ext_np_local(x_np, y_np): 
    return np.cos(2 * pi_val * x_np) * np.sin(pi_val * y_np)

def u_y_ext_np_local(x_np, y_np, Q_val_plot): 
    return np.sin(pi_val * x_np) * (Q_val_plot / 4.0) * (y_np ** 4)

# === Data Generation Functions ===
def get_collocation_and_forces(Q_val, N_col=1000, seed_value=123, device_in=None):
    np.random.seed(seed_value)
    Xc = lhs(2, N_col, criterion='center')
    x_col_np = xmin + (xmax - xmin) * Xc[:, 0:1]
    y_col_np = ymin + (ymax - ymin) * Xc[:, 1:2]
    xy_col_np = np.hstack([x_col_np, y_col_np])
    
    pi = pi_val
    term_lambda_equivalent_fx = (
        -4 * pi ** 2 * np.cos(2 * pi * x_col_np) * np.sin(pi * y_col_np) +
        pi * np.cos(pi * x_col_np) * Q_val * (y_col_np ** 3)
    )
    term_mu_equivalent_fx = (
        -9 * pi ** 2 * np.cos(2 * pi * x_col_np) * np.sin(pi * y_col_np) +
        pi * np.cos(pi * x_col_np) * Q_val * (y_col_np ** 3)
    )
    fx_np = 1.0 * term_lambda_equivalent_fx + 0.5 * term_mu_equivalent_fx
    
    fy_np = (
        lam_const_val * (3 * np.sin(pi * x_col_np) * Q_val * y_col_np ** 2 - 2 * pi ** 2 * np.sin(2 * pi * x_col_np) * np.cos(pi * y_col_np)) +
        mu_const_val * (
            6 * np.sin(pi * x_col_np) * Q_val * y_col_np ** 2
            - 2 * pi ** 2 * np.sin(2 * pi * x_col_np) * np.cos(pi * y_col_np)
            - (pi ** 2) * np.sin(pi * x_col_np) * Q_val * y_col_np ** 4 / 4.0
        )
    )
    
    xy_col_t = torch.tensor(xy_col_np, dtype=DTYPE, device=device_in).requires_grad_(True)
    fx_col_t = torch.tensor(fx_np, dtype=DTYPE, device=device_in)
    fy_col_t = torch.tensor(fy_np, dtype=DTYPE, device=device_in)
    return xy_col_t, fx_col_t, fy_col_t

def get_boundary_data_separated(Q_val, device_in, N_b=50, seed_value_offset=1):
    np.random.seed(seed_value_offset)
    xt_lhs_np = lhs(1, N_b, criterion='center')
    np.random.seed(seed_value_offset + 1) 
    yt_lhs_np = lhs(1, N_b, criterion='center')

    x_up_np = xmin + (xmax - xmin) * xt_lhs_np; y_up_np = np.full_like(x_up_np, ymax)
    x_lo_np = xmin + (xmax - xmin) * xt_lhs_np; y_lo_np = np.full_like(x_lo_np, ymin)
    y_ri_np = ymin + (ymax - ymin) * yt_lhs_np; x_ri_np = np.full_like(y_ri_np, xmax)
    y_le_np = ymin + (ymax - ymin) * yt_lhs_np; x_le_np = np.full_like(y_le_np, xmin)

    xy_up_t = torch.tensor(np.hstack([x_up_np, y_up_np]), dtype=DTYPE, device=device_in)
    xy_lo_t = torch.tensor(np.hstack([x_lo_np, y_lo_np]), dtype=DTYPE, device=device_in)
    xy_ri_t = torch.tensor(np.hstack([x_ri_np, y_ri_np]), dtype=DTYPE, device=device_in)
    xy_le_t = torch.tensor(np.hstack([x_le_np, y_le_np]), dtype=DTYPE, device=device_in)
    
    X_list_boundary_t = [xy_up_t, xy_lo_t, xy_ri_t, xy_le_t]

    ux_up_target_t = torch.zeros((N_b, 1), dtype=DTYPE, device=device_in)
    ux_lo_target_t = torch.zeros((N_b, 1), dtype=DTYPE, device=device_in)
    ux_b_target_t = torch.cat([ux_up_target_t, ux_lo_target_t], dim=0)

    uy_lo_target_t = torch.zeros((N_b, 1), dtype=DTYPE, device=device_in)
    uy_ri_target_t = torch.zeros((N_b, 1), dtype=DTYPE, device=device_in)
    uy_le_target_t = torch.zeros((N_b, 1), dtype=DTYPE, device=device_in)
    uy_b_target_t = torch.cat([uy_lo_target_t, uy_ri_target_t, uy_le_target_t], dim=0)

    Sxx_ri_target_t = torch.zeros((N_b, 1), dtype=DTYPE, device=device_in)
    Sxx_le_target_t = torch.zeros((N_b, 1), dtype=DTYPE, device=device_in)
    Sxx_b_target_t = torch.cat([Sxx_ri_target_t, Sxx_le_target_t], dim=0)
    
    Syy_up_val_np = (lam_const_val + 2 * mu_const_val) * Q_val * np.sin(pi_val * x_up_np)
    Syy_b_target_t = torch.tensor(Syy_up_val_np, dtype=DTYPE, device=device_in).reshape(-1,1)

    return X_list_boundary_t, ux_b_target_t, uy_b_target_t, Sxx_b_target_t, Syy_b_target_t

# === Model Definition ===
class PINN(nn.Module):
    """
    Physics-Informed Neural Network for 2D elasticity problems.
    Can be used in two modes:
    1. Standard PINN mode (precomp=None): Uses a neural network to approximate solution
    2. GPT-PINN mode (precomp with basis functions): Uses a "network of networks" approach
    """
    def __init__(self, layers_config=None, precomp=None, dropout_p=0.0, disable_scaling_flag=False):
        super().__init__()
        self.precomp = precomp
        self.disable_scaling = disable_scaling_flag

        if precomp is None: 
            # Standard PINN mode - single neural network
            if layers_config is None: 
                self.layers_cfg = [2] + [20]*8 + [5] 
            else:
                self.layers_cfg = layers_config
            
            # Neural network architecture
            seq = []
            for i in range(len(self.layers_cfg)-2):
                seq += [nn.Linear(self.layers_cfg[i], self.layers_cfg[i+1]), nn.Tanh()]
                if dropout_p > 0:
                    seq += [nn.Dropout(dropout_p)]
            seq += [nn.Linear(self.layers_cfg[-2], self.layers_cfg[-1])]
            self.net = nn.Sequential(*seq)
            
            # Initialize weights
            for m in self.net:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.zeros_(m.bias)
        else: 
            # GPT-PINN mode - "network of networks" approach
            if not all(k in precomp for k in ['P', 'Rx', 'Ry']):
                raise ValueError("Precomputed data for GPT-PINN must contain 'P', 'Rx', and 'Ry' tensors.")
            
            # Store basis functions
            self.P_base = precomp['P'] 
            self.Rx_base = precomp['Rx']
            self.Ry_base = precomp['Ry']
            
            # Initialize coefficients (meta-network weights)
            # This is the reduced-dimension network with only one hidden layer
            self.c = nn.Parameter(torch.ones(self.P_base.shape[1], dtype=self.P_base.dtype))

    def _scale_input(self, x_input_tensor):
        """Scale input to [-1, 1] for better neural network performance"""
        if self.disable_scaling: 
            return x_input_tensor
        min_vals = torch.tensor([xmin, ymin], dtype=x_input_tensor.dtype, device=x_input_tensor.device)
        max_vals = torch.tensor([xmax, ymax], dtype=x_input_tensor.dtype, device=x_input_tensor.device)
        return 2.0 * (x_input_tensor - min_vals) / (max_vals - min_vals) - 1.0

    def forward(self, xy=None, indices=None):
        if self.precomp is None: 
            # Standard PINN mode
            if xy is None:
                raise ValueError("xy must be provided in standard PINN mode")
            return self.net(self._scale_input(xy))
        else: 
            # GPT-PINN mode - combine basis functions with coefficients
            current_c_device = self.c.device
            P_on_device = self.P_base.to(current_c_device)
            
            if indices is not None: 
                # Use only specific indices of the precomputed basis
                P_subset = P_on_device[indices]
                return torch.einsum('pbf,b->pf', P_subset, self.c)
            elif xy is not None: 
                # For direct xy input (handled by predict_and_plot)
                raise NotImplementedError("GPT-PINN forward with xy is handled by predict_and_plot logic")
            else: 
                # Use all precomputed basis functions
                return torch.einsum('pbf,b->pf', P_on_device, self.c)

    def loss_residual_gpt(self, X_col_coords_for_gpt, f_x_gpt, f_y_gpt):
        """Calculate PDE residual loss for GPT-PINN"""
        N_col = X_col_coords_for_gpt.shape[0]
        current_c_device = self.c.device
        
        # Move basis data to current device
        Rx_basis_at_col = self.Rx_base[:N_col].to(current_c_device)
        Ry_basis_at_col = self.Ry_base[:N_col].to(current_c_device)
        
        # Calculate predicted derivatives using basis functions and coefficients
        Rx_pred = torch.einsum('pb,b->p', Rx_basis_at_col, self.c)
        Ry_pred = torch.einsum('pb,b->p', Ry_basis_at_col, self.c)
        
        # Reshape to match target forces
        Rx_pred = Rx_pred.view_as(f_x_gpt)
        Ry_pred = Ry_pred.view_as(f_y_gpt)
        
        # Mean squared error loss between predicted and target forces
        loss_eq = torch.mean((Rx_pred - f_x_gpt)**2 + (Ry_pred - f_y_gpt)**2)
        return loss_eq

    def loss_bcs_gpt(self, Xb_list_coords_for_gpt, Ub_list_targets_for_gpt, N_col_offline):
        """Calculate boundary condition loss for GPT-PINN"""
        current_c_device = self.c.device
        current_offset = N_col_offline 
        
        # Collect indices for all boundary points
        all_indices = []
        for Xb_tensor in Xb_list_coords_for_gpt:
            num_pts_on_this_boundary = Xb_tensor.shape[0]
            all_indices.append(torch.arange(current_offset, 
                                           current_offset + num_pts_on_this_boundary, 
                                           device=current_c_device))
            current_offset += num_pts_on_this_boundary
        boundary_indices = torch.cat(all_indices)
        
        # Forward pass for boundary points
        U_pred_bc = self.forward(indices=boundary_indices) 
        
        # Combine all boundary targets and calculate MSE loss
        Ub_target_combined = torch.cat(Ub_list_targets_for_gpt, dim=0).to(U_pred_bc.device)
        loss_bc_val = torch.mean((U_pred_bc - Ub_target_combined)**2) 
        return loss_bc_val

# === Loss Function for Offline PINN Precomputation ===
def loss_fn_offline_pinn(model, xy_c_t, fx_c_t, fy_c_t,
                         X_list_b_t, ux_b_tgt_t, uy_b_tgt_t, Sxx_b_tgt_t, Syy_b_tgt_t,
                         loss_weights, device_in):
    """
    Calculate physics-informed loss for standard PINN during offline precomputation.
    Includes PDE residual loss, constitutive equation loss, and boundary condition losses.
    """
    global lam_tensor, mu_tensor 
    if lam_tensor is None or mu_tensor is None or lam_tensor.device != device_in or mu_tensor.device != device_in:
        current_lam_device = lam_tensor.device if lam_tensor is not None else "None"
        current_mu_device = mu_tensor.device if mu_tensor is not None else "None"
        raise ValueError(f"lam_tensor({current_lam_device})/mu_tensor({current_mu_device}) device mismatch or not initialized. Expected: {device_in}")

    loss_components_for_sum = []
    loss_r_val, loss_const_val, loss_b_u_val, loss_b_s_val = [torch.tensor(0.0, device=device_in) for _ in range(4)]

    # Calculate PDE residual and constitutive equation losses
    if xy_c_t.numel() > 0:
        pred_col = model(xy_c_t)
        ux_c, uy_c = pred_col[:,0:1], pred_col[:,1:2]
        sxx_c, syy_c, sxy_c = pred_col[:,2:3], pred_col[:,3:4], pred_col[:,4:5]

        # Calculate displacement gradients
        dux_c_grads = torch_grad(outputs=ux_c.sum(), inputs=xy_c_t, create_graph=True)[0]
        dux_dx_c, dux_dy_c = dux_c_grads[:,0:1], dux_c_grads[:,1:2]
        
        duy_c_grads = torch_grad(outputs=uy_c.sum(), inputs=xy_c_t, create_graph=True)[0]
        duy_dx_c, duy_dy_c = duy_c_grads[:,0:1], duy_c_grads[:,1:2]

        # Constitutive equation residuals
        r_const1 = sxx_c - ((lam_tensor + 2 * mu_tensor) * dux_dx_c + lam_tensor * duy_dy_c)
        r_const2 = syy_c - ((lam_tensor + 2 * mu_tensor) * duy_dy_c + lam_tensor * dux_dx_c)
        r_const3 = sxy_c - (mu_tensor * (dux_dy_c + duy_dx_c))
        loss_const_val = r_const1.abs().mean() + r_const2.abs().mean() + r_const3.abs().mean()
        loss_components_for_sum.append(loss_weights.get('const', 1.0) * loss_const_val)
        
        # Calculate stress gradients
        dsxx_c_grads = torch_grad(outputs=sxx_c.sum(), inputs=xy_c_t, create_graph=True)[0]
        dsxx_dx_c = dsxx_c_grads[:,0:1]
        
        dsxy_c_grads = torch_grad(outputs=sxy_c.sum(), inputs=xy_c_t, create_graph=True)[0]
        dsxy_dx_c, dsxy_dy_c = dsxy_c_grads[:,0:1], dsxy_c_grads[:,1:2]
        
        dsyy_c_grads = torch_grad(outputs=syy_c.sum(), inputs=xy_c_t, create_graph=True)[0]
        dsyy_dy_c = dsyy_c_grads[:,1:2]
        
        # PDE residuals (equilibrium equations)
        r_eq1 = dsxx_dx_c + dsxy_dy_c - fx_c_t
        r_eq2 = dsxy_dx_c + dsyy_dy_c - fy_c_t
        loss_r_val = r_eq1.abs().mean() + r_eq2.abs().mean()
        loss_components_for_sum.append(loss_weights.get('eq', 1.0) * loss_r_val)

    # Calculate boundary condition losses
    if X_list_b_t[0].numel() > 0:
        xy_up_t, xy_lo_t, xy_ri_t, xy_le_t = X_list_b_t
        pred_up = model(xy_up_t)
        pred_lo = model(xy_lo_t)
        pred_ri = model(xy_ri_t)
        pred_le = model(xy_le_t)
        
        # Displacement boundary conditions
        current_loss_b_u = torch.tensor(0.0, device=device_in)
        if ux_b_tgt_t.numel() > 0:
            ux_up_p, ux_lo_p = pred_up[:,0:1], pred_lo[:,0:1]
            current_loss_b_u += (torch.cat([ux_up_p, ux_lo_p], dim=0) - ux_b_tgt_t).abs().mean()
        if uy_b_tgt_t.numel() > 0:
            uy_lo_p, uy_ri_p, uy_le_p = pred_lo[:,1:2], pred_ri[:,1:2], pred_le[:,1:2]
            current_loss_b_u += (torch.cat([uy_lo_p, uy_ri_p, uy_le_p], dim=0) - uy_b_tgt_t).abs().mean()
        if current_loss_b_u.item() != 0.0 or current_loss_b_u.requires_grad: 
             loss_components_for_sum.append(loss_weights.get('bc_u', 1.0) * current_loss_b_u)
        loss_b_u_val = current_loss_b_u

        # Stress boundary conditions
        current_loss_b_s = torch.tensor(0.0, device=device_in)
        if Sxx_b_tgt_t.numel() > 0:
            sxx_ri_p, sxx_le_p = pred_ri[:,2:3], pred_le[:,2:3]
            current_loss_b_s += (torch.cat([sxx_ri_p, sxx_le_p], dim=0) - Sxx_b_tgt_t).abs().mean()
        if Syy_b_tgt_t.numel() > 0:
            syy_up_p = pred_up[:,3:4]
            current_loss_b_s += (syy_up_p - Syy_b_tgt_t).abs().mean()
        if current_loss_b_s.item() != 0.0 or current_loss_b_s.requires_grad:
            loss_components_for_sum.append(loss_weights.get('bc_s', 1.0) * current_loss_b_s)
        loss_b_s_val = current_loss_b_s
    
    # Combine all loss components
    if not loss_components_for_sum:
        # Handle empty loss case
        dummy_param = next(iter(model.parameters()), None) 
        if dummy_param is not None: 
            total_loss_val = (dummy_param - dummy_param.detach()) * 0.0
        else: 
            total_loss_val = torch.tensor(0.0, device=device_in, requires_grad=True) 
    else:
        total_loss_val = torch.sum(torch.stack(loss_components_for_sum))
        
    return total_loss_val, loss_r_val, loss_const_val, loss_b_u_val, loss_b_s_val

# === Offline Precomputation (Saves model state_dicts) ===
def offline_precomp(
    num_basis_to_generate: int, pinn_epochs: int, base_lr_offline: float,
    pinn_layers_cfg: list, disable_scaling_offline: bool, dropout_offline: float,
    loss_weights_offline: dict, N_col_offline: int, N_b_offline: int,
    device_str: str, basis_path: str, seed_val: int, Q_plot_target: float = None
):
    """
    Generate basis functions for GPT-PINN by training individual PINNs at 
    selected parameter values. Similar to the approach in B_*.py files.
    """
    dev = torch.device(device_str)
    
    if loss_weights_offline is None:
        loss_weights_offline = OFFLINE_PINN_LOSS_WEIGHTS

    # Select parameter values for basis functions
    Q_vals_all = q_train.cpu().numpy()
    if Q_plot_target is not None and num_basis_to_generate == 1:
        print(f"INFO: Targeting Q={Q_plot_target} for single basis generation and plotting.")
        Q_sel = np.array([Q_plot_target])
        actual_num_basis = 1
        min_q_train, max_q_train = Q_vals_all.min(), Q_vals_all.max()
        if not (min_q_train <= Q_plot_target <= max_q_train):
             print(f"Warning: Q_plot_target={Q_plot_target} is outside the q_train range [{min_q_train},{max_q_train}]")
    else:
        # Distribute basis functions evenly across parameter range
        actual_num_basis = min(num_basis_to_generate, len(Q_vals_all))
        if actual_num_basis == 0:
            print("Warning: Number of basis functions to generate is 0. Skipping offline_precomp.")
            return None
        indices = np.linspace(0, len(Q_vals_all)-1, actual_num_basis, dtype=int)
        Q_sel = Q_vals_all[indices]

    # Initialize storage for basis function data
    P_list, Rx_list, Ry_list = [], [], []
    model_state_dicts = [] 
    collocation_coord_seed = seed_val 
    current_pinn_layers_cfg_used = pinn_layers_cfg if pinn_layers_cfg else ([2] + [20]*8 + [5])

    successful_q_values_for_basis = [] # Store Q values for which training succeeded

    for i, Q_basis_val in enumerate(Q_sel):
        print(f"[Precomp] Training basis {i+1}/{actual_num_basis} for Q = {Q_basis_val:.4f}")
        # Set random seeds for reproducibility
        current_pinn_training_seed = seed_val + i + 100 
        torch.manual_seed(current_pinn_training_seed)
        np.random.seed(current_pinn_training_seed) 
        random.seed(current_pinn_training_seed)
        if dev.type == 'cuda': 
            torch.cuda.manual_seed_all(current_pinn_training_seed)
        
        # Generate training data for this parameter value
        xy_c_t, fx_c_t, fy_c_t = get_collocation_and_forces(
            Q_basis_val, N_col=N_col_offline, seed_value=collocation_coord_seed, device_in=dev
        )
        boundary_points_seed = seed_val + 1 + i 
        X_list_b_t, ux_b_tgt_t, uy_b_tgt_t, Sxx_b_tgt_t, Syy_b_tgt_t = \
            get_boundary_data_separated(Q_basis_val, dev, N_b=N_b_offline, seed_value_offset=boundary_points_seed)

        # Create and train PINN for this parameter value
        net_single_q = PINN(
            layers_config=current_pinn_layers_cfg_used, 
            precomp=None,
            dropout_p=dropout_offline,
            disable_scaling_flag=disable_scaling_offline
        ).to(dev)
        
        optimizer = optim.Adam(net_single_q.parameters(), lr=base_lr_offline)
        
        # Learning rate scheduler (similar to B_main.py approach)
        def lr_lambda_offline_factory(base_lr):
            def lr_lambda_offline(step_num):
                if step_num < 1000: 
                    return 1.0
                elif step_num < 3000: 
                    return (1e-3 / base_lr) if base_lr > 1e-9 else 0.0
                else: 
                    return (5e-4 / base_lr) if base_lr > 1e-9 else 0.0
            return lr_lambda_offline
        
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda_offline_factory(base_lr_offline))

        # Training loop
        training_successful_for_q = True
        for ep in range(1, pinn_epochs + 1):
            net_single_q.train()
            optimizer.zero_grad()
            
            total_loss, l_r, l_const, l_b_u, l_b_s = loss_fn_offline_pinn(
                net_single_q, xy_c_t, fx_c_t, fy_c_t,
                X_list_b_t, ux_b_tgt_t, uy_b_tgt_t, Sxx_b_tgt_t, Syy_b_tgt_t,
                loss_weights_offline, dev
            )
            
            # Check for NaN/Inf loss
            if torch.isnan(total_loss) or torch.isinf(total_loss) or not total_loss.requires_grad:
                loss_val_str = str(total_loss.item()) if hasattr(total_loss, 'item') else str(total_loss)
                print(f"  Epoch {ep}: Problem with total_loss. Stopping for this Q_basis. Loss: {loss_val_str}, Grad: {total_loss.requires_grad}")
                training_successful_for_q = False
                break 
                
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Print training progress
            if ep % (max(1, pinn_epochs // 10)) == 0 or ep == 1 or ep == pinn_epochs:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  E {ep:05d}/{pinn_epochs} | LR {current_lr:.1e} | "
                      f"Tot {total_loss.item():.3e} | Eq {l_r.item():.3e} | "
                      f"Const {l_const.item():.3e} | BC_U {l_b_u.item():.3e} | BC_S {l_b_s.item():.3e}")
        
        if training_successful_for_q:
            net_single_q.eval() 
            
            # Plot fields immediately after training if requested
            if Q_plot_target is not None and np.isclose(Q_basis_val, Q_plot_target):
                print(f"INFO: Plotting fields for Q={Q_basis_val:.4f} immediately after training.")
                N_plot_offline = 101 
                x_space = np.linspace(xmin, xmax, N_plot_offline)
                y_space = np.linspace(ymin, ymax, N_plot_offline)
                X_np_grd, Y_np_grd = np.meshgrid(x_space, y_space) 
                xy_plt_flat = np.hstack((X_np_grd.flatten()[:,None], Y_np_grd.flatten()[:,None])) 
                xy_plt_tensor = torch.tensor(xy_plt_flat, dtype=DTYPE, device=dev) 
                
                with torch.no_grad():
                    predictions_on_grid_t = net_single_q(xy_plt_tensor) 
                    
                predictions_on_grid_np = predictions_on_grid_t.cpu().numpy()
                
                # Plot using the helper function
                plot_solution_fields_matching_reference( 
                    model_outputs_nn=predictions_on_grid_np, 
                    X_plot_coords_np=X_np_grd,
                    Y_plot_coords_np=Y_np_grd, 
                    Q_val_plot=Q_basis_val,
                    config_case_name=f"Offline PINN (Q={Q_basis_val:.2f})",
                    prefix=f"offline_pinn_Q{Q_basis_val:.2f}"
                )
            
            # Generate and store basis function data
            # 1. Get model outputs
            xy_c_eval, _, _ = get_collocation_and_forces(
                Q_basis_val, N_col=N_col_offline, seed_value=collocation_coord_seed, device_in=dev
            )
            X_list_b_eval, _, _, _, _ = get_boundary_data_separated(
                Q_basis_val, dev, N_b=N_b_offline, seed_value_offset=boundary_points_seed
            )
            X_all_list_eval = [xy_c_eval.detach()] + [xb.detach() for xb in X_list_b_eval]
            X_all_eval = torch.cat(X_all_list_eval, dim=0)
            
            with torch.no_grad():
                P_i_all_fields = net_single_q(xy=X_all_eval)
                P_list.append(P_i_all_fields.unsqueeze(1))

            # 2. Get derivatives (important for GPT-PINN approach)
            X_all_grad = X_all_eval.clone().detach().requires_grad_(True)
            U_all_for_grad = net_single_q(xy=X_all_grad) 
            Sxx_deriv = U_all_for_grad[:, 2:3]
            Sxy_deriv = U_all_for_grad[:, 4:5]
            Syy_deriv = U_all_for_grad[:, 3:4]
            
            dSxx_dx_val = torch_grad(Sxx_deriv.sum(), X_all_grad, create_graph=True, retain_graph=True)[0][:,0:1]
            dSxy_grads = torch_grad(Sxy_deriv.sum(), X_all_grad, create_graph=True, retain_graph=True)[0]
            dSxy_dx_val = dSxy_grads[:,0:1]
            dSxy_dy_val = dSxy_grads[:,1:2]
            dSyy_dy_val = torch_grad(Syy_deriv.sum(), X_all_grad, create_graph=False)[0][:,1:2]
            
            # These represent the terms in PDE residuals
            Rx_i = (dSxx_dx_val + dSxy_dy_val).detach()
            Ry_i = (dSxy_dx_val + dSyy_dy_val).detach()
            
            Rx_list.append(Rx_i)
            Ry_list.append(Ry_i)
            model_state_dicts.append({k: v.cpu() for k, v in net_single_q.state_dict().items()})
            successful_q_values_for_basis.append(Q_basis_val)

    if not P_list: 
        print("ERROR: No basis functions were successfully trained and stored.")
        return None
        
    # Combine all basis function data
    P_tensor = torch.cat(P_list, dim=1)
    Rx_tensor = torch.cat(Rx_list, dim=1)
    Ry_tensor = torch.cat(Ry_list, dim=1)
    Q_sel_final = np.array(successful_q_values_for_basis) 

    # Save precomputed data
    precomputed_data = {
        'P': P_tensor.cpu(),
        'Rx': Rx_tensor.cpu(),
        'Ry': Ry_tensor.cpu(),
        'Q_sel': Q_sel_final, 
        'N_col_offline': N_col_offline,
        'N_b_offline': N_b_offline, 
        'seed_val': seed_val,
        'model_state_dicts': model_state_dicts,
        'pinn_layers_cfg': current_pinn_layers_cfg_used 
    }
    torch.save(precomputed_data, basis_path)
    print(f"[Precomp] Done. Saved to {basis_path}. Shapes P:{P_tensor.shape}, Rx:{Rx_tensor.shape}, Ry:{Ry_tensor.shape}, Num State Dicts: {len(model_state_dicts)}")
    return precomputed_data

# === Online GPT-PINN Solver ===
def solve_gpt_for_Q(Q_new: float, basis_path: str, lr: float, steps: int, device_in: torch.device) -> tuple:
    """
    Solve a parametric problem for a new Q value using GPT-PINN approach.
    Similar to gpt_test in B_test.py.
    """
    # Load precomputed basis
    if not os.path.exists(basis_path):
        print(f"Error: Basis file {basis_path} not found for GPT solve.")
        return None, None
    try:
        precomp = torch.load(basis_path, map_location=device_in) 
    except Exception as e:
        print(f"Error loading basis file {basis_path}: {e}")
        return None, None

    # Get training parameters from precomputed data
    N_col_offline_loaded = precomp.get('N_col_offline', 1000) 
    
    # Create GPT-PINN model with precomputed basis
    model_gpt = PINN(precomp=precomp).to(device_in) 
    
    # Prepare optimizer with adaptive learning rate scheduler
    optimizer = optim.Adam([model_gpt.c], lr=lr)
    
    def lr_lambda_gpt(step_num): 
        if step_num < steps * 0.5: 
            return 1.0 
        elif step_num < steps * 0.8: 
            return 0.5 
        else: 
            return 0.1
            
    scheduler = LambdaLR(optimizer, lr_lambda_gpt)

    # Generate collocation points and forces for this Q value
    gpt_colloc_seed = precomp.get('seed_val', 123) 
    xy_col_gpt, fx_gpt, fy_gpt = get_collocation_and_forces(
        Q_new, N_col=N_col_offline_loaded, seed_value=gpt_colloc_seed, device_in=device_in
    )
    
    # Function to get boundary data in the format expected by GPT-PINN
    def get_boundary_data_original_format_for_gpt(Q, device_curr, N_b_curr=50, seedval=123):
        np.random.seed(seedval + 2000) 
        xt_lhs_np = lhs(1, N_b_curr, criterion='center')
        np.random.seed(seedval + 2001) 
        yt_lhs_np = lhs(1, N_b_curr, criterion='center') 
        
        # Top boundary
        x_top_np = xmin + (xmax - xmin) * xt_lhs_np
        y_top_np = np.full_like(x_top_np, ymax)
        xy_top_t = torch.tensor(np.hstack([x_top_np, y_top_np]), dtype=DTYPE, device=device_curr)
        ux_top_t = torch.zeros((N_b_curr, 1), dtype=DTYPE, device=device_curr)
        Syy_top_np = (lam_const_val + 2 * mu_const_val) * Q * np.sin(pi_val * x_top_np)
        Syy_top_t = torch.tensor(Syy_top_np, dtype=DTYPE, device=device_curr).reshape(-1, 1)
        
        # Bottom boundary
        x_bot_np = xmin + (xmax - xmin) * xt_lhs_np
        y_bot_np = np.full_like(x_bot_np, ymin)
        xy_bot_t = torch.tensor(np.hstack([x_bot_np, y_bot_np]), dtype=DTYPE, device=device_curr)
        ux_bot_t = torch.zeros((N_b_curr, 1), dtype=DTYPE, device=device_curr)
        uy_bot_t = torch.zeros((N_b_curr, 1), dtype=DTYPE, device=device_curr)
        
        # Right boundary
        y_ri_np = ymin + (ymax - ymin) * yt_lhs_np
        x_ri_np = np.full_like(y_ri_np, xmax)
        xy_ri_t = torch.tensor(np.hstack([x_ri_np, y_ri_np]), dtype=DTYPE, device=device_curr)
        uy_ri_t = torch.zeros((N_b_curr, 1), dtype=DTYPE, device=device_curr)
        Sxx_ri_t = torch.zeros((N_b_curr, 1), dtype=DTYPE, device=device_curr)
        
        # Left boundary
        y_le_np = ymin + (ymax - ymin) * yt_lhs_np
        x_le_np = np.full_like(y_le_np, xmin)
        xy_le_t = torch.tensor(np.hstack([x_le_np, y_le_np]), dtype=DTYPE, device=device_curr)
        uy_le_t = torch.zeros((N_b_curr, 1), dtype=DTYPE, device=device_curr)
        Sxx_le_t = torch.zeros((N_b_curr, 1), dtype=DTYPE, device=device_curr)
        
        # Combine all boundary coordinates and target values
        Xb_list_orig = [xy_top_t, xy_bot_t, xy_ri_t, xy_le_t]
        Ub_list_orig = [
            torch.cat([ux_top_t, torch.zeros((N_b_curr, 1), dtype=DTYPE, device=device_curr), 
                       torch.zeros((N_b_curr, 1), dtype=DTYPE, device=device_curr), 
                       Syy_top_t, torch.zeros((N_b_curr, 1), dtype=DTYPE, device=device_curr)], dim=1),
            torch.cat([ux_bot_t, uy_bot_t, torch.zeros((N_b_curr, 3), dtype=DTYPE, device=device_curr)], dim=1),
            torch.cat([torch.zeros((N_b_curr, 1), dtype=DTYPE, device=device_curr), uy_ri_t, Sxx_ri_t, 
                       torch.zeros((N_b_curr, 2), dtype=DTYPE, device=device_curr)], dim=1),
            torch.cat([torch.zeros((N_b_curr, 1), dtype=DTYPE, device=device_curr), uy_le_t, Sxx_le_t, 
                       torch.zeros((N_b_curr, 2), dtype=DTYPE, device=device_curr)], dim=1)
        ]
        return Xb_list_orig, Ub_list_orig
    
    # Generate boundary data for this Q value
    N_b_gpt = precomp.get('N_b_offline', 50)
    gpt_bc_seed = precomp.get('seed_val', 123) + 500 
    Xb_list_gpt, Ub_list_gpt = get_boundary_data_original_format_for_gpt(
        Q_new, device_in, N_b_curr=N_b_gpt, seedval=gpt_bc_seed
    )

    # Training loop - similar to gpt_test in B_test.py
    model_gpt.train()
    loss_hist = []
    print_every_gpt = max(1, steps // 20 if steps > 0 else 1) 
    
    # Similar to what's done in B_main.py - initialize coefficients based on 
    # weighted distance to parameter values used in training
    # Get parameter values for basis functions
    Q_sel_loaded = precomp.get('Q_sel', np.array([]))
    if len(Q_sel_loaded) > 1:
        # Find two closest parameter values
        dist = np.abs(Q_sel_loaded - Q_new)
        d = np.argsort(dist)
        first = d[0]
        second = d[1]
        
        a = dist[first]
        b = dist[second]
        bottom = a + b
        
        # Set initial coefficients based on parameter distance
        # 新代码
        with torch.no_grad():
            model_gpt.c.zero_()
            model_gpt.c[first] = torch.tensor(b / bottom, dtype=model_gpt.c.dtype, device=model_gpt.c.device)
            model_gpt.c[second] = torch.tensor(a / bottom, dtype=model_gpt.c.dtype, device=model_gpt.c.device)
            
    for step_idx in range(steps):
        try:
            optimizer.zero_grad()
            
            # Calculate PDE residual and boundary losses
            loss_r_gpt = model_gpt.loss_residual_gpt(xy_col_gpt, fx_gpt, fy_gpt) 
            loss_b_gpt = model_gpt.loss_bcs_gpt(Xb_list_gpt, Ub_list_gpt, N_col_offline_loaded)
            loss = loss_r_gpt + loss_b_gpt
            
            # Check for NaN/Inf loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"GPT Q={Q_new:.3f} step {step_idx}: NaN/Inf loss detected ({loss.item() if hasattr(loss, 'item') else loss}). Stopping GPT solve.")
                return None, loss_hist 
                
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            loss_hist.append(loss.item())
            
            # Print progress
            if step_idx == 0 or (step_idx + 1) % print_every_gpt == 0 or step_idx == steps - 1:
                print(f"[GPT Q={Q_new:.3f}] step {step_idx}/{steps-1}, loss={loss.item():.3e}, LR={scheduler.get_last_lr()[0]:.1e}")
        except RuntimeError as e:
            print(f"RuntimeError during GPT step {step_idx} for Q={Q_new:.3f}: {e}")
            return None, loss_hist 
            
    return model_gpt, loss_hist

# === Testing Function (GPT-PINN Test) ===
def test_for_Q(Q_val: float, basis_path: str, lr_gpt_test: float, steps_gpt_test: int, device_in: torch.device):
    """Test GPT-PINN performance for a specific Q value"""
    if not os.path.exists(basis_path):
        print(f"Error: Basis file {basis_path} not found for testing.")
        return None, None, float('nan'), float('nan')
        
    print(f"\n--- Testing GPT-PINN for Q = {Q_val:.4f} ---")
    
    # Solve GPT-PINN for this Q value
    solve_result = solve_gpt_for_Q(
        Q_new=Q_val, basis_path=basis_path, lr=lr_gpt_test, steps=steps_gpt_test, device_in=device_in
    )
    if solve_result is None or solve_result[0] is None: 
        print(f"solve_gpt_for_Q failed for Q={Q_val}")
        return None, None, float('nan'), float('nan')
    model_gpt, _ = solve_result

    # Load precomputed basis data
    precomp = torch.load(basis_path, map_location=device_in)
    P_basis = precomp['P'].to(device_in)
    c_optimized = model_gpt.c.detach()
    
    # Reconstruct solution using optimized coefficients
    U_reconstructed_all_pts = torch.tensordot(P_basis, c_optimized, dims=([1], [0]))
    
    # Load parameters from precomputed data
    N_col_loaded = precomp.get('N_col_offline', 1000)
    N_b_loaded = precomp.get('N_b_offline', 50)
    master_seed_loaded = precomp.get('seed_val', 123) 

    # Generate evaluation points
    colloc_seed_eval = master_seed_loaded
    Q_sel_loaded = precomp.get('Q_sel', np.array([]))
    
    # Find boundary seed offset - try to match the one used during training if possible
    q_match_idx_list = np.where(np.isclose(Q_sel_loaded, Q_val))[0]
    idx_for_boundary_seed = 0 
    if len(q_match_idx_list) > 0: 
        idx_for_boundary_seed = q_match_idx_list[0]
    boundary_seed_offset_eval = master_seed_loaded + 1 + idx_for_boundary_seed
    
    # Generate evaluation data
    xy_c_eval, _, _ = get_collocation_and_forces(
        Q_val, N_col=N_col_loaded, device_in=device_in, seed_value=colloc_seed_eval
    )
    Xb_list_eval, _, _, _, _ = get_boundary_data_separated(
        Q_val, device_in, N_b=N_b_loaded, seed_value_offset=boundary_seed_offset_eval
    )
    X_all_eval_coords = torch.cat([xy_c_eval.detach()] + [xb.detach() for xb in Xb_list_eval], dim=0)

    # Calculate exact solution for comparison
    Ux_exact = u_x_ext(X_all_eval_coords[:,0:1], X_all_eval_coords[:,1:2]).to(device_in)
    Uy_exact = u_y_ext(X_all_eval_coords[:,0:1], X_all_eval_coords[:,1:2], Q_val=Q_val).to(device_in)
    
    # Extract predicted displacements
    Ux_pred = U_reconstructed_all_pts[:,0:1]
    Uy_pred = U_reconstructed_all_pts[:,1:2]
    
    # Calculate relative L2 errors
    norm_ux_exact = torch.linalg.norm(Ux_exact) + 1e-8
    norm_uy_exact = torch.linalg.norm(Uy_exact) + 1e-8
    err_Ux = torch.linalg.norm(Ux_pred-Ux_exact) / norm_ux_exact
    err_Uy = torch.linalg.norm(Uy_pred-Uy_exact) / norm_uy_exact
    
    print(f"  Q={Q_val:.4f}|Rel.L2 err Ux:{err_Ux.item():.3e},Uy:{err_Uy.item():.3e}")
    return U_reconstructed_all_pts.cpu().numpy(), X_all_eval_coords.cpu().numpy(), err_Ux.item(), err_Uy.item()

# === Predict and Plot Function ===
def predict_and_plot_for_Q(
    Q_val_plot: float, 
    basis_path: str,
    gpt_lr: float, 
    gpt_steps: int, 
    device_obj: torch.device,
    master_seed_for_gpt_eval: int
):
    """
    Predict and plot solution fields for a specific Q value using GPT-PINN.
    This implements the combined basis approach with proper initialization of 
    meta-network weights based on parameter distance.
    """
    print(f"\n--- Predicting and Plotting for Q = {Q_val_plot:.4f} ---")

    if not os.path.exists(basis_path):
        print(f"Error: Basis file '{basis_path}' not found.")
        return

    try:
        precomp_data_cpu = torch.load(basis_path, map_location="cpu") 
    except Exception as e:
        print(f"Error loading basis file {basis_path}: {e}")
        return
    
    # Create temporary basis file with only essentials needed for solve_gpt_for_Q
    temp_basis_path_for_gpt = "temp_gpt_basis_predict_plot.pt" 
    precomp_data_for_gpt_solve = {}
    
    required_keys_for_gpt_solve = ['P', 'Rx', 'Ry', 'N_col_offline', 'N_b_offline', 'seed_val', 'Q_sel']
    for key_to_copy in required_keys_for_gpt_solve:
        if key_to_copy in precomp_data_cpu:
            val = precomp_data_cpu[key_to_copy]
            precomp_data_for_gpt_solve[key_to_copy] = val.to(device_obj) if isinstance(val, torch.Tensor) else val
        elif key_to_copy not in ['P', 'Rx', 'Ry', 'Q_sel']: 
            print(f"Warning: Key '{key_to_copy}' not found in precomp_data_cpu for GPT solve. solve_gpt_for_Q might use defaults.")
        elif key_to_copy in ['P', 'Rx', 'Ry']:
             print(f"ERROR: Critical key '{key_to_copy}' not found in precomp_data_cpu for GPT solve. Cannot proceed.")
             return
             
    torch.save(precomp_data_for_gpt_solve, temp_basis_path_for_gpt)

    # Save RNG states
    rng_states = {
        "torch": torch.get_rng_state(), 
        "np": np.random.get_state(), 
        "random": random.getstate()
    }
    if device_obj.type == 'cuda': 
        rng_states["cuda"] = torch.cuda.get_rng_state_all()
    
    # Set seeds for reproducible prediction
    torch.manual_seed(master_seed_for_gpt_eval)
    np.random.seed(master_seed_for_gpt_eval)
    random.seed(master_seed_for_gpt_eval)
    if device_obj.type == 'cuda': 
        torch.cuda.manual_seed_all(master_seed_for_gpt_eval)

    # Solve GPT-PINN for this Q value
    gpt_model_tuple = solve_gpt_for_Q(
        Q_new=Q_val_plot, 
        basis_path=temp_basis_path_for_gpt, 
        lr=gpt_lr, 
        steps=gpt_steps, 
        device_in=device_obj
    )
    
    # Clean up temporary file
    try: 
        os.remove(temp_basis_path_for_gpt)
    except OSError as e: 
        print(f"Warning: could not remove temp file {temp_basis_path_for_gpt}: {e}")
    
    # Restore RNG states
    torch.set_rng_state(rng_states["torch"])
    np.random.set_state(rng_states["np"])
    random.setstate(rng_states["random"])
    if device_obj.type == 'cuda' and "cuda" in rng_states: 
        torch.cuda.set_rng_state_all(rng_states["cuda"])

    if gpt_model_tuple is None or gpt_model_tuple[0] is None:
        print(f"Failed to get GPT coefficients for Q={Q_val_plot}.")
        return
        
    # Get optimized coefficients (meta-network weights)
    optimized_c = gpt_model_tuple[0].c.detach().to(device_obj)

    # Load basis model state dictionaries
    basis_model_state_dicts_cpu = precomp_data_cpu.get('model_state_dicts')
    basis_pinn_layers_cfg = precomp_data_cpu.get('pinn_layers_cfg', [2] + [20]*8 + [5])

    if not basis_model_state_dicts_cpu or len(basis_model_state_dicts_cpu) == 0:
        print("Error: No basis model state_dicts found in precomp file.")
        return
        
    if len(basis_model_state_dicts_cpu) != optimized_c.shape[0]:
        num_states = len(basis_model_state_dicts_cpu)
        print(f"Error: Mismatch in number of basis model state_dicts ({num_states}) and coeffs ({optimized_c.shape[0]}).")
        return

    # Create a regular grid for plotting
    N_plot = 101
    x_space = np.linspace(xmin, xmax, N_plot)
    y_space = np.linspace(ymin, ymax, N_plot)
    X_np_grid, Y_np_grid = np.meshgrid(x_space, y_space)
    xy_plot_flat_np = np.hstack((X_np_grid.flatten()[:,None], Y_np_grid.flatten()[:,None]))
    xy_plot_tensor = torch.tensor(xy_plot_flat_np, dtype=DTYPE, device=device_obj)

    # Calculate predictions from each basis model
    all_basis_preds_on_grid = []
    num_valid_basis_models = 0
    
    for i_basis, state_dict_cpu in enumerate(basis_model_state_dicts_cpu):
        if state_dict_cpu is None:
            print(f"Warning: Basis model {i_basis} has a None state_dict entry. Assuming zero contribution.")
            all_basis_preds_on_grid.append(torch.zeros(xy_plot_tensor.shape[0], 5, dtype=DTYPE, device=device_obj))
            continue
        
        # Create basis model instance and load weights
        basis_model_instance = PINN(layers_config=basis_pinn_layers_cfg, precomp=None).to(device_obj)
        state_dict_device = {k: v.to(device_obj) for k,v in state_dict_cpu.items()}
        
        try:
            basis_model_instance.load_state_dict(state_dict_device)
        except RuntimeError as e_load:
            print(f"Error loading state_dict for basis {i_basis}: {e_load}. Skipping this basis.")
            all_basis_preds_on_grid.append(torch.zeros(xy_plot_tensor.shape[0], 5, dtype=DTYPE, device=device_obj))
            continue

        # Generate predictions from this basis model
        basis_model_instance.eval()
        with torch.no_grad():
            preds_one_basis_t = basis_model_instance(xy_plot_tensor)
        all_basis_preds_on_grid.append(preds_one_basis_t)
        num_valid_basis_models += 1
    
    if num_valid_basis_models == 0:
        print("No valid basis models could be loaded. Cannot create combined prediction.")
        return

    # Combine predictions using optimized coefficients - this is the key GPT-PINN operation
    stacked_basis_preds = torch.stack(all_basis_preds_on_grid, dim=1)
    final_predictions_on_grid_t = torch.einsum('pbf,b->pf', stacked_basis_preds, optimized_c)
    final_predictions_on_grid_np = final_predictions_on_grid_t.cpu().numpy()

    # Plot the final solution fields
    plot_solution_fields_matching_reference(
        model_outputs_nn=final_predictions_on_grid_np,
        X_plot_coords_np=X_np_grid, 
        Y_plot_coords_np=Y_np_grid,
        Q_val_plot=Q_val_plot, 
        config_case_name=f"GPT-PINN Prediction (Q={Q_val_plot:.2f})",
        prefix=f"gpt_pred_Q{Q_val_plot:.2f}"
    )

def plot_solution_fields_matching_reference(
    model_outputs_nn: np.ndarray, X_plot_coords_np: np.ndarray, 
    Y_plot_coords_np: np.ndarray, Q_val_plot: float,            
    config_case_name: str = "Validation", prefix: str = "validation"
):
    """Plot predicted solution fields and compare with analytical solution"""
    # Reshape predicted fields to match coordinates grid
    target_shape = X_plot_coords_np.shape 
    Ux_nn = model_outputs_nn[:, 0].reshape(target_shape)
    Uy_nn = model_outputs_nn[:, 1].reshape(target_shape)
    Sxx_nn = model_outputs_nn[:, 2].reshape(target_shape)
    Syy_nn = model_outputs_nn[:, 3].reshape(target_shape)
    Sxy_nn = model_outputs_nn[:, 4].reshape(target_shape)
    
    # Get exact solution for comparison
    Ux_exact_np = u_x_ext_np_local(X_plot_coords_np, Y_plot_coords_np)
    Uy_exact_np = u_y_ext_np_local(X_plot_coords_np, Y_plot_coords_np, Q_val_plot)

    # Plot displacements (NN vs exact) and errors
    fig_disp, axes_disp = plt.subplots(2, 3, figsize=(18, 10))
    fig_disp.suptitle(f'Predictions & Errors - {config_case_name} (Q={Q_val_plot:.2f})', fontsize=16)
    
    # u_x plots
    im = axes_disp[0,0].pcolormesh(X_plot_coords_np, Y_plot_coords_np, Ux_nn, cmap='jet', shading='gouraud')
    fig_disp.colorbar(im, ax=axes_disp[0,0])
    axes_disp[0,0].set_title('$u_x$ (NN)')
    
    im = axes_disp[0,1].pcolormesh(X_plot_coords_np, Y_plot_coords_np, Ux_exact_np, cmap='jet', shading='gouraud')
    fig_disp.colorbar(im, ax=axes_disp[0,1])
    axes_disp[0,1].set_title('$u_x$ (Exact)')
    
    im = axes_disp[0,2].pcolormesh(X_plot_coords_np, Y_plot_coords_np, np.abs(Ux_nn - Ux_exact_np), cmap='hot', shading='gouraud')
    fig_disp.colorbar(im, ax=axes_disp[0,2])
    axes_disp[0,2].set_title('Error $u_x$')
    
    # u_y plots
    im = axes_disp[1,0].pcolormesh(X_plot_coords_np, Y_plot_coords_np, Uy_nn, cmap='jet', shading='gouraud')
    fig_disp.colorbar(im, ax=axes_disp[1,0])
    axes_disp[1,0].set_title('$u_y$ (NN)')
    
    im = axes_disp[1,1].pcolormesh(X_plot_coords_np, Y_plot_coords_np, Uy_exact_np, cmap='jet', shading='gouraud')
    fig_disp.colorbar(im, ax=axes_disp[1,1])
    axes_disp[1,1].set_title('$u_y$ (Exact)')
    
    im = axes_disp[1,2].pcolormesh(X_plot_coords_np, Y_plot_coords_np, np.abs(Uy_nn - Uy_exact_np), cmap='hot', shading='gouraud')
    fig_disp.colorbar(im, ax=axes_disp[1,2])
    axes_disp[1,2].set_title('Error $u_y$')
    
    # Format all plots
    for ax_row in axes_disp:
        for ax_curr in ax_row: 
            ax_curr.axis('square')
            ax_curr.set_xlabel('x')
            ax_curr.set_ylabel('y') 
            
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    disp_plot_path = f"{prefix}_displacements_Q_{Q_val_plot:.2f}.png"
    plt.savefig(disp_plot_path, dpi=300)
    print(f"Displacement plot saved to {disp_plot_path}")
    plt.show(block=False)

    # Plot stress fields
    fig_stress, axes_stress_list = plt.subplots(1, 3, figsize=(18, 5)) 
    fig_stress.suptitle(f'Predicted Stresses - {config_case_name} (Q={Q_val_plot:.2f})', fontsize=16)
    
    im = axes_stress_list[0].pcolormesh(X_plot_coords_np, Y_plot_coords_np, Sxx_nn, cmap='viridis', shading='gouraud')
    fig_stress.colorbar(im, ax=axes_stress_list[0])
    axes_stress_list[0].set_title('$\\sigma_{xx}$ (NN)')
    
    im = axes_stress_list[1].pcolormesh(X_plot_coords_np, Y_plot_coords_np, Syy_nn, cmap='viridis', shading='gouraud')
    fig_stress.colorbar(im, ax=axes_stress_list[1])
    axes_stress_list[1].set_title('$\\sigma_{yy}$ (NN)')
    
    im = axes_stress_list[2].pcolormesh(X_plot_coords_np, Y_plot_coords_np, Sxy_nn, cmap='viridis', shading='gouraud')
    fig_stress.colorbar(im, ax=axes_stress_list[2])
    axes_stress_list[2].set_title('$\\sigma_{xy}$ (NN)')
    
    # Format all plots
    for ax_curr in axes_stress_list: 
        ax_curr.axis('square')
        ax_curr.set_xlabel('x')
        ax_curr.set_ylabel('y') 
        
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    stress_plot_path = f"{prefix}_stresses_Q_{Q_val_plot:.2f}.png"
    plt.savefig(stress_plot_path, dpi=300)
    print(f"Stress plot saved to {stress_plot_path}")
    plt.show(block=True)

def plot_coefficients(coeffs_data_path: str):
    """Plot GPT-PINN coefficients (meta-network weights) vs Q values"""
    try: 
        coeffs_dict_all_q = torch.load(coeffs_data_path, map_location="cpu")
    except FileNotFoundError: 
        print(f"Coeff file {coeffs_data_path} not found.")
        return
        
    if not isinstance(coeffs_dict_all_q, dict) or not coeffs_dict_all_q: 
        print(f"Coeff data invalid.")
        return
        
    # Sort Q values for consistent plotting
    Qs_sorted = sorted(coeffs_dict_all_q.keys())
    if not Qs_sorted: 
        print("No Q values found in coefficients dict.")
        return
    
    # Determine number of basis functions
    first_c_val = coeffs_dict_all_q[Qs_sorted[0]]
    num_basis = 0
    if hasattr(first_c_val, 'shape') and len(first_c_val.shape) > 0:
         num_basis = first_c_val.shape[0]
    elif isinstance(first_c_val, (torch.Tensor, np.ndarray)) and first_c_val.ndim == 0: 
         num_basis = 1 
    elif isinstance(first_c_val, (int, float)):
         num_basis = 1
    if num_basis == 0: 
        if hasattr(first_c_val, 'numel') and first_c_val.numel() == 1: 
            num_basis = 1
        elif hasattr(first_c_val, 'size') and hasattr(first_c_val, 'ndim') and first_c_val.ndim == 0 and first_c_val.size == 1: 
            num_basis = 1
    if num_basis == 0: 
        print("Number of basis functions is zero. Skipping coefficient plot.")
        return
        
    # Extract and process coefficients data
    Cs_list = []
    processed_Qs = [] 
    for q_val in Qs_sorted:
        c_tensor = coeffs_dict_all_q[q_val]
        if isinstance(c_tensor, torch.Tensor): 
            c_vec = c_tensor.cpu().numpy().squeeze()
        elif isinstance(c_tensor, np.ndarray): 
            c_vec = c_tensor.squeeze()
        else: 
            c_vec = np.array([c_tensor]) if num_basis == 1 else None
        
        if c_vec is None: 
            continue
        if c_vec.ndim == 0: 
            c_vec = c_vec.reshape(1) 
        if c_vec.ndim == 1 and len(c_vec) == num_basis: 
            Cs_list.append(c_vec)
            processed_Qs.append(q_val) 
            
    if not Cs_list: 
        print("No valid coefficients to plot after shape check.")
        return
        
    # Plot coefficients vs Q values
    Cs_np_array = np.stack(Cs_list, axis=0)
    plt.figure(figsize=(10,6))
    for i in range(Cs_np_array.shape[1]): 
        plt.plot(processed_Qs, Cs_np_array[:,i], marker='o', ls='-', ms=4, label=f"$c_{i}$")
    
    plt.xlabel("$Q$")
    plt.ylabel("$c_i$")
    plt.title(f"GPT Coefficients vs $Q$ ({Cs_np_array.shape[1]} basis)")
    plt.legend(bbox_to_anchor=(1.02,1), loc='upper left')
    plt.grid(True)
    plt.tight_layout(rect=[0,0,0.85,1])
    plt.savefig("gpt_coeffs_vs_Q.png", dpi=300)
    print(f"Coeff plot saved.")
    plt.show(block=False)

# === Training Pipeline Orchestrator ===
def train_pipeline_orchestrator(
    num_basis: int, offline_epochs: int, offline_base_lr: float,
    offline_pinn_cfg: list, offline_N_col: int, offline_N_b: int,
    offline_disable_scaling: bool, offline_dropout: float,
    gpt_lr: float, gpt_steps: int, basis_file_path: str,
    coeffs_dir: str, all_coeffs_file: str,
    device_obj: torch.device, master_seed: int,
    Q_plot_after_offline_target: float = None 
):
    """
    Orchestrate the entire GPT-PINN training pipeline:
    1. Offline precomputation of basis functions
    2. Online GPT-PINN optimization for parameter sweep
    """
    # Create directory for saving coefficients
    os.makedirs(coeffs_dir, exist_ok=True)
    force_recompute_offline = False
    
    # Handle special case for single basis function plotting
    if Q_plot_after_offline_target is not None and num_basis == 1:
        force_recompute_offline = True 
        if os.path.exists(basis_file_path):
            print(f"INFO: Deleting existing basis file '{basis_file_path}' to ensure fresh computation for plotting Q={Q_plot_after_offline_target}.")
            try: 
                os.remove(basis_file_path)
            except OSError as e: 
                print(f"Error removing file: {e}")

    # Step 1: Offline precomputation (if needed)
    if not os.path.exists(basis_file_path) or force_recompute_offline:
        print(f"=== Starting Offline Precomputation ===")
        offline_data = offline_precomp( 
            num_basis_to_generate=num_basis, 
            pinn_epochs=offline_epochs, 
            base_lr_offline=offline_base_lr,
            pinn_layers_cfg=offline_pinn_cfg, 
            disable_scaling_offline=offline_disable_scaling,
            dropout_offline=offline_dropout,
            loss_weights_offline=OFFLINE_PINN_LOSS_WEIGHTS,
            N_col_offline=offline_N_col, 
            N_b_offline=offline_N_b,
            device_str=str(device_obj), 
            basis_path=basis_file_path, 
            seed_val=master_seed,
            Q_plot_target=Q_plot_after_offline_target 
        )
        if offline_data is None: 
            print("Offline precomputation did not produce a basis. Exiting train pipeline.")
            return
    else: 
        print(f"--- Loading existing basis from {basis_file_path} ---")

    # Step 2: Online GPT-PINN training for parameter sweep
    if os.path.exists(basis_file_path):
        print(f"\n=== Starting Online GPT-PINN Training ===")
        gpt_coeffs_all_q = {}
        q_train_np_local = q_train.cpu().numpy() 
        
        # Process each Q value in the parameter sweep
        for q_online in q_train_np_local: 
            print(f"\n--- Solving GPT for Q = {q_online:.4f} ---")
            gpt_model_tuple = solve_gpt_for_Q( 
                Q_new=q_online,
                basis_path=basis_file_path,
                lr=gpt_lr,
                steps=gpt_steps,
                device_in=device_obj
            )
            
            # Store coefficients if successful
            if gpt_model_tuple is not None and gpt_model_tuple[0] is not None: 
                gpt_model = gpt_model_tuple[0]
                gpt_coeffs_all_q[q_online] = gpt_model.c.detach().cpu()
                # Save individual coefficient files
                torch.save(gpt_model.c.detach().cpu(), os.path.join(coeffs_dir, f"coeffs_Q_{q_online:.4f}.pt"))
            else: 
                print(f"GPT solve failed for Q={q_online:.4f}, skipping.")
                
        # Save combined coefficients file
        if gpt_coeffs_all_q: 
            torch.save(gpt_coeffs_all_q, os.path.join(coeffs_dir, all_coeffs_file))
            print(f"\nAll GPT coeffs saved.")
        else: 
            print("\nNo GPT coefficients were generated.")
    else:
        print("Basis file not found. Skipping online GPT training.")
        
    print("Training pipeline finished.")

# === Inspect Precomputation (Loads Saved Basis Models for Plotting) ===
def inspect_precomp_basis_function(Q_target:float, basis_path:str, device_obj:torch.device):
    """
    Load and visualize a single basis function (PINN) for Q value closest to target.
    Useful for inspecting the basis functions used by GPT-PINN.
    """
    print(f"\n--- Inspecting Basis (from saved model state) for Q closest to {Q_target:.4f} ---")
    
    # Load precomputed data
    try: 
        precomp_data = torch.load(basis_path, map_location="cpu") 
    except FileNotFoundError: 
        print(f"Basis file {basis_path} not found.")
        return
    
    # Extract required data from precomputed file
    P_all_cpu = precomp_data.get('P')
    Q_sel_from_file = precomp_data.get('Q_sel')
    model_state_dicts_cpu = precomp_data.get('model_state_dicts')
    pinn_layers_cfg_saved = precomp_data.get('pinn_layers_cfg')

    if P_all_cpu is None or Q_sel_from_file is None or model_state_dicts_cpu is None or pinn_layers_cfg_saved is None:
        print("Essential data (P, Q_sel, model_state_dicts, or pinn_layers_cfg) not found in basis file. Cannot inspect.")
        return
    
    # Get parameter details from precomputed data
    N_col_offline = precomp_data.get('N_col_offline', 1000)
    N_b_offline = precomp_data.get('N_b_offline', 50)
    master_seed_for_basis = precomp_data.get('seed_val', 123) 
    
    # Find Q value closest to target
    q_diff = np.abs(Q_sel_from_file - Q_target)
    idx_matching_q = np.argmin(q_diff) 
    Q_actual_for_basis = Q_sel_from_file[idx_matching_q]
    print(f"Target Q for inspection: {Q_target:.4f}")
    print(f"Actual Q for this basis function (from Q_sel): Q={Q_actual_for_basis:.4f} at index {idx_matching_q}")

    # Check if we have a valid model state for this Q value
    if not (0 <= idx_matching_q < len(model_state_dicts_cpu)) or model_state_dicts_cpu[idx_matching_q] is None:
        print(f"No valid saved model state found for Q={Q_actual_for_basis:.4f} (index {idx_matching_q}). Cannot plot from saved model state.")
        return

    # Load the saved model
    inspect_model = PINN(
        layers_config=pinn_layers_cfg_saved, 
        precomp=None, 
        disable_scaling_flag=False
    ).to(device_obj)
    
    state_dict_to_load = {k: v.to(device_obj) for k, v in model_state_dicts_cpu[idx_matching_q].items()}
    inspect_model.load_state_dict(state_dict_to_load)
    inspect_model.eval()
    print(f"Loaded saved model state for Q={Q_actual_for_basis:.4f} for pcolormesh plotting.")
    
    # Generate a regular grid for visualization
    N_plot = 101 
    x_space = np.linspace(xmin, xmax, N_plot)
    y_space = np.linspace(ymin, ymax, N_plot)
    X_np_grid, Y_np_grid = np.meshgrid(x_space, y_space)
    xy_plot_flat = np.hstack((X_np_grid.flatten()[:,None], Y_np_grid.flatten()[:,None]))
    xy_plot_tensor = torch.tensor(xy_plot_flat, dtype=DTYPE, device=device_obj)
    
    # Generate predictions
    with torch.no_grad(): 
        predictions_on_grid_t = inspect_model(xy_plot_tensor)
    predictions_on_grid_np = predictions_on_grid_t.cpu().numpy()

    # Plot the solution fields
    plot_solution_fields_matching_reference(
        model_outputs_nn=predictions_on_grid_np, 
        X_plot_coords_np=X_np_grid, 
        Y_plot_coords_np=Y_np_grid,
        Q_val_plot=Q_actual_for_basis, 
        config_case_name=f"Inspect Saved Basis (Q={Q_actual_for_basis:.2f})",
        prefix=f"inspect_pcolormesh_Q{Q_actual_for_basis:.2f}_idx{idx_matching_q}"
    )

    # Calculate L2 error for this basis function using precomputed P tensor
    P_all_device = P_all_cpu.to(device_obj)
    P_one_basis_on_X_all = P_all_device[:, idx_matching_q, :] 
    
    # Generate reference data points matching those used in precomputation
    coll_seed_insp_orig = master_seed_for_basis 
    bound_seed_off_insp_orig = master_seed_for_basis + 1 + idx_matching_q
    xy_c_orig_eval, _, _ = get_collocation_and_forces(
        Q_actual_for_basis, N_col=N_col_offline, seed_value=coll_seed_insp_orig, device_in=device_obj
    )
    Xb_list_orig_eval, _, _, _, _ = get_boundary_data_separated(
        Q_actual_for_basis, device_obj, N_b=N_b_offline, seed_value_offset=bound_seed_off_insp_orig
    )
    X_all_orig_eval_coords = torch.cat([xy_c_orig_eval.detach()] + [xb.detach() for xb in Xb_list_orig_eval], dim=0)

    # Calculate L2 error if shapes match
    if X_all_orig_eval_coords.shape[0] == P_one_basis_on_X_all.shape[0]:
        Ux_ex_orig = u_x_ext(X_all_orig_eval_coords[:,0:1], X_all_orig_eval_coords[:,1:2]).to(device_obj)
        Uy_ex_orig = u_y_ext(X_all_orig_eval_coords[:,0:1], X_all_orig_eval_coords[:,1:2], Q_val=Q_actual_for_basis).to(device_obj)
        
        Ux_pr_orig = P_one_basis_on_X_all[:,0:1]
        Uy_pr_orig = P_one_basis_on_X_all[:,1:2]
        
        norm_ux_ex_orig = torch.linalg.norm(Ux_ex_orig) + 1e-8 
        norm_uy_ex_orig = torch.linalg.norm(Uy_ex_orig) + 1e-8
        
        err_Ux_orig = torch.linalg.norm(Ux_pr_orig - Ux_ex_orig) / norm_ux_ex_orig
        err_Uy_orig = torch.linalg.norm(Uy_pr_orig - Uy_ex_orig) / norm_uy_ex_orig
        
        print(f"  L2 Error of stored P-tensor data (on its original eval grid): Ux={err_Ux_orig.item():.3e}, Uy={err_Uy_orig.item():.3e}")
    else: 
        print(f"Warning: Shape mismatch for stored basis L2 error calc.")



# === Main Script Logic ===
def parse_args():
    """Parse command line arguments"""
    p = argparse.ArgumentParser(description="Parametric PINN (GPT-PINN Architecture)")
    
    # Mode selection
    p.add_argument("--mode", choices=["train", "test", "plot", "inspect", "predict_and_plot"], default="train") 
    
    # Offline precomputation parameters
    p.add_argument("--num_basis", type=int, default=20)
    p.add_argument("--offline_epochs", type=int, default=3000) 
    p.add_argument("--offline_lr", type=float, default=1e-2)
    p.add_argument("--offline_arch_depth", type=int, default=8)
    p.add_argument("--offline_arch_width", type=int, default=20)
    p.add_argument("--offline_N_col", type=int, default=1000)
    p.add_argument("--offline_N_b", type=int, default=50)
    p.add_argument("--offline_disable_scaling", action='store_true', help="Disable input scaling for offline PINNs.")
    p.add_argument("--offline_dropout", type=float, default=0.0, help="Dropout probability for offline PINNs.")
    
    # Online GPT-PINN parameters
    p.add_argument("--gpt_lr", type=float, default=1e-3)
    p.add_argument("--gpt_steps", type=int, default=500)
    
    # File paths
    p.add_argument("--basis_path", type=str, default="parametric_basis_with_states.pt") 
    p.add_argument("--coeffs_dir", type=str, default="ckpts_coeffs_parametric")
    
    # Testing/plotting parameters
    p.add_argument("--Q_test_val", type=float, default=4.0)
    p.add_argument("--Q_plot_after_offline", type=float, default=None, 
                   help="Q value for immediate plot after its offline PINN training.")
    p.add_argument("--Q_predict_val", type=float, default=None, 
                   help="Q value for 'predict_and_plot' mode.")
    
    # Device and seed settings
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=123)
    
    return p.parse_args()

def main():
    """Main function to coordinate all operations based on command line arguments"""
    # Parse arguments
    args = parse_args()
    master_seed = args.seed
    
    # Set random seeds for reproducibility
    np.random.seed(master_seed)
    torch.manual_seed(master_seed)
    random.seed(master_seed)
    
    # Set device (GPU or CPU)
    str_dev_requested = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    dev_obj = None
    
    if str_dev_requested == "cuda" and torch.cuda.is_available():
        dev_obj = torch.device(f"cuda:{torch.cuda.current_device()}")
    elif str_dev_requested.startswith("cuda:") and torch.cuda.is_available():
        try:
            dev_obj = torch.device(str_dev_requested)
        except Exception as e:
            print(f"Warning: Requested CUDA device {str_dev_requested} not available ({e}). Defaulting.")
            if torch.cuda.is_available(): 
                dev_obj = torch.device(f"cuda:{torch.cuda.current_device()}")
            else: 
                dev_obj = torch.device("cpu")
    else: 
        dev_obj = torch.device("cpu")

    if dev_obj.type == 'cuda': 
        torch.cuda.manual_seed_all(master_seed)
        
    print(f"Device: {dev_obj}, Seed: {master_seed}")
    
    # Initialize global tensors for material properties
    global lam_tensor, mu_tensor 
    if lam_tensor is None or lam_tensor.device != dev_obj:
        lam_tensor = torch.tensor(lam_const_val, dtype=DTYPE, device=dev_obj)
    if mu_tensor is None or mu_tensor.device != dev_obj:
        mu_tensor = torch.tensor(mu_const_val, dtype=DTYPE, device=dev_obj)

    # Neural network configuration and file paths
    offline_cfg = [2] + [args.offline_arch_width] * args.offline_arch_depth + [5]
    all_coeffs_filename = "all_gpt_coeffs_parametric.pt" 
    all_coeffs_fp = os.path.join(args.coeffs_dir, all_coeffs_filename)

    # Execute requested mode
    if args.mode == "train":
        # Full training pipeline - offline precomputation and online GPT-PINN
        train_pipeline_orchestrator(
            num_basis=args.num_basis,
            offline_epochs=args.offline_epochs,
            offline_base_lr=args.offline_lr,
            offline_pinn_cfg=offline_cfg,
            offline_N_col=args.offline_N_col,
            offline_N_b=args.offline_N_b,
            offline_disable_scaling=args.offline_disable_scaling,
            offline_dropout=args.offline_dropout,
            gpt_lr=args.gpt_lr,
            gpt_steps=args.gpt_steps,
            basis_file_path=args.basis_path,
            coeffs_dir=args.coeffs_dir,
            all_coeffs_file=all_coeffs_filename, 
            device_obj=dev_obj,
            master_seed=master_seed,
            Q_plot_after_offline_target=args.Q_plot_after_offline
        )
    elif args.mode == "test":
        # Test GPT-PINN on a specific Q value
        _ = test_for_Q(
            Q_val=args.Q_test_val,
            basis_path=args.basis_path,
            lr_gpt_test=args.gpt_lr,
            steps_gpt_test=args.gpt_steps,
            device_in=dev_obj
        )
        print(f"INFO: Test mode computed errors on non-regular grid. For pcolormesh plot of Q={args.Q_test_val}, "
              f"use '--mode predict_and_plot --Q_predict_val {args.Q_test_val}'.")

    elif args.mode == "predict_and_plot": 
        # Predict and plot solution fields for a specific Q value
        q_to_predict = args.Q_predict_val if args.Q_predict_val is not None else args.Q_test_val
        if q_to_predict is None: 
             raise ValueError("No Q value specified for prediction. Use --Q_predict_val or ensure --Q_test_val has a default.")
        print(f"INFO: Using Q={q_to_predict:.4f} for prediction and plotting.")
        
        predict_and_plot_for_Q(
            Q_val_plot=q_to_predict,
            basis_path=args.basis_path, 
            gpt_lr=args.gpt_lr, 
            gpt_steps=args.gpt_steps, 
            device_obj=dev_obj,
            master_seed_for_gpt_eval=args.seed + 777 
        )
    elif args.mode == "plot": 
        # Plot GPT-PINN coefficients
        plot_coefficients(all_coeffs_fp)
        # For field plots, guide user to predict_and_plot or inspect
        print(f"\nINFO: Coefficients plotted. To plot solution fields for a specific Q (e.g., Q={args.Q_test_val}) "
              f"using pcolormesh style (showing GPT combined solution):")
        print(f"      Use: '--mode predict_and_plot --Q_predict_val {args.Q_test_val}'")
        print(f"      This requires the basis file ('{args.basis_path}') to contain model state_dicts.")
        print(f"\n      To inspect a single saved basis function (not the GPT combined solution):")
        print(f"      Use: '--mode inspect --Q_test_val {args.Q_test_val}'")

    elif args.mode == "inspect":
        # Inspect a specific basis function
        inspect_precomp_basis_function(
            Q_target=args.Q_test_val,
            basis_path=args.basis_path,
            device_obj=dev_obj
        )
    else: 
        raise ValueError(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()
