import torch
import torch.nn as nn
from torch.optim import LBFGS
from tqdm import tqdm
import numpy as np

from . import data_loader 
from .model import PINNsformer
from .physics import HeatEquationPINNLoss
from .data_loader import get_n_params
from .data_loader import get_pinn_domain_points

def init_weights(m):
    """
    Initializes model weights using Xavier uniform distribution.
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

def run(args):
    """
    Main training function.
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.sampler == 'lhs':
        print("Preparing training data points using Latin Hypercube Sampling...")
        points_dict = data_loader.get_pinn_domain_points(args)
    else:
        print("Preparing training data points using grid sampling...")
        points_dict = data_loader.get_separated_points(args)

    res_tensor = points_dict['res']
    ic_tensor = points_dict['ic']
    bc_tensors = points_dict['bc']
    
    res_tensor = data_loader.make_time_sequence(res_tensor, args.time_steps, args.step_size)
    ic_tensor = data_loader.make_time_sequence(ic_tensor, args.time_steps, args.step_size)
    for key in bc_tensors:
        bc_tensors[key] = data_loader.make_time_sequence(bc_tensors[key], args.time_steps, args.step_size)
    
    x_star, y_star, z_star, t_star, T_star_true = data_loader.get_evaluation_data(args)
    x_star, y_star, z_star, t_star = x_star.unsqueeze(1), y_star.unsqueeze(1), z_star.unsqueeze(1), t_star.unsqueeze(1)
    print("Data preparation complete.")

    model = PINNsformer(
        d_out=args.d_out, d_hidden=args.d_hidden, d_model=args.d_model,
        N=args.n_layers, heads=args.n_heads
    ).to(device)
    model.apply(init_weights)
    print(f"Model initialized with {get_n_params(model)} parameters.")
    
    loss_computer = HeatEquationPINNLoss(args)
    optimizer = LBFGS(model.parameters(), line_search_fn='strong_wolfe', lr=args.learning_rate)

    model.train()
    for epoch in range(args.epochs):
        
        def closure():
            optimizer.zero_grad()
            total_loss, _, _, _, _ = loss_computer.compute_loss(
                model, res_tensor, ic_tensor, bc_tensors
            )
            total_loss.backward(retain_graph=True)
            return total_loss

        optimizer.step(closure)
        
        final_loss, pde, bc, ic, reg = loss_computer.compute_loss(
            model, res_tensor, ic_tensor, bc_tensors
        )
        
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        print(f"Loss PDE: {pde:.6f}, Loss_BC: {bc:.6f}, Loss_IC: {ic:.6f}")
        print(f"Total Train Loss: {final_loss.item():.6f}")

        model.eval()
        with torch.no_grad():
            T_pred = model(x_star, y_star, z_star, t_star).squeeze().cpu().numpy()
        model.train()
        
        rl2_error = np.linalg.norm(T_star_true - T_pred) / np.linalg.norm(T_star_true)
        print(f"Relative L2 error: {rl2_error:.6f}")

    print(f"\nTraining complete. Saving model to {args.model_save_path}")
    torch.save(model.state_dict(), args.model_save_path)
