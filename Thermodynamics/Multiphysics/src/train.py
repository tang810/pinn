import os
import torch
from torch.optim import LBFGS

from .model import PINNsformer
from .data_loader import get_training_data, get_evaluation_data
from .physics import ThermoElasticPINNLoss


def train_model(args):
    # device: prefer args.device if provided, else auto
    if hasattr(args, "device") and args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get training data
    res_seq, bc_seq, bc_normals, bc_tag, ic_seq, ic_T = get_training_data(args, device)

    # Initialize model
    model = PINNsformer(
        d_model=args.d_model,
        d_hidden=args.d_hidden,
        N=args.N,
        heads=args.heads
    ).to(device)

    # Initialize physics loss
    loss_fn = ThermoElasticPINNLoss(
        rho=args.rho, Cp=args.Cp, k=args.k,
        E=args.E, nu=args.nu, alpha=args.alpha, T0=args.T0,
        w_heat=args.w_heat, w_elastic=args.w_elastic, w_ic=args.w_ic, w_reg=args.w_reg,
        w_bc=getattr(args, "w_bc", 1.0),
        bc_outer_type=getattr(args, "bc_outer_type", "dirichlet"),
        bc_outer_value=getattr(args, "bc_outer_value", args.T0),
        bc_hole_type=getattr(args, "bc_hole_type", "neumann"),
        bc_hole_value=getattr(args, "bc_hole_value", 0.0),
    )

    # Optimizer (LBFGS)
    optimizer = LBFGS(
        model.parameters(),
        lr=args.lr,
        max_iter=20,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    # Load evaluation data for monitoring (optional)
    eval_inputs = None
    T_eval_tensor = None

    if getattr(args, "data_path", None) and os.path.exists(args.data_path):
        x_eval, y_eval, z_eval, t_eval, T_eval = get_evaluation_data(args.data_path)

        eval_inputs = torch.stack(
            [
                torch.tensor(x_eval, dtype=torch.float32),
                torch.tensor(y_eval, dtype=torch.float32),
                torch.tensor(z_eval, dtype=torch.float32),
                torch.tensor(t_eval, dtype=torch.float32),
            ],
            dim=1,
        ).to(device)

        # make it a time sequence input: [N, time_steps, 4]
        eval_inputs = eval_inputs.unsqueeze(1).repeat(1, args.time_steps, 1)
        T_eval_tensor = torch.tensor(T_eval, dtype=torch.float32).to(device)
    else:
        print(
            f"[WARN] Evaluation data not found: {getattr(args, 'data_path', None)}. "
            "Skip evaluation during training."
        )

    def closure():
        optimizer.zero_grad()
        total_loss, loss_heat, loss_elastic, loss_ic, loss_reg, loss_bc = loss_fn.compute_loss(
            model, res_seq, bc_seq, bc_normals, bc_tag, ic_seq, ic_T
        )
        total_loss.backward(retain_graph=True)
        return total_loss

    # Training loop
    for epoch in range(args.epochs):
        loss = optimizer.step(closure)

        # Optional evaluation
        if eval_inputs is not None:
            with torch.no_grad():
                T_pred = model(eval_inputs)[:, 0, 0]
                rel_l2 = torch.norm(T_pred - T_eval_tensor) / torch.norm(T_eval_tensor)
            print(
                f"Epoch {epoch + 1}/{args.epochs}, "
                f"Loss: {loss.item():.6f}, Rel L2: {rel_l2.item():.6f}"
            )
        else:
            print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {loss.item():.6f}")

    # Save the model
    model_path = os.path.join(args.output_dir, "pinnsformer_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")