import argparse
import os
import time

import torch
import torch.optim as optim

from src.data import generate_test_data, generate_training_data
from src.loss import compute_loss
from src.pinn_model import PINN
from src.train import save_model
from src.visual import (
    create_frequency_animation,
    create_wave_propagation_animation,
    plot_frequency_heatmap,
    plot_losses,
    plot_node_frequencies,
    plot_power_comparison,
)


def parse_args():
    parser = argparse.ArgumentParser(description="PINN for power system simulation")
    parser.add_argument("--mode", choices=["quick", "train"], default="quick")
    parser.add_argument("--data", choices=["simul", "load"], default="simul")
    parser.add_argument("--data_path", type=str, default="data/simul_placeholder.txt")
    parser.add_argument("--num_nodes", type=int, default=10)
    parser.add_argument("--num_epochs", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--t_max", type=float, default=1.0)
    parser.add_argument("--num_time_steps", type=int, default=200)
    parser.add_argument("--model_dir", type=str, default="model")
    parser.add_argument("--result_dir", type=str, default="results")
    parser.add_argument("--model_name", type=str, default="power_system_pinn_model.pt")
    return parser.parse_args()


def predict(model, t_flat, x_flat):
    model.eval()
    with torch.no_grad():
        delta_f, delta_p_e, delta_p_m = model(t_flat, x_flat)
    return (
        delta_f.cpu().numpy(),
        delta_p_e.cpu().numpy(),
        delta_p_m.cpu().numpy(),
    )


def train_model(num_nodes=10, num_epochs=3000, batch_size=2048, t_max=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PINN(num_nodes=num_nodes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.5, patience=500
    )

    losses = []
    start_time = time.time()
    print("Starting training...")

    for epoch in range(num_epochs):
        t, x = generate_training_data(batch_size, num_nodes, t_max)
        total_loss, eq1_loss, eq2_loss, eq3_loss, boundary_loss = compute_loss(
            model, t, x, num_nodes
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss)

        if (epoch + 1) % 100 == 0:
            loss_values = {
                "Total Loss": total_loss.item(),
                "Rotor Motion Equation": eq1_loss.item(),
                "Power Balance": eq2_loss.item(),
                "Power Transfer Equation": eq3_loss.item(),
                "Boundary Conditions": boundary_loss.item(),
            }
            losses.append(loss_values)
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Total={loss_values['Total Loss']:.6f}"
            )

    elapsed = time.time() - start_time
    print(f"Training completed. elapsed={elapsed:.2f}s")
    return model, losses


def export_results(model, args):
    _, _, t_flat, x_flat, t_grid, x_grid = generate_test_data(
        num_nodes=args.num_nodes,
        t_max=args.t_max,
        num_time_steps=args.num_time_steps,
    )
    delta_f, p_e, p_m = predict(model, t_flat, x_flat)

    plot_frequency_heatmap(
        delta_f,
        t_grid,
        x_grid,
        args.num_nodes,
        args.t_max,
        save_path=os.path.join(args.result_dir, "frequency_heatmap.png"),
    )
    plot_node_frequencies(
        delta_f,
        t_grid,
        x_grid,
        save_path=os.path.join(args.result_dir, "node_frequencies.png"),
    )
    for node in [0, max(0, args.num_nodes // 2 - 1), args.num_nodes - 1]:
        plot_power_comparison(
            p_e,
            p_m,
            t_grid,
            x_grid,
            node_idx=node,
            save_path=os.path.join(args.result_dir, f"power_comparison_node_{node}.png"),
        )
    create_frequency_animation(
        delta_f,
        t_grid,
        x_grid,
        args.num_nodes,
        save_path=os.path.join(args.result_dir, "frequency_animation.gif"),
    )
    create_wave_propagation_animation(
        delta_f,
        t_grid,
        x_grid,
        args.num_nodes,
        save_path=os.path.join(args.result_dir, "wave_propagation.gif"),
    )


def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    model_path = os.path.join(args.model_dir, args.model_name)
    print("=" * 60)
    print(f"mode: {args.mode}")
    print(f"data: {args.data}")
    print(f"data_path: {args.data_path}")
    print(f"model_path: {model_path}")
    print(f"result_dir: {args.result_dir}")
    print("=" * 60)

    if args.data == "load":
        print("[info] load mode placeholder: using built-in simulation path for now.")

    if args.mode == "train":
        model, losses = train_model(
            num_nodes=args.num_nodes,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            t_max=args.t_max,
        )
        save_model(model, model_path)
        plot_losses(losses, save_path=os.path.join(args.result_dir, "losses.png"))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"model file not found: {model_path}")
        model = PINN(num_nodes=args.num_nodes).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

    export_results(model, args)
    print("[done] results exported")


if __name__ == "__main__":
    main()
