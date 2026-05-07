
import os
import argparse
import torch
from src.model import build_model
from src.data_loader import generate_data_simul, load_data_from_file, Config
from src.train import train_model
from src.visualize import plot_result, make_gif
from src.utils import load_model, save_model

def main():
    parser = argparse.ArgumentParser(description="Orographic Rainfall PINN Simulation")
    
    # Mode Settings
    parser.add_argument('--mode', type=str, choices=['quick', 'train'], default='quick',
                        help="Choose mode: 'quick' for fast visualization, 'train' for training the model")
    
    # Data Settings
    parser.add_argument('--data', type=str, choices=['simul', 'load'], default='simul',
                        help="Data source: 'simul' for simulation data, 'file' for pre-loaded data")
    parser.add_argument('--data_path', type=str, default=None, help="Path to data file if --load is used")
    
    args = parser.parse_args()
    
    # Device Configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Paths
    BASE_DIR = "/data/AI4PDE_CN/src/PINN4Science/Geology/Orographic_Rainfall_PINN"
    MODEL_PATH = os.path.join(BASE_DIR, "model", "model_terrain_pinn.pt")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    GEOJSON_PATH = os.path.join(BASE_DIR, "data", "北京.json")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "model"), exist_ok=True)

    # 1. Model Construction
    model = build_model(DEVICE)
    
    # 2. Model Loading
    model, loaded = load_model(model, MODEL_PATH, DEVICE)
    
    if args.mode == 'quick' :
        # Quick visualization mode
        print(">>> Quick Mode: Generating Visualization...")
        if not loaded:
            print("Warning: No pre-trained model found for quick mode. Using random weights.")
        
        plot_result(model, t_query=0.7, device=DEVICE, geojson_path=GEOJSON_PATH, 
                   save_path=os.path.join(RESULTS_DIR, "rain_terrain_quick.png"))
        return

    if args.mode == 'train' :
        # Training Mode
        iterations = 3000
        
        # Data Loading Strategy
        if args.load and args.data_path:
            print(f"Loading data from {args.data_path}...")
            # Note: Current train_model expects a generator function. 
            # Adapting this would require changing train_loop to accept fixed tensors or a generator wrapper.
            # For now, we assume simulation based training as primary PINN method.
            print("Warning: File-based training not fully implemented for PINN residual. Switching to simulation.")
            data_gen = lambda: generate_data_simul(device=DEVICE)
        else:
            print("Using Simulation Data Generation...")
            data_gen = lambda: generate_data_simul(device=DEVICE)

        print(f">>> Training Mode: Start training for {iterations} iterations...")
        model = train_model(model, data_gen, iterations=iterations, device=DEVICE)
        save_model(model, MODEL_PATH)
        
        # Generate Results after training
        print("Generating post-training results...")
        plot_result(model, t_query=0.7, device=DEVICE, geojson_path=GEOJSON_PATH,
                   save_path=os.path.join(RESULTS_DIR, "rain_terrain_final.png"))
        make_gif(model, DEVICE, GEOJSON_PATH, save_path=os.path.join(RESULTS_DIR, "rainfall_highres.gif"))
        return

    # Default behavior if no args provided (Run both or print help)
    if not args.quick and not args.train:
        print("No mode selected. Suggest running with --help")
        print("Defaulting to quick visualization check...")
        plot_result(model, t_query=0.5, device=DEVICE, geojson_path=GEOJSON_PATH,
                   save_path=os.path.join(RESULTS_DIR, "rain_terrain_check.png"))

if __name__ == "__main__":
    main()