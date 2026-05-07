
import os
import argparse
import torch
from src.model import build_model
from src.data_loader import generate_data_simul, load_data_from_file
from src.train import train_model
from src.visualize import plot_result, make_gif
from src.utils import load_model, save_model

def main():
    parser = argparse.ArgumentParser(description="Beijing Snowfall PINN Simulation")
    
    # Mode Settings
    parser.add_argument('--mode', type=str, choices=['quick', 'train'], default='quick',
                        help="Choose mode: 'quick' for fast visualization, 'train' for training the model")
    
    # Data Settings
    parser.add_argument('--data', type=str, choices=['simul', 'load'], default='simul',
                        help="Data source: 'simul' for simulation data, 'file' for pre-loaded data")
    parser.add_argument('--data_path', type=str, default=None, help="Path to data file if --load is used")
    
    args = parser.parse_args()
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Paths
    BASE_DIR = "/data/AI4PDE_CN/src/PINN4Science/Geology/Beijing_Snowfall_PINN"
    MODEL_PATH = os.path.join(BASE_DIR, "model", "model_snow_pinn_v2.pt")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    GEOJSON_PATH = os.path.join(BASE_DIR, "data", "北京.json")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "model"), exist_ok=True)

    # 1. Model
    model = build_model(DEVICE)
    model, loaded = load_model(model, MODEL_PATH, DEVICE)
    
    # 2. Modes
    if args.mode == 'quick' :
        print(">>> Quick Mode: Generating Visualization...")
        if not loaded:
            print("Warning: No pre-trained model found for quick mode. Using random weights.")
        plot_result(model, t_val=0.8, device=DEVICE, geojson_path=GEOJSON_PATH, 
                   save_path=os.path.join(RESULTS_DIR, "snow_quick_viz.png"))
        return

    if args.mode == 'train' :
        iterations = 5000
        if args.load and args.data_path:
            print("File-based training not implemented, using simulation.")
            data_gen = lambda: generate_data_simul(device=DEVICE)
        else:
            print("Using Simulation Data Generation...")
            data_gen = lambda: generate_data_simul(device=DEVICE)

        print(f">>> Training Mode: Start training for {iterations} iterations...")
        model = train_model(model, data_gen, iterations=iterations, device=DEVICE)
        save_model(model, MODEL_PATH)
        
        print("Generating post-training results...")
        plot_result(model, t_val=0.8, device=DEVICE, geojson_path=GEOJSON_PATH,
                   save_path=os.path.join(RESULTS_DIR, "snow_sim_beijing_t80.png"))
        make_gif(model, DEVICE, GEOJSON_PATH, save_path=os.path.join(RESULTS_DIR, "snowfall_pinn_beijing.gif"))
        return
    
    if not args.quick and not args.train:
        print("No simulation mode selected. Defaulting to quick check.")
        plot_result(model, t_val=0.5, device=DEVICE, geojson_path=GEOJSON_PATH,
                   save_path=os.path.join(RESULTS_DIR, "snow_check.png"))

if __name__ == "__main__":
    main()
