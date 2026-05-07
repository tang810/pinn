# config.py

import torch

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Material properties
E  = 1.0    # Young's modulus
mu = 0.25   # Poisson's ratio

# File paths
load_data_path            = "./data/Coord.mat"
test_data_path            = "./data/FEA.mat"
model_path                = "./model/pinn_elasticity_model.pt"
loss_curve_path           = "./viz/loss_curve.png"
plot_displacement_path    = "./viz/predict_displacement.png"
plot_stress_path          = "./viz/predict_stress.png"

# Training hyperparameters
n_iter = 10000
lr     = 1e-2

#batch size
batch_size = 9261