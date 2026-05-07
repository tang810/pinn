# Physics-Informed Neural Network (PINN) for KdV Equation

This repository contains a Python implementation of a Physics-Informed Neural Network (PINN) to solve the Korteweg-de Vries (KdV) equation. The code is split into four main files for better organization and modularity.

## Files

1. **data.py**: Handles data generation and loading.
2. **net.py**: Defines the neural network architecture.
3. **pinn_solver.py**: Implements the PDE residual and boundary conditions.
4. **train.py**: Trains the model and saves the results.

## Requirements

- Python 3.x
- deepxde
- numpy
- torch

## Usage

1. **Generate Data**: Run `data.py` to generate the exact solution data.
2. **Train the Model**: Run `train.py` to train the PINN model.
3. **View Results**: The training results and plots will be saved automatically.

## Parameters

- `a`, `g`, `h`: Physical constants.
- `b`, `rho1`, `rho2`: Variables to be optimized.
- `c_t`, `lamda_t`: Constants for the exact solution.

## Training

The model is trained using the Adam optimizer with a learning rate of 1e-3. The training process includes a callback to monitor the values of the variables `b`, `rho1`, and `rho2`.

## Results

After training, the loss history and training state are saved and plotted. The variable values are saved in `test2.txt`.

## References

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686-707.
