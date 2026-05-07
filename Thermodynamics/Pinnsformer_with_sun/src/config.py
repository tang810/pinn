# -- Path settings --
DATA_PATH = '/data/AI4PDE_CN/src/PINN4Science/Thermodynamics/Pinnsformer_with_sun/data/data_7246.txt'
MODEL_SAVE_PATH = '/data/AI4PDE_CN/src/PINN4Science/Thermodynamics/Pinnsformer_with_sun/model/pinnsformer_withsun.pt'
ANIMATION_SAVE_PATH = '/data/AI4PDE_CN/src/PINN4Science/Thermodynamics/Pinnsformer_with_sun/results/animation.gif'

# -- Device settings --
DEVICE = "cuda:0" 

# -- Model hyperparameters --
D_OUT = 1
D_MODEL = 64
D_HIDDEN = 512
N_LAYERS = 1
N_HEADS = 2

# -- Training settings --
SAMPLER = 'lhs'
LEARNING_RATE = 1
EPOCHS = 100
TIME_STEPS = 5
STEP_SIZE = 2e-2

# -- Regularization settings --
LAMBDA_PDE = 1
LAMBDA_BC = 1
LAMBDA_IC = 1
LAMBDA_REG = 0 # no reg loss

# -- Physical constants --
K = 1.0
RHO = 1.0
C_P = 1.0
EPS = 0.1
SIGMA = 5.67e-8
T_AMB = 3.0
T_0 = 273.0
Q_SUM = 400.0
