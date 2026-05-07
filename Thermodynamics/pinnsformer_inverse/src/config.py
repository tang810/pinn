import os

# =========================
# Path settings
# =========================
# Use project-root-relative paths so both local and server runs work without manual edits.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(PROJECT_ROOT, 'data', 'data_7246.txt')
model_save_path = os.path.join(PROJECT_ROOT, 'model', 'pinnsformer_inverse.pt')
animation_save_path = os.path.join(PROJECT_ROOT, 'results', 'animation.gif')

# =========================
# Device
# =========================
device = "cuda"

# =========================
# Model hyperparameters
# =========================
d_model = 128
d_hidden = 512
n_layers = 4
n_heads = 1

# =========================
# Training settings
# =========================
sampler = 'lhs'
learning_rate = 1e-4
epochs = 200
batch_size = 200
data_batch_size = 400
time_steps = 5
step_size = 2e-2

# =========================
# Resume training
# =========================
resume_epoch = 0

# =====================================================
# Loss switches (not directly used in code, kept for reference)
# =====================================================
use_pde = True
use_data = True
use_ic = True
use_bc = True
use_bc_disp = False

# =========================
# Thermal constants
# =========================
k = 1.0
rho = 1.0
c_p = 1.0
t_0 = 273.0
t_ref = t_0
q_source = 0.0
q_sum = q_source
q_bc_x1 = 400.0

# =========================
# Elastic constants
# =========================
e = 1.0
nu = 0.3
alpha_t = 0.01

# =========================
# Radiation / numerical constants
# =========================
eps = 0.1
sigma = 5.67e-8
t_amb = 3.0

# =========================
# Loss weights
# =========================
lambda_pde = 1.0
lambda_elastic = 0.1
lambda_ic = 1.0
lambda_ic_disp = 0.0
lambda_bc = 0.0
lambda_bc_disp = 0.0
lambda_data = 3.0
lambda_reg = 0.0
lambda_heatflux = 0.1
lambda_rad = 0.1

# =========================
# Inverse mode settings
# =========================
inverse_mode = True               # 开启反演
infer_params = ['k']              # 只反演导热系数 k
