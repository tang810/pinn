from dataclasses import dataclass


@dataclass
class Settings:
    data_path: str = "data/data_7246.txt"

    true_alpha_x: float = 1.0
    true_alpha_perp: float = 1.0

    n_f: int = 20000
    n_bc_per_face: int = 2500
    n_ic: int = 4000
    adam_iters: int = 5000
    noise_level: float = 0.01
    layers: tuple = (4, 20, 20, 20, 20, 20, 20, 20, 20, 1)

    lambda_data: float = 1.0
    lambda_pde: float = 1.0
    lambda_bc: float = 1.0
    lambda_ic: float = 1.0
    lambda_reg: float = 0.01

    k: float = 1.0
    rho: float = 1.0
    c_p: float = 1.0
    eps: float = 0.1
    sigma: float = 5.67e-8
    t_amb: float = 3.0
    t_0: float = 273.0
    q_sum: float = 400.0
    bc_res_scale: float = 100.0

    surface_tol: float = 1e-6

    run_full_matrix: bool = True
    default_n_sensors: int = 5
    default_n_time_steps: int = 80
    sensor_counts: tuple = (3, 5)
    time_step_counts: tuple = (80, 100)

    random_seed: int = 1234
    use_lbfgs: bool = True


@dataclass
class Paths:
    model_dir: str = "model"
    result_dir: str = "result"


def build_train_settings() -> Settings:
    return Settings()


def build_quick_settings() -> Settings:
    cfg = Settings()
    cfg.n_f = 3000
    cfg.n_bc_per_face = 600
    cfg.n_ic = 1000
    cfg.adam_iters = 300
    cfg.run_full_matrix = False
    cfg.use_lbfgs = False
    return cfg
