import os
def define_arguments(parser):
    # General settings
    parser.add_argument('--mode', type=str, default='quick', choices=['train', 'quick', 'viz'],
                        help='Mode to run the script: train, quick, or viz')
    parser.add_argument('--data_path', type=str, default='data/data_7246.txt',
                        help='Path to the data file')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save outputs')

    # Model parameters
    parser.add_argument('--d_model', type=int, default=128,
                        help='Model dimension for Transformer')
    parser.add_argument('--d_hidden', type=int, default=128,
                        help='Hidden dimension in output head')
    parser.add_argument('--N', type=int, default=1,
                        help='Number of Transformer layers')
    parser.add_argument('--heads', type=int, default=2,
                        help='Number of attention heads')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1.0,
                        help='Learning rate (for LBFGS)')
    parser.add_argument('--res_points', type=int, default=20000,
                        help='Number of residual (interior) points')
    parser.add_argument('--bc_points', type=int, default=5000,
                        help='Number of boundary points')
    parser.add_argument('--ic_points', type=int, default=5000,
                        help='Number of initial condition points')
    parser.add_argument('--sampler', type=str, default='lhs', choices=['lhs', 'grid'],
                        help='Sampling method: lhs or grid')

    # Time sequence settings
    parser.add_argument('--time_steps', type=int, default=10,
                        help='Number of time steps for the sequence')
    parser.add_argument('--step_size', type=float, default=0.01,
                        help='Time step size for sequence generation')

    # Physics parameters (Thermal and Elastic)
    parser.add_argument('--rho', type=float, default=7800.0,
                        help='Density (kg/m^3)')
    parser.add_argument('--Cp', type=float, default=500.0,
                        help='Heat capacity (J/kg*K)')
    parser.add_argument('--k', type=float, default=45.0,
                        help='Thermal conductivity (W/m*K)')
    parser.add_argument('--E', type=float, default=210e9,
                        help="Young's modulus (Pa)")
    parser.add_argument('--nu', type=float, default=0.3,
                        help="Poisson's ratio")
    parser.add_argument('--alpha', type=float, default=1.2e-5,
                        help='Thermal expansion coefficient (1/K)')
    parser.add_argument('--T0', type=float, default=273.15,
                        help='Reference temperature (K)')

    # Radiative parameters (optional usage)
    parser.add_argument('--eps', type=float, default=0.9,
                        help='Emissivity')
    parser.add_argument('--sigma', type=float, default=5.670374419e-8,
                        help='Stefan-Boltzmann constant')
    parser.add_argument('--T_amb', type=float, default=300.0,
                        help='Ambient temperature (K)')
    parser.add_argument('--q_sum', type=float, default=0.0,
                        help='Heat source term (W/m^3)')

    # Loss weights
    parser.add_argument('--w_heat', type=float, default=1.0,
                        help='Weight for heat equation loss')
    parser.add_argument('--w_elastic', type=float, default=0.1,
                        help='Weight for elastic equilibrium loss')
    parser.add_argument('--w_ic', type=float, default=1.0,
                        help='Weight for initial condition loss')
    parser.add_argument('--w_reg', type=float, default=1e-4,
                        help='Weight for regularization')

    # Geometry settings
    parser.add_argument('--geometry', type=str, default='csg', choices=['cube', 'csg'],
                        help='Training geometry: cube or csg')
    parser.add_argument('--geometry_backend', type=str, default='sym', choices=['sdf', 'sym'],
                    help='Geometry backend: sdf (local) or sym (PhysicsNeMo)')

    # CSG sampling params (only used when --geometry csg)
    parser.add_argument('--csg_bound', type=float, default=1.05,
                        help='Sampling box bound, points sampled in [-bound, bound]^3')
    parser.add_argument('--csg_hole_ratio', type=float, default=0.60,
                        help='Ratio of hole-boundary points among all boundary points')
    parser.add_argument('--csg_chunk', type=int, default=300000,
                        help='Chunk size for rejection-style sampling')
    parser.add_argument('--csg_max_rounds', type=int, default=50,
                        help='Max sampling rounds')
    parser.add_argument('--sdf_eps', type=float, default=1e-3,
                        help='Finite difference epsilon for SDF gradients')
    parser.add_argument('--proj_iters', type=int, default=2,
                        help='Number of SDF projection iterations for boundary points')
    parser.add_argument('--interior_margin', type=float, default=0.0,
                        help='Filter interior points by s_geom < -interior_margin (optional)')

    # Boundary condition loss weight + BC definitions
    parser.add_argument('--w_bc', type=float, default=1.0,
                        help='Weight for boundary condition loss')

    parser.add_argument('--bc_outer_type', type=str, default='dirichlet', choices=['dirichlet', 'neumann'],
                        help='Boundary type on outer surface')
    parser.add_argument('--bc_outer_value', type=float, default=273.15,
                        help='Dirichlet value (K) or Neumann value (dT/dn) on outer surface')

    parser.add_argument('--bc_hole_type', type=str, default='neumann', choices=['dirichlet', 'neumann'],
                        help='Boundary type on hole surfaces')
    parser.add_argument('--bc_hole_value', type=float, default=0.0,
                        help='Dirichlet value (K) or Neumann value (dT/dn) on hole surfaces')
    
    parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:0')

    # Viz settings
    parser.add_argument("--data_path_gt", type=str, default="data/data_7246.txt", help="data file with GT: x y z t T")
    parser.add_argument("--out_dir", type=str, default="results/viz_sym_gt", help="output directory")
    parser.add_argument("--plot_on", type=str, default="bc", choices=["bc", "res"], help="plot bc boundary or res interior")
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--cleanup_frames", action="store_true", help="delete png frames after gif")

    parser.add_argument("--ckpt", type=str, default="model/pinnsformer_withsun.pt", help="model checkpoint for prediction")
    parser.add_argument("--make_stress", action="store_true", help="also output stress gif based on prediction")
    parser.add_argument("--point_size", type=float, default=18, help="scatter marker size (smaller = finer)")
    parser.add_argument("--point_alpha", type=float, default=0.28, help="scatter alpha (smaller = less blocky)")
    parser.add_argument("--dpi", type=int, default=160, help="figure dpi")

    parser.add_argument("--pred_denorm", action="store_true", help="inverse-normalize T_pred into Kelvin using GT Tmin/Tmax: T=pred*(Tmax-Tmin)+Tmin")
    parser.add_argument("--cbar_mode", type=str, default="shared", choices=["shared", "separate"], help="shared: GT & Pred use same colorbar range (GT Tmin/Tmax). separate: Pred uses its own min/max")
    parser.add_argument("--t_tol", type=float, default=1e-8, help="t matching tolerance for np.isclose (float time)")
    parser.add_argument("--input_norm", type=str, default="none", choices=["none", "box01", "01tobox"], help="normalize xyz before feeding model: none or box01 (map [box_p1,box_p2] -> [0,1])")
    parser.add_argument("--t_norm", type=str, default="none", choices=["none", "minmax01"], help="normalize time tk before feeding model: none or minmax01 (map [tmin,tmax] -> [0,1])")
    parser.add_argument("--tmin", type=float, default=None, help="optional manual tmin for t_norm")
    parser.add_argument("--tmax", type=float, default=None, help="optional manual tmax for t_norm")
    parser.add_argument("--pred_calib", type=str, default="none", choices=["none", "global", "per_frame"], help="Calibrate T_pred to Kelvin by fitting T_cal=a*T_pred+b using GT.")
    parser.add_argument("--calib_points", type=int, default=20000, help="Number of points used to fit calibration (subsample for speed).")

# Sym geometry box settings
    parser.add_argument('--sym_shape', type=str, default='csg', choices=['csg', 'box'],
                        help='Sym geometry shape: csg (demo) or box (cuboid)')
    parser.add_argument('--box_p1', type=float, nargs=3, default=[-1.0, -1.0, -1.0],
                        help='Box lower corner (x y z) for sym_shape=box')
    parser.add_argument('--box_p2', type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        help='Box upper corner (x y z) for sym_shape=box')


def define_default_values(args):
    # Expand data path if relative
    args.data_path = os.path.expanduser(args.data_path)
    args.output_dir = os.path.expanduser(args.output_dir)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
