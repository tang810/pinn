import argparse
import torch
from src import train
from src import evaluate
from src import config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        choices=['train', 'eval', 'quick'],
        help='train, eval, or quick (alias of eval)'
    )
    parser.add_argument('--model_save_path', type=str, default=config.model_save_path)
    parser.add_argument('--data_path', type=str, default=config.data_path)
    parser.add_argument('--animation_save_path', type=str, default=config.animation_save_path)
    parser.add_argument('--batch_size', type=int, default=config.batch_size)
    parser.add_argument('--data_batch_size', type=int, default=config.data_batch_size)
    parser.add_argument('--epochs', type=int, default=config.epochs)
    parser.add_argument('--learning_rate', type=float, default=config.learning_rate)
    parser.add_argument('--d_model', type=int, default=config.d_model)
    parser.add_argument('--d_hidden', type=int, default=config.d_hidden)
    parser.add_argument('--n_layers', type=int, default=config.n_layers)
    parser.add_argument('--n_heads', type=int, default=config.n_heads)
    parser.add_argument('--device', type=str, default=config.device)
    parser.add_argument('--k', type=float, default=config.k)
    parser.add_argument('--rho', type=float, default=config.rho)
    parser.add_argument('--c_p', type=float, default=config.c_p)
    parser.add_argument('--t_0', type=float, default=config.t_0)
    parser.add_argument('--t_ref', type=float, default=config.t_ref)
    parser.add_argument('--q_sum', type=float, default=config.q_sum)
    parser.add_argument('--e', type=float, default=config.e)
    parser.add_argument('--nu', type=float, default=config.nu)
    parser.add_argument('--alpha_t', type=float, default=config.alpha_t)

    # 边界热流参数（x=1面）
    parser.add_argument('--q_bc_x1', type=float, default=config.q_bc_x1, help='Heat flux on x=1 boundary (W/m²)')

    # 辐射常数
    parser.add_argument('--eps', type=float, default=config.eps)
    parser.add_argument('--sigma', type=float, default=config.sigma)
    parser.add_argument('--t_amb', type=float, default=config.t_amb)

    # 损失权重
    parser.add_argument('--lambda_pde', type=float, default=config.lambda_pde)
    parser.add_argument('--lambda_ic', type=float, default=config.lambda_ic)
    parser.add_argument('--lambda_ic_disp', type=float, default=config.lambda_ic_disp)
    parser.add_argument('--lambda_bc', type=float, default=config.lambda_bc)
    parser.add_argument('--lambda_bc_disp', type=float, default=config.lambda_bc_disp)
    parser.add_argument('--lambda_data', type=float, default=config.lambda_data)
    parser.add_argument('--lambda_elastic', type=float, default=config.lambda_elastic)
    parser.add_argument('--lambda_heatflux', type=float, default=config.lambda_heatflux)
    parser.add_argument('--lambda_rad', type=float, default=config.lambda_rad)

    # 恢复训练
    parser.add_argument('--resume_epoch', type=int, default=config.resume_epoch, help='Resume from checkpoint epoch (0 means no resume)')

    # ========== 新增反演相关参数 ==========
    parser.add_argument('--inverse_mode', action='store_true', default=config.inverse_mode, help='Enable inverse mode')
    parser.add_argument('--infer_params', type=str, nargs='+', default=config.infer_params, help='List of parameters to infer, e.g., k alpha q_bc_x1')

    # ========== 新增预训练模型路径参数 ==========
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='Path to pretrained model for inverse fine-tuning')

    args = parser.parse_args()
    # Normalize device early so downstream loaders/modules can safely use args.device.
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        args.device = 'cpu'

    if args.mode == 'train':
        train.run(args)
    elif args.mode in ('eval', 'quick'):
        evaluate.run(args)
    else:
        print("Invalid mode. Choose one of: 'train', 'eval', 'quick'.")

if __name__ == "__main__":
    main()
