import argparse
import os
import torch
from src import config, train, evaluate

def main():
    parser = argparse.ArgumentParser(description="PINNsformer Training and Evaluation")

    parser.add_argument('--mode', type=str, default='quick', choices=['train', 'quick'], help='Run mode: train or evaluate')
    parser.add_argument('--sampler', type=str, default=config.SAMPLER, choices=['grid', 'lhs'], help='Collocation point sampler: grid or lhs')
    parser.add_argument('--data_path', type=str, default=config.DATA_PATH, help='Path to the evaluation data')
    parser.add_argument('--model_save_path', type=str, default=config.MODEL_SAVE_PATH, help='Path to save/load the model')
    parser.add_argument('--animation_save_path', type=str, default=config.ANIMATION_SAVE_PATH, help='Path to save the result animation')
    parser.add_argument('--d_out', type=int, default=config.D_OUT)
    parser.add_argument('--d_model', type=int, default=config.D_MODEL)
    parser.add_argument('--d_hidden', type=int, default=config.D_HIDDEN)
    parser.add_argument('--n_layers', type=int, default=config.N_LAYERS)
    parser.add_argument('--n_heads', type=  int, default=config.N_HEADS)
    parser.add_argument('--device', type=str, default=config.DEVICE)
    parser.add_argument('--epochs', type=int, default=config.EPOCHS)
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE)
    parser.add_argument('--time_steps', type=int, default=config.TIME_STEPS)
    parser.add_argument('--step_size', type=float, default=config.STEP_SIZE)

    args = parser.parse_args()
    
    for key, value in config.__dict__.items():
        if key.isupper() and not hasattr(args, key.lower()):
            setattr(args, key.lower(), value)

    if isinstance(args.device, str) and args.device.startswith('cuda') and not torch.cuda.is_available():
        args.device = 'cpu'

    print("--- Running with the following configuration ---")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("------------------------------------------------")

    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.animation_save_path), exist_ok=True)

    if args.mode == 'train':
        train.run(args)
    elif args.mode == 'quick':
        evaluate.run(args)

if __name__ == '__main__':
    main()
