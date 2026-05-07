import argparse
import yaml
from easydict import EasyDict as edict

class LoadFromFile(argparse.Action):
    # parser.add_argument('--file', type=open, action=LoadFromFile)
    def __call__(self, parser, namespace, values, option_string=None):
        if values.name.endswith("yaml") or values.name.endswith("yml"):
            with values as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            for key in config.keys():
                if key not in namespace:
                    raise ValueError(f"Unknown argument in config file: {key}")
            namespace.__dict__.update(config)
        else:
            raise ValueError("Configuration file must end with yaml or yml")

def get_args():
    
    # General arguments
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--conf', '-c', type=open, action=LoadFromFile, help='Configuration yaml file')
    parser.add_argument('--config', type=str, help='Configuration yaml file')

    parser.add_argument('--data_name', type=str, default='PDBBind', help='name of dataset')
    parser.add_argument('--data_dir', type=str, default='data_files/processed/PDBBind_processed/', help='directory for storing dataset')
    parser.add_argument('--data_path', type=str, default='data_files/inference/input_protein_ligand.csv', help='path of dataset')
    parser.add_argument('--data_split_dir', type=str, default='data_files/processed/splits/timesplit_no_lig_overlap_train', 
                        help='directory storing index for spliting dataset')
    parser.add_argument('--data_split_ratio', type=float, default=0.8, help='ratio for spliting dataset')
    parser.add_argument('--data_cache_dir', type=str, default='data_files/processed/cache', help='directory for storing cached dataset')
    parser.add_argument('--data_output_dir', type=str, default='data_files/output/diffdock', help='directory for storing output')
    parser.add_argument('--data_esm_embd_dir', type=str, default='data_files/processed/esm2_output', 
                        help='directory for storing LM embeddings for the receptor features')
    parser.add_argument('--data_esm_embd_path', type=str, default='data_files/processed/esm2_3billion_embeddings.pt', 
                        help='path for storing LM embeddings for the receptor features')
    
    parser.add_argument('--data_node_num_max', default=125, type=int, help='max number of node in each graph')
    parser.add_argument('--data_centralization', default=True, type=bool, help='whether to rescale data to [-1,1]')
    parser.add_argument('--data_channelization', default=True, type=bool, help='whether to add channel')
    parser.add_argument('--data_remove_diagonal', default=True, type=bool, help='whether to remove diagonal in adjacency matrix')
    
    parser.add_argument('--data_pin_memory', action='store_true', default=False, help='where or not use pin memory')
    parser.add_argument('--data_complex_limit', type=int, default=0, help='If positive, the number of training and validation complexes is capped')
    parser.add_argument('--data_worker_num', type=int, default=1, help='Number of workers for preprocessing')
    parser.add_argument('--data_all_atom', action='store_true', default=False, help='whether to use the all atoms model')
    parser.add_argument('--data_rec_radius', type=float, default=15, help='cutoff on distances for receptor edges')
    parser.add_argument('--data_ca_nb_num_max', type=int, default=24, help='maximum number of neighbors for each residue')
    parser.add_argument('--data_atom_radius', type=float, default=5, help='cutoff on distances for atom connections')
    parser.add_argument('--data_atom_nb_num_max', type=int, default=8, help='maximum number of atom neighbours for receptor')
    parser.add_argument('--data_matching_popsize', type=int, default=20, help='differential evolution popsize parameter in matching')
    parser.add_argument('--data_matching_maxiter', type=int, default=20, help='differential evolution maxiter parameter in matching')
    parser.add_argument('--data_lig_size_max', type=int, default=None, help='maximum number of heavy atoms in ligand')
    parser.add_argument('--data_remove_hs', action='store_true', default=True, help='whether to remove Hs')
    parser.add_argument('--data_conformer_num', type=int, default=1, help='number of conformers to match to each ligand')
    parser.add_argument('--data_keep_local_structure', action='store_true', default=False,
                        help='keeps the local structure when specifying an input with 3D coordinates instead of generating them with RDKit')
    parser.add_argument('--data_sample_per_complex', type=int, default=5, help='Number of samples to generate')
    
    parser.add_argument('--model_features_num', default=256, type=int, help='number of features')
    parser.add_argument('--model_graph_layer_type', default='PosTransLayer', type=str, help='type of graph layer')
    parser.add_argument('--model_graph_layers_num', default=4, type=int, help='number of graph layer')
    parser.add_argument('--model_channels_num', default=1, type=int, help='number of channels')
    parser.add_argument('--model_actv_type', default='silu', type=str, help='type of activatioin function')
    
    parser.add_argument('--model_conv_layer_num', type=int, default=6, help='Number of interaction layers') # 1
    parser.add_argument('--model_max_radius', type=float, default=5.0, help='Radius cutoff for geometric graph')
    parser.add_argument('--model_scale_by_sigma', action='store_true', default=True, help='Whether to normalise the score')
    parser.add_argument('--model_ns', type=int, default=48, help='Number of hidden features per node of order 0')
    parser.add_argument('--model_nv', type=int, default=10, help='Number of hidden features per node of order >0')
    parser.add_argument('--model_dist_embd_dim', type=int, default=64, help='Embedding size for the distance')
    parser.add_argument('--model_cro_dist_embd_dim', type=int, default=64, help='Embeddings size for the cross distance')
    parser.add_argument('--model_no_batch_norm', action='store_true', default=False, help='If set, it removes the batch norm')
    parser.add_argument('--model_use_second_order_repr', action='store_true', default=False, help='Whether to use only up to first order representations or also second')
    parser.add_argument('--model_cro_dist_max', type=float, default=80, help='Maximum cross distance in case not dynamic')
    parser.add_argument('--model_dynamic_cro_max', action='store_true', default=True, help='Whether to use the dynamic distance cutoff')
    parser.add_argument('--model_dropout_rate', type=float, default=0.1, help='MLP dropout rate') # 
    parser.add_argument('--model_embd_type', type=str, default="sinusoidal", help='Type of diffusion time embedding')
    parser.add_argument('--model_lm_embd_type', type=str, default='esm', help='Type of language model embedding')
    parser.add_argument('--model_sigma_embd_dim', type=int, default=64, help='Size of the embedding of the diffusion time')
    parser.add_argument('--model_embd_scale', type=int, default=10000, help='Parameter of the diffusion time embedding')

    parser.add_argument('--model_score_dir', type=str, default='checkpoints/diffdock/score', help='Path to folder with trained score model and hyperparameters')
    parser.add_argument('--model_score_ckpt', type=str, default='best_ema_inference_epoch_model.pt', help='Checkpoint to use for the score model')
    parser.add_argument('--model_confi_dir', type=str, default='checkpoints/diffdock/confidence', help='Path to folder with trained confidence model and hyperparameters')
    parser.add_argument('--model_confi_ckpt', type=str, default='best_model_epoch75.pt', help='Checkpoint to use for the confidence model')

    parser.add_argument('--sde_beta_min', default=0.1, type=float, help='minimum beta')
    parser.add_argument('--sde_beta_max', default=1.0, type=float, help='maximum beta')
    parser.add_argument('--diffusion_time_steps_num', default=1000, type=int, help='number of time steps in diffusion')
    
    parser.add_argument('--diffusion_tr_weight', type=float, default=0.33, help='Weight of translation loss')
    parser.add_argument('--diffusion_rot_weight', type=float, default=0.33, help='Weight of rotation loss')
    parser.add_argument('--diffusion_tor_weight', type=float, default=0.33, help='Weight of torsional loss')
    parser.add_argument('--diffusion_rot_sigma_min', type=float, default=0.03, help='Minimum sigma for rotational component')
    parser.add_argument('--diffusion_rot_sigma_max', type=float, default=1.55, help='Maximum sigma for rotational component')
    parser.add_argument('--diffusion_tr_sigma_min', type=float, default=0.1, help='Minimum sigma for translational component')
    parser.add_argument('--diffusion_tr_sigma_max', type=float, default=19.0, help='Maximum sigma for translational component')
    parser.add_argument('--diffusion_tor_sigma_min', type=float, default=0.0314, help='Minimum sigma for torsional component')
    parser.add_argument('--diffusion_tor_sigma_max', type=float, default=3.14, help='Maximum sigma for torsional component')
    parser.add_argument('--diffusion_no_torsion', action='store_true', default=False, help='If set only rigid matching')
    parser.add_argument('--diffusion_no_random', action='store_true', default=False, help='Use no randomness in reverse diffusion')
    parser.add_argument('--diffusion_no_final_step_noise', action='store_true', default=False, help='Use no noise in the final step of the reverse diffusion')
    parser.add_argument('--diffusion_use_ode', action='store_true', default=False, help='Use ODE formulation for inference')

    parser.add_argument('--train_batch_size', default=32, type=int, help='batch size in training')
    parser.add_argument('--train_lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--train_lr_decay', default=0.999, type=float, help='decay rate')
    parser.add_argument('--train_epoch_num', default=100, type=int, help='number of training epoch')
    parser.add_argument('--train_warmup_step', default=0, type=float, help='number of warmup step')
    parser.add_argument('--train_grad_clip', default=0., type=int, help='upper bound of norm for gradient clipping')
    parser.add_argument('--train_use_ema', default=False, type=bool, help='whether use exponential moving average')
    
    parser.add_argument('--sample_signal_noise_ratio', default=0.16, type=float, help='signal to noise ratio')
    parser.add_argument('--sample_time_steps_num_each', default=1, type=int, help='number of time steps in each stop of correction')
    parser.add_argument('--sample_scale_eps', default='0.7', type=float, help='root directory')
    parser.add_argument('--sample_noise_removal', default=True, type=bool, help='root directory')
    parser.add_argument('--sample_probability_flow', default=False, type=bool, help='root directory')
    parser.add_argument('--sample_eps', default=1.0e-4, type=float, help='root directory')

    parser.add_argument('--eval_batch_size', default=16, type=int, help='batch size in evaluation')
    
    parser.add_argument('--inference_sample_size', type=int, default=32, help='')
    parser.add_argument('--inference_step_num', type=int, default=10, help='Number of denoising steps') # 20

    parser.add_argument('--device', default='cpu', type=str, help='device')
    parser.add_argument('--no_parallel', default=True, type=bool, help='Whether or not use parallel training')
    
    parser.add_argument('--ckpt', default='gdss_qm9', type=str, help='root directory')

    parser.add_argument('--cudnn_benchmark', action='store_true', default=True, help='CUDA optimization parameter for faster training')

    args = parser.parse_args()
    return args

def load_yaml(dir, name):
    with open(f'{dir}/{name}.yaml') as f:
        args = argparse.Namespace(**yaml.full_load(f))
    return args

def load_config(path):
    with open(path, 'r') as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))
    return config
