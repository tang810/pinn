# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# import torch.distributed as dist
from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import SequentialSampler  # noqa
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import gc
import sys

sys.path.append(".")
from dit4science.model.dit_models import DiT_models  # noqa
from dit4science.model.dit_diffusion import create_diffusion  # noqa
from dit4science.data.dataset_dit import TurbDataset  # noqa
from dit4science.model.UNet import UNet, weights_init  # noqa

# Autoencoder related import
from dit4science.model.autoencoder import AutoencoderKL  # noqa

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
#################################################################################  # noqa
#                             Training Helper Functions                         #  # noqa
#################################################################################  # noqa


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed  # noqa
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{logging_dir}/log.txt")
            ]
    )
    logger = logging.getLogger(__name__)
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(
        arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################  # noqa
#                                  Training Loop                                #  # noqa
#################################################################################  # noqa

def main(args):
    """
    Trains a new DiT model.
    """
    # assert torch.cuda.is_available(), "Training currently requires at least one GPU."   # noqa
    # Setup DDP:
    # dist.init_process_group("nccl")
    # assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."   # noqa
    # rank = dist.get_rank()
    # device = rank % torch.cuda.device_count()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # seed = args.global_seed * dist.get_world_size() + rank
    seed = 42
    torch.manual_seed(seed)
    # torch.cuda.set_device(device)
    # print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")   # noqa

    # Setup an experiment folder:
    # if rank == 0:
    os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)   # noqa
    experiment_index = len(glob(f"{args.results_dir}/*"))
    model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)   # noqa
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-channel_idx{args.predchannel}"  # Create an experiment folder   # noqa
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints   # noqa
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")
    # else:
    #     logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."   # noqa
    model = UNet()
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training  # noqa
    requires_grad(ema, False)
    model = model.to(device)
    model.apply(weights_init)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule   # noqa
    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")  # noqa

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):   # noqa
    opt = torch.optim.Adam(model.parameters(), betas=(0.5, 0.999), lr=0.0006, weight_decay=0)  # noqa

    # dataset = ImageFolder(args.data_path, transform=transform)
    prop = None
    print("Prediction on channel: ", args.predchannel)
    dataset = TurbDataset(
        dataProp=prop, dataDir=args.data_path, gaussian_norm=False,
        channel_idx=args.predchannel)

    sampler = SequentialSampler(
        dataset,
        # num_replicas=dist.get_world_size(),
        # rank=rank,
        # shuffle=True,
        # seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    # update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights   # noqa
    model.train()  # important! This enables embedding dropout for classifier-free guidance  # noqa
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    # ########### VAE init ###################
    # Autoencoder config
    config = {
        "monitor": "val/rec_loss",
        "embed_dim": 4,
        "lossconfig": {
            "target": "dit4science.loss.LPIPSWithDiscriminator",
            "params": {
                "disc_start": 50001,
                "kl_weight": 0.000001,
                "disc_weight": 0.5
            }
        },
        "ddconfig": {
            "double_z": True,
            "z_channels": 4,
            "resolution": 256,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 128,
            "ch_mult": [1, 2, 4, 4],  # num_down = len(ch_mult) - 1
            "num_res_blocks": 2,
            "attn_resolutions": [],
            "dropout": 0.0
        }
    }

    vae_ckpt_cond = args.vae_ckpt_cond
    vae_ckpt_res = args.vae_ckpt_res

    vae_cond = AutoencoderKL(**config).to(device)
    vae_cond.init_from_ckpt(vae_ckpt_cond)

    vae_res = AutoencoderKL(**config).to(device)
    vae_res.init_from_ckpt(vae_ckpt_res)
    loss_fn = torch.nn.L1Loss()
    #########################################

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        # torch.cuda.empty_cache()
        gc.collect()
        # sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        # x is condition (initial state of the field).
        # y is the desired physical field.
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            # x = upsample(x).to(device)
            # y = upsample(y).to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = x
                y = vae_res.encode(y).sample()
                x = vae_cond.encode(x).sample()
                # print("x shape and y shape")
                # print(x.shape, y.shape)
            # t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)   # noqa
            # model_kwargs = dict(y=x)
            out = model(x)
            loss = loss_fn(out, y)
            model.zero_grad()
            loss.backward()
            opt.step()
            # update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                # torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)   # noqa
                # dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")   # noqa
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args
                }
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")


if __name__ == "__main__":
    # Specify pretrained VAE path here:
    vae_ckpt_cond = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'weights', "cond_all_epoch=152-step=167840.ckpt")   # noqa
    vae_ckpt_res = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'weights', "res_all_epoch=109-step=120669.ckpt")  # noqa

    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).   # noqa
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--predchannel", type=int, default=-1)  # the channel to predict: 0-pressure, 1-Ux, 2-Uy, -1-all channel   # noqa
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="UNet")  # noqa
    parser.add_argument("--image-size", type=int, choices=[128, 256, 512], default=128)  # noqa
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=4200)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training   # noqa
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--ckpt-every", type=int, default=100)
    parser.add_argument("--vae-ckpt-cond", type=str, default=vae_ckpt_cond)
    parser.add_argument("--vae-ckpt-res", type=str, default=vae_ckpt_res)
    args = parser.parse_args()
    main(args)

# Train DiT
# python scripts/train_latent_unet.py --data-path=./data/train/  # noqa
