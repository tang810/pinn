import os
import sys
sys.path.append(os.path.dirname(__file__))

import yaml
import argparse
import numpy as np
from pathlib import Path
from dit4science.model.vae import VQVAE
from vae_experiment import VAEXperiment  # noqa
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dit4science.data.vae_dataset import VAEDataset
# from dataset import VAEDataset
# from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.strategies import DDPStrategy



config = {
    "model_params": {
        "name": 'VQVAE',
        "in_channels": 3,
        "embedding_dim": 64,
        "num_embeddings": 512,
        "img_size": 64,
        "beta": 0.25,
    },
    "data_params": {
        "data_path": "/home/chunyang/projects/ai4science/DDPM4Science/data/test/",
        "train_batch_size": 64,
        "val_batch_size":  64,
        "patch_size": 64,
        "num_workers": 4,
    },
    "exp_params":{
        "LR": 0.005,
        "weight_decay": 0.0,
        "scheduler_gamma": 0.0,
        "kld_weight": 0.00025,
        "manual_seed": 1265,
    },
    "trainer_params":{
        "gpus": [3],
        "max_epochs": 100,
    },
    "logging_params":{
        "save_dir": "logs/",
        "name": "VQVAE",
    }
}

tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                               name=config['model_params']['name'],)

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)

model = VQVAE(**config['model_params'])
experiment = VAEXperiment(model,
                          config['exp_params'])

data = VAEDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)

data.setup()
# runner = Trainer(logger=tb_logger,
#                  callbacks=[
#                      LearningRateMonitor(),
#                      ModelCheckpoint(save_top_k=2, 
#                                      dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
#                                      monitor= "val_loss",
#                                      save_last= True),
#                  ],
#                  strategy=DDPStrategy(find_unused_parameters=False),
#                  devices=config["trainer_params"]["gpus"], accelerator="gpu",
#                  max_epochs=config["trainer_params"]["max_epochs"],
#             )


runner = Trainer(logger=tb_logger,
                 callbacks=[
                     LearningRateMonitor(),
                     ModelCheckpoint(save_top_k=2, 
                                     dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
                                     monitor= "val_loss",
                                     save_last= True),
                 ],
                 strategy=DDPStrategy(find_unused_parameters=False),
                 **config['trainer_params'])

Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, datamodule=data)