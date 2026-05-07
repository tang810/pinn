import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler

from einops import rearrange
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

import os
from typing import List,Tuple,Dict

import sys

sys.path.append(os.path.abspath('./ai4flow/acdm'))
sys.path.append(os.path.abspath('./ai4flow'))

from turbpred.data_transformations import Transforms
from turbpred.model_diffusion import DiffusionModel
from turbpred.params import DataParams
from turbpred.model import PredictionModel
from turbpred.logger import Logger
from turbpred.params import DataParams, TrainingParams, LossParams, ModelParamsEncoder, ModelParamsDecoder, ModelParamsLatent
from turbpred.loss import PredictionLoss
from turbpred.loss_history import LossHistory
from turbpred.trainer_diffusion import TrainerDiffusion, TesterDiffusion
from dataset.airfoil_acdm import airfoilDataset
from dataset.airfoil_acdm import Transforms


useGPU = True
self_cond = True
gpuID = "1"
startFromCheckpoint = False
os.environ["CUDA_VISIBLE_DEVICES"] = gpuID

trainSet = airfoilDataset(name="V2D dataset",
                     dataDir="/home/chunyang/projects/DDPM4Science/data/airfoil_stable/train_6k",
                     sequenceLength=[3, 2], rey=1, self_cond = True)


p_d = DataParams(batch=32, augmentations=["normalize"], sequenceLength=[3,2], randSeqOffset=True,
            dataSize=[128,64], dimension=2, simFields=["pres"], simParams=["rey", "u_x", "u_y"], normalizeMode="incMixed")
transTrain = Transforms()
trainSet.transform = transTrain
trainSet.printDatasetInfo()
trainSampler = RandomSampler(trainSet)
trainLoader = DataLoader(trainSet, sampler=trainSampler,
                    batch_size=p_d.batch, drop_last=True, num_workers=4)

# pretrainPath = "/home/chunyang/projects/autoreg-pde-diffusion/models/models_inc/128_acdm-r20_00/Model.pth"
pretrainPath = ""

p_l = LossParams()
p_me = None
p_md = ModelParamsDecoder(arch="direct-ddpm+Prev", diffSteps=20, diffSchedule="linear", diffCondIntegration="noisy", trainingNoise=0.0)
p_ml = None
if startFromCheckpoint:
    p_t = TrainingParams(epochs=3, lr=0.001)
else:
    p_t = TrainingParams(epochs=3100, lr=0.0001)

model = PredictionModel(p_d, p_t, p_l, p_me, p_md, p_ml,
                        pretrainPath="" if not startFromCheckpoint else pretrainPath,
                        useGPU=useGPU)
model.printModelInfo()

optimizer = torch.optim.Adam(model.parameters(), lr=p_t.lr, weight_decay=p_t.weightDecay)
logger = Logger("ACDM_self_cond_airfoil", addNumber=True)
logger.setup(model, optimizer)
trainHistory = LossHistory("_train", "Training", logger.tfWriter, len(trainLoader),
                                    0, 1, printInterval=1, logInterval=1, simFields=p_d.simFields)


trainer = TrainerDiffusion(model, trainLoader, optimizer, trainHistory, logger.tfWriter, p_t)

print('Starting Training')
logger.saveTrainState(0)


for epoch in range(0, p_t.epochs):
    trainer.trainingStep(epoch)
    logger.saveTrainState(epoch)
    trainHistory.updateAccuracy([p_d,p_t,p_l,p_me,p_md,p_ml], [], epoch==p_t.epochs-1)