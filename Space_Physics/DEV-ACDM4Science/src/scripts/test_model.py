import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, SequentialSampler
from ai4flow.dataset.airfoil_acdm import airfoilDataset
from ai4flow.acdm.turbpred.model import PredictionModel, DiffusionModel
from ai4flow.acdm.turbpred.params import DataParams, TrainingParams, LossParams, ModelParamsEncoder, ModelParamsDecoder, ModelParamsLatent
from ai4flow.dataset.airfoil_acdm import Transforms

# global switchs:
self_cond = True  # use original Ux Uy as the conditioner

# global vars:
device = "cuda:0" if torch.cuda.is_available() else "cpu"
test_dataset_path = "/home/chunyang/projects/DDPM4Science/data/airfoil_stable/test"
model_weight_path = "/home/chunyang/projects/DDPM4Science/runs/ACDM_self_cond_airfoil_04/Model_E1600.pth"

# relative error comparison

## first test dataset:
testSet = airfoilDataset(name="V2D dataset",
                     dataDir=test_dataset_path,
                     sequenceLength=[3, 2], rey=1, self_cond=self_cond)
testTransformations = Transforms()
testSet.transform = testTransformations
testSampler = SequentialSampler(testSet)
testLoader = DataLoader(testSet, sampler=testSampler, batch_size=len(testSet), drop_last=False)

## load model:
p_d = DataParams(batch=32, augmentations=["normalize"], sequenceLength=[3,2], randSeqOffset=True,
            dataSize=[128,64], dimension=2, simFields=["pres"], simParams=["rey", "u_x", "u_y"], normalizeMode="incMixed")
p_t = TrainingParams(epochs=3, lr=0.001)
startFromCheckpoint = False
useGPU = True
gpuID = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = gpuID
p_l = LossParams()
p_me = None
p_md = ModelParamsDecoder(arch="direct-ddpm+Prev", diffSteps=20, diffSchedule="linear", diffCondIntegration="noisy", trainingNoise=0.0)
p_ml = None

model = PredictionModel(p_d, p_t, p_l, p_me, p_md, p_ml,
                        pretrainPath="",
                        useGPU=useGPU)
# model.printModelInfo()

model = PredictionModel.load(model_weight_path, useGPU=True)
model.to(device)
model.eval()

## make prediction, for each samples in the dataset:
evalOptions = {"numEvals": 1,
    "sequentialEvalRuns": {"lowRey": True, "highRey": True, "varReyIn": True},
    "samplingMode": "ddpm",
    "posteriorSampling": "random",
    "initialSampling": "random",
    "conditioningIntegration": "noisy"
},

if isinstance(model.modelDecoder, DiffusionModel):
    if "samplingMode" in evalOptions:
        model.modelDecoder.inferenceSamplingMode = evalOptions["samplingMode"]
    if "posteriorSampling" in evalOptions:
        model.modelDecoder.inferencePosteriorSampling = evalOptions["posteriorSampling"]
    if "initialSampling" in evalOptions:
        model.modelDecoder.inferenceInitialSampling = evalOptions["initialSampling"]
    if "conditioningIntegration" in evalOptions:
        model.modelDecoder.inferenceConditioningIntegration = evalOptions["conditioningIntegration"]

elif isinstance(model.modelDecoder, torch.nn.ModuleList):
    for module in model.modelDecoder:
        if isinstance(module, DiffusionModel):
            if "samplingMode" in evalOptions:
                module.inferenceSamplingMode = evalOptions["samplingMode"]
            if "posteriorSampling" in evalOptions:
                module.inferencePosteriorSampling = evalOptions["posteriorSampling"]
            if "initialSampling" in evalOptions:
                module.inferenceInitialSampling = evalOptions["initialSampling"]
            if "conditioningIntegration" in evalOptions:
                module.inferenceConditioningIntegration = evalOptions["conditioningIntegration"]

with torch.no_grad():
    predSamples = []
    for s, sample in enumerate(testLoader, 0):
        data = sample["data"].to(device)
        simParameters = sample["simParameters"].to(device) if type(sample["simParameters"]) is not dict else None
        prediction, _, _ = model(data, simParameters)
        # print("jijiji", data.shape, prediction.shape)
        prediction = prediction.unsqueeze(0).unsqueeze(0)
        predSamples += [prediction.cpu().numpy()]
    predSamples = np.concatenate(predSamples, axis=2)
    print("sampling finished")

preds = predSamples.squeeze().squeeze()

# for i in range 
