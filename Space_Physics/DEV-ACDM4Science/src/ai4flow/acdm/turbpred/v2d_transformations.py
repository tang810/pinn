import torch
import torch.nn.functional as F
import numpy as np

from turbpred.params import DataParams


class Transforms(object):

    def __init__(self):
        self.dim = 2
        self.normMean = np.array([0.97496, -0.00572, -0.04187, 0], dtype=np.float32)
        self.normStd =  np.array([0.33951, 0.31578, 0.15982, 1], dtype=np.float32)


    def __call__(self, sample:dict):

        data = sample["data"]
        simParameters = sample["simParameters"]
        allParameters = sample["allParameters"]
        obsMask = sample.get("obsMask", None)
        path = sample["path"]

        filterArrParam = 3

        meanParam = self.normMean[filterArrParam].reshape((1,-1))
        stdParam = self.normStd[filterArrParam].reshape((1,-1))
        simParameters = (simParameters - meanParam) / stdParam

        meanData = self.normMean.reshape((1,-1,1,1)) if self.dim == 2 else self.normMean.reshape((1,-1,1,1,1))
        stdData = self.normStd.reshape((1,-1,1,1)) if self.dim == 2 else self.normStd.reshape((1,-1,1,1,1))

        if self.dim == 2:
            data = (data - meanData) / stdData

        # toTensor
        result = torch.from_numpy(data).to(torch.float32)
        if obsMask is not None:
            obsMask = torch.from_numpy(obsMask)

        outDict = {"data": result, "simParameters": simParameters, "allParameters": allParameters, "path": path}
        if obsMask is not None:
            outDict["obsMask"] = obsMask

        return outDict