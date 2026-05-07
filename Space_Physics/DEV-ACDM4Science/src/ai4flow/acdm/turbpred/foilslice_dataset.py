import numpy as np
import os
from natsort import natsorted
import re
import torch
import einops
from scipy.ndimage import zoom

from torch.utils.data import Dataset
from typing import List,Tuple


class FoilsliceDataset(Dataset):
    """Dataset class for loading data from FoilSlice data folder.

    Args:
        Dataset (_type_): _description_
    """

    def __init__(self, name:str, dataDir:str, idx_range:List[int], sequenceLength:List[int], rey=1000):
        # Params setup
        self.transform = None
        self.name = name
        self.dataDir = dataDir
        self.idx_range = idx_range
        self.seqLength = sequenceLength[0]
        self.seqSkip = sequenceLength[1]
        self.rey = rey

        # load data
        print("Loading data from \t", self.dataDir)
        self.file_paths = []
        self.sample_paths = []
        for x in os.listdir(self.dataDir):
            file_path = os.path.join(
                self.dataDir, x
            )
            match = re.search(r'\d+(?=.npy)', file_path)
            digits = int(match.group())
            if (digits < self.idx_range[1] and digits >= self.idx_range[0]):
                self.file_paths.append(file_path)
        self.file_paths = natsorted(self.file_paths)
        print(f"Loading completed. Files loaded: \t{len(self.file_paths)}")

        # group data to group of samples
        for idx in range(0, len(self.file_paths), self.seqLength*self.seqSkip):
            group = []
            if idx + self.seqSkip * self.seqLength > len(self.file_paths):
                break
            for i in range(self.seqLength):
                group.append(self.file_paths[idx + i*(self.seqSkip)])
            self.sample_paths.append(group)
        print(f"Loading completed. Files loaded: \t{len(self.sample_paths)}")

        print()
        self.printDatasetInfo()

        return

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx:int) -> dict:
        # get data
        samples = []
        for i, pth in enumerate(self.sample_paths[idx]):
            sample = np.load(pth)
            # append extra dimension - Reynold number layer:
            rey_layer = np.ones_like(sample[0][np.newaxis, :, :])
            sample = np.concatenate([sample, rey_layer], axis=0)
            sample = zoom(sample, (1, 0.4 , 0.4), order=1)
            # print("sample size", sample.shape)
            sample = self.crop_img(sample) # crop image to ideal size (128, 64)
            samples.append(sample)
        data = np.stack(samples, axis=0)
        # print(data.shape)

        # get simParameters
        simParam = np.array([[self.rey] for i in range(3)], dtype=np.float32)

        # get allParameters
        allParam = {
            'Reynolds Number': np.array([self.rey, self.rey, self.rey], dtype=np.float32),
            'Mach Number': np.array([0., 0., 0.], dtype=np.float32),
            'Drag Coefficient': np.array([0., 0., 0.], dtype=np.float32),
            'Lift Coefficient': np.array([0., 0., 0.], dtype=np.float32),
            'Z Slice': np.array([0., 0., 0.], dtype=np.float32)
            }

        # get obstacle mask and perform cropping:
        obsMask = self.get_obsMask()
        obsMask = self.crop_img(obsMask).squeeze()

        res = {
            'data': data,
            'simParameters': simParam,
            'allParameters': allParam,
            'path': self.sample_paths[idx],
            'obsMask': obsMask,
        }

        if self.transform:
            res = self.transform(res)
        else:
            print("WARNING: no data transformations are employed!")

        return res

    def get_obsMask(self, source_data_path="/home/chunyang/projects/ai4science/ACDM4Science/data/foil_slice_processed/foilslice_00000.npy"):
        data = np.load(source_data_path)
        # print("original data shape", data.shape)
        data = zoom(data, (1, 0.4 , 0.4), order=1)
        mask = (data[0] != 0)
        mask = mask[np.newaxis, :, :]
        return mask

    def printDatasetInfo(self):
        print("Dataset info detail:")
        print(f"\t Sequence length of each sample: {self.seqLength}")
        print(f"\t Sequence skiping samples of: {self.seqSkip}")

    def crop_img(self, img):
        """Corp a 256*256 image to 128*64

        Args:
            img (ndarray): the input image, of shape (channel, 256, 256)

        Returns:
            res (ndarray): croped version image, shape: (channel, 128, 64)
        """
        height = 128
        width = 64
        # print("incoming img size", img.shape)
        res = einops.rearrange(img[:, 150:150+width, 150:150+height], 'c h w -> c w h')
        # print("output img size: ", res.shape)
        return res


class Transforms(object):

    def __init__(self):
        self.dim = 2
        self.normMean = np.array([9248.837052, -109.30064, -1077520.188, 0], dtype=np.float32)
        self.normStd =  np.array([1595.112, 521.837, 5346411.39068, 1], dtype=np.float32)


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