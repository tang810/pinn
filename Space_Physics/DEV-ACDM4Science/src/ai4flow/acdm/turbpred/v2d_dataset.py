import numpy as np
import os
from natsort import natsorted
import re
import torch
import einops

from torch.utils.data import Dataset
from turbpred.v2d_transformations import Transforms
from typing import List,Tuple


class V2dDataset(Dataset):
    """Dataset class for loading data from Vort2D data folder.

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

    def get_obsMask(self):
        x_min, x_max = -7.5, 28.5
        y_min, y_max = -20.0, 20.0

        # Define grid size
        grid_size = 256

        # Create a grid of points
        x = np.linspace(x_min, x_max, grid_size)
        y = np.linspace(y_min, y_max, grid_size)
        X, Y = np.meshgrid(x, y)

        # Calculate distance of each point from the center (0,0)
        distance_from_center = np.sqrt(X**2 + Y**2)

        # Create mask where points inside the circle are set to 0 and outside are set to 1
        mask = np.where(distance_from_center <= 0.5, 0, 1)
        return mask[np.newaxis, :, :]

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
        res = einops.rearrange(img[:, 95:159, 30:158], 'c h w -> c w h')
        return res
