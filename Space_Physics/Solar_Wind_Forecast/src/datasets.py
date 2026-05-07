import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    def __init__(self, arr, span=24, prior=12):
        self.arr = torch.as_tensor(arr, dtype=torch.float32)
        self.span = span
        self.prior = prior
        self.T, self.D = self.arr.shape
        self.idxs = list(range(self.T - (span + prior)))

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        t = self.idxs[i]
        x = self.arr[t : t + self.span]                 # (span, D)
        y = self.arr[t + self.span + self.prior - 1]    # (D,)
        return x, y
