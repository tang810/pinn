import torch
import itertools
from functools import partial as func_partial

def partial(func, *args, **kwargs):
    return func_partial(func, *args, **kwargs)

def chain(*iterables):
    return itertools.chain(*iterables)

def load_ckpt(path):
    return torch.load(path)