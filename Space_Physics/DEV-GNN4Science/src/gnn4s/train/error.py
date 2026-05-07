import torch
import numpy as np

import gnn4s

class ErrorRelativeLp():
    def __init__(self, model, var_dict, p: int=2):
        self.model = model
        self.var_dict = var_dict
        self.p = p

    def __call__(self, data):
        data.node_pre = self.model(data)
        error = {}
        for key in self.var_dict['node_label']:
            res = data.node_pre[key] - data.node_label[key]
            err = ((res**self.p).sum() / 
                   (data.node_label[key]**self.p).sum()
                   ) ** (1/self.p)
            error[f'error_{key}'] = err.detach()
        return error