import torch
from torch_geometric.nn.data_parallel import DataParallel

def count_parameters_number(model):
    numel = sum([p.numel() for p in model.parameters()])
    print('Model with', numel, 'parameters')

def to_data_parallel(model, device, no_parallel):
    if device == 'cuda' and not no_parallel:
        model = DataParallel(model).to(device)
    return model

def load_state_dict(model, state_dict, mode, device, no_parallel=True):
    model.load_state_dict(state_dict)
    if mode=='train':
        model.train()
    if mode=='eval':
        model.eval()
    
    if device == 'cuda' and not no_parallel:
        model = DataParallel(model).to(device)
    else:
        model.to(device)
    return model