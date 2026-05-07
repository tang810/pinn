import traceback
from src.model import PINNsformer
import torch

try:
    model = PINNsformer(d_model=64, d_hidden=512, N=1, heads=2)
    state = torch.load('model/pinnsformer_withsun.pt', map_location='cpu')
    print('Dict Keys loaded.')
    missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
    print('Missing keys:', missing_keys)
    print('Unexpected keys:', unexpected_keys)
except Exception as e:
    print('---ERROR---')
    print(traceback.format_exc())
