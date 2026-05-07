import torch
import torch.nn as nn

class Activation(nn.Module):
    def __init__(self, actv_type: str='tanh') -> None:
        super().__init__()
        self.actv_type = actv_type
        
        if actv_type == 'tanh':
            self.actv_fn = nn.Tanh()
        elif actv_type == 'elu':
            self.actv_fn = nn.ELU()
        elif actv_type == 'relu':
            self.actv_fn = nn.ReLU()
        elif actv_type == 'silu':
            self.actv_fn = nn.SiLU()
        elif actv_type == 'lrelu':
            self.actv_fn = nn.LeakyReLU(negative_slope=0.2)
        else:
            raise NotImplementedError('activation function does not exist!')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.actv_fn(x)

class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int=1, hidden_dim: int=64, layer_num: int=1, 
                 bias: bool=True, norm: bool=False, activation=nn.Tanh(), dropout_rate: float=0.0,
                 device='cpu') -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.bias = bias
        self.norm = norm
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.device = device
        
        if isinstance(self.activation, str):
            self.activation = Activation(self.activation)
        
        self.dropout = None
        if self.dropout_rate>0.0:
            self.dropout = nn.Dropout(self.dropout_rate)

        lin_layer = []
        in_dim = self.input_dim
        for i in range(self.layer_num):
            lin_layer.append(torch.nn.Linear(in_dim, self.hidden_dim, bias=self.bias))
            in_dim = self.hidden_dim
        lin_layer.append(torch.nn.Linear(in_dim, self.output_dim, bias=self.bias))
        self.lin_layer = torch.nn.Sequential(*lin_layer)
        
        if self.norm:
            self.norm_layer = nn.LayerNorm(self.output_dim)

        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.layer_num):
            x = self.lin_layer[i](x)
            x = self.activation(x)
            if self.dropout is not None:
                x = self.dropout(x)
        x = self.lin_layer[-1](x)
        
        if self.norm:
            x = self.norm_layer(x)
        return x
