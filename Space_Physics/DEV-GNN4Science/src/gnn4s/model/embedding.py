import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels.
    """
    def __init__(self, embd_size: int=256, embd_scale: float=1.0) -> None:
        super().__init__()
        self.embd_size = embd_size
        self.embd_scale = embd_scale

        self.weight = nn.Parameter(torch.randn(embd_size//2) * self.embd_scale, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = x[:, None] * self.weight[None, :] * 2 * np.pi
        embd = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return embd

class TimeStepEmbedding():
    def __init__(self, embd_type: str='sinusoidal', embd_dim: int=32, embd_scale: float=10000) -> None:
        """ Time step embedding.
        Args:
            embd_type: type of embedding
            embd_dim: dimension of embedding
            embd_scale: scale of embedding
        """
        self.embd_type = embd_type
        self.embd_dim = embd_dim
        self.embd_scale = embd_scale
        
        if self.embd_type == 'sinusoidal':
            self.embd_func = (lambda x : self.sinusoidal_embedding(self.embd_scale * x, self.embd_dim))
        elif self.embd_type == 'fourier':
            self.embd_func = GaussianFourierProjection(embd_size=self.embd_dim, embd_scale=self.embd_scale)
    
    def sinusoidal_embedding(self, timesteps: torch.Tensor, embd_dim: int=32, max_pos: float=10000
                             ) -> torch.Tensor:
        """ Sinusoidal embedding.
        Args:
            timesteps: time step
            embd_dim: dimension of embedding
            max_pos: maximum position
        """
        assert len(timesteps.shape) == 1
        half_dim = embd_dim // 2
        embd = math.log(max_pos) / (half_dim - 1)
        embd = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -embd)
        embd = timesteps.float()[:, None] * embd[None, :]
        embd = torch.cat([torch.sin(embd), torch.cos(embd)], dim=1)
        if embd_dim % 2 == 1:  # zero pad
            embd = F.pad(embd, (0, 1), mode='constant')
        assert embd.shape == (timesteps.shape[0], embd_dim)
        return embd
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """ Generate embedding.
        Args:
            x: input
        """
        return self.embd_func(x)
