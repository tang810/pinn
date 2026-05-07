import abc
import torch
import numpy as np
from .sde import VPSDE, subVPSDE

_CORRECTORS = {}
_PREDICTORS = {}

class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score, probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        if isinstance(sde, tuple):
            self.rsde = (sde[0].reverse(score, probability_flow),
                         sde[1].reverse(score, probability_flow))
        else:
            self.rsde = sde.reverse(score, probability_flow)
        self.score = score

    @abc.abstractmethod
    def update_fn(self, x, mask, t):
        """One update of the predictor.

        Args:
            x: A PyTorch tensor representing the current state.
            t: A PyTorch tensor representing the current time step.

        Returns:
            x: A PyTorch tensor of the next state.
            x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass

class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, score, signal_noise_ratio: float=0.16, scale_eps: float=1e-4, time_steps_num: int=1):
        super().__init__()
        self.sde = sde
        self.score = score
        self.signal_noise_ratio = signal_noise_ratio
        self.scale_eps = scale_eps
        self.time_steps_num = time_steps_num

    @abc.abstractmethod
    def update_fn(self, x, mask, t):
        """One update of the corrector.

        Args:
            x: A PyTorch tensor representing the current state.
            t: A PyTorch tensor representing the current time step.

        Returns:
            x: A PyTorch tensor of the next state.
            x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass

def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(f'Already registered predictor with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)

def register_corrector(cls=None, *, name=None):
    """A decorator for registering corrector classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _CORRECTORS:
            raise ValueError(f'Already registered corrector with name: {local_name}')
        _CORRECTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)

@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score, probability_flow=False):
        super().__init__(sde, score, probability_flow)

    def update_fn(self, x, mask, t):
        dt = -1. / self.rsde.time_steps_num
        z = torch.randn_like(x)
        z = torch.tril(z, -1)
        z = z + z.transpose(-1, -2)
        drift, diffusion = self.rsde.sde(None, x, mask, t, is_adj=True)
        drift = torch.tril(drift, -1)
        drift = drift + drift.transpose(-1, -2)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
        return x, x_mean

@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score, probability_flow=False, is_adj=False):
        super().__init__(sde, score, probability_flow)
        self.is_adj = is_adj

    def update_fn(self, x, adj, mask, t):
        f, g = self.rsde.discretize(x, adj, mask, t, self.is_adj)

        if self.is_adj:
            z = gen_noise(adj, mask)
            adj_mean = adj - f
            g_ = g[:,None,None] if len(z.shape)==3 else g[:,None,None,None]
            adj = adj_mean + g_ * z
            return adj, adj_mean
        else:
            z = gen_noise(x, mask, sym=False)
            x_mean = x - f
            g_ = g[:,None,None] if len(z.shape)==3 else g[:,None,None,None]
            x = x_mean + g_ * z
            return x, x_mean

@register_predictor(name='none')
class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, sde, score, probability_flow=False):
        pass

    def update_fn(self, x, mask, t):
        return x, x

@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
    def __init__(self, sde, score_fn, signal_noise_ratio: float=0.16, scale_eps: float=1e-4,
                 time_steps_num: int=1, is_adj=False):
        super().__init__(sde, score_fn, signal_noise_ratio, scale_eps, time_steps_num)
        self.is_adj = is_adj

    def update_fn(self, x, adj, mask, t):
        if isinstance(self.sde, VPSDE) or isinstance(self.sde, subVPSDE):
            timestep = (t * (self.sde.time_steps_num - 1) / self.sde.time_end).long()
            alpha = self.sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)
        
        if self.is_adj:
            for i in range(self.time_steps_num):
                grad = self.score(x, adj, mask, t)
                noise = gen_noise(adj, mask)
                grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
                step_size = (self.signal_noise_ratio * noise_norm / grad_norm) ** 2 * 2 * alpha
                adj_mean = adj + step_size[:,None,None] * grad
                adj = adj_mean + torch.sqrt(step_size*2)[:,None,None] * noise * self.scale_eps
            return adj, adj_mean
        else:
            for i in range(self.time_steps_num):
                grad = self.score(x, adj, mask, t)
                noise = gen_noise(x, mask, sym=False)
                grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
                step_size = (self.signal_noise_ratio * noise_norm / grad_norm) ** 2 * 2 * alpha
                x_mean = x + step_size[:,None,None] * grad
                x = x_mean + torch.sqrt(step_size*2)[:,None,None] * noise * self.scale_eps
            return x, x_mean

@register_corrector(name='none')
class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, sde, score, signal_noise_ratio: float=0.16, scale_eps: float=1e-4, time_steps_num: int=1):
        pass

    def update_fn(self, x, mask, t):
        return x, x

def mask_x(x, mask):
    if mask is None:
        mask = torch.ones((x.shape[0], x.shape[1]), device=x.device)
    return x * mask[:,:,None]

def mask_adj(adj, mask):
    if mask is None:
        mask = torch.ones((adj.shape[0],adj.shape[-1]), device=adj.device)

    if len(adj.shape) == 4:
        mask = mask.unsqueeze(1)  # [batch_size,1,node_num]
    adj = adj * mask.unsqueeze(-1)
    adj = adj * mask.unsqueeze(-2)
    return adj

def gen_noise(x, mask, sym=True):
    z = torch.randn_like(x)
    if sym:
        z = z.triu(1)
        z = z + z.transpose(-1,-2)
        z = mask_adj(z, mask)
    else:
        z = mask_x(z, mask)
    return z
