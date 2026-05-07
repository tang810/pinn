import torch

import gnn4s

class Score():
    def __init__(self, model, sde, train: bool=False, continuous: bool=False) -> None:
        """ Fractional function that the diffusion model aim to learn.
        Args:
            model: diffusion model
            sde: sde describing the evolution of the diffusion process
            train: whether to train or evaluate the mode
            continue: Whether to use a continuous or discrete computation approach
        """
        self.model = model
        self.sde = sde
        self.train = train
        self.continuous = continuous

        if self.train:
            self.model.train()
        else:
            self.model.eval()

        if not (isinstance(self.sde, gnn4s.diffusion.VPSDE) or isinstance(self.sde, gnn4s.diffusion.subVPSDE) or
            isinstance(self.sde, gnn4s.diffusion.VESDE)):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
        
        if isinstance(self.sde, gnn4s.diffusion.VPSDE) or isinstance(self.sde, gnn4s.diffusion.subVPSDE):
            if self.continuous or isinstance(self.sde, gnn4s.diffusion.subVPSDE):
                def score_fn(x, adj, mask, t):
                    # For VP-trained models, t=0 corresponds to the lowest noise level
                    # The maximum value of time embedding is assumed to 999 for continuously-trained models.
                    labels = t * 999
                    score = self.model(x, adj, mask, labels)
                    std = self.sde.marginal_prob(torch.zeros_like(adj), t)[1]
                    score = -score / std
                    return score
            else:
                def score_fn(x, adj, mask, t):
                    # For VP-trained models, t=0 corresponds to the lowest noise level
                    labels = t * (self.sde.N - 1)
                    score = self.model(x, adj, mask, labels)
                    std = self.sde.sqrt_1m_alpha_cumprod.to(labels.device)[labels.long()]
                    score = -score / std[:, None, None, None]
                    return score
        
        if isinstance(self.sde, gnn4s.diffusion.VESDE):
            if continuous:
                def score_fn(x, adj, mask, t):
                    score = self.model(x, adj, mask)
                    return score
                '''
                def score_fn(x, t, *args, **kwargs):
                    labels = self.sde.marginal_prob(torch.zeros_like(x), t)[1]
                    score = self.model(x, labels, *args, **kwargs)
                    return score
                '''
            else:
                def score_fn(x, adj, mask, t):
                    # For VE-trained models, t=0 corresponds to the highest noise level
                    labels = self.sde.time_end - t
                    labels *= self.sde.time_step_num - 1
                    labels = torch.round(labels).long()
                    score = self.model(x, adj, mask, labels)
                    return score
        
        self.score_fn = score_fn

    def __call__(self, x, adj, mask, t):
        """ Calulate the value that the diffusion model aim to learn.
        Args:
            x: node features
            adj: adjacency matrices
            mask: mask
            t: time
        """
        return self.score_fn(x, adj, mask, t)
