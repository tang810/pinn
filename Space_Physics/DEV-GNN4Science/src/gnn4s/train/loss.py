import torch
import numpy as np
import copy

import gnn4s

class LossLp():
    def __init__(self, model, var_dict, p: int=2):
        self.model = model
        self.var_dict = var_dict
        self.p = p

    def __call__(self, data):
        data.node_pre = self.model(data)
        loss = {}
        for key in self.var_dict['node_label']:
            res = data.node_pre[key] - data.node_label[key]
            loss[f'loss_{key}'] = ((res**self.p).mean()) ** (1/self.p)
        return loss

class LossMS():
    """ mean squared loss """
    def __init__(self, model) -> None:
        """ initialization
        Args:
            model: model
        """
        self.model = model

    def __call__(self, batch_data) -> dict:
        """ calculate the value of the loss function
        Args:
            batch_data: batch data
        Returns:
            value of the loss function
        """
        res = self.model(batch_data) - batch_data.y
        
        mask = batch_data.mask
        loss = (res[mask]**2).sum()
        return {'loss': loss}

class LossMSE():
    def __init__(self, model, stride, bc_type) -> None:
        """ Mean Squared Loss Function.
        Args:
            model: model
        """
        self.model = model
        self.stride = stride
        self.bc_type = bc_type

        self.predictor = gnn4s.infer.PredictorCardiovascular(self.model, self.bc_type)
    
    def mse(self, input, target, mask = None):
        """ Mean square error.
        Args:
            input: first tensor
            target: second tensor (ideally, the result we are trying to match)
            mask: tensor of 1 and 0 with same size as input and target. If not 
                None, selects only components for which it equals 1. 
                Default -> None
        Returns:
            The mean square error
        """
        if mask == None:
            return ((input - target) ** 2).mean()
        return (mask * (input - target) ** 2).mean() 

    def __call__(self, batch_data):
        loss = 0
        
        batch_data_c = copy.deepcopy(batch_data)
        ns = batch_data_c.ndata['next_steps']
        inmask = batch_data.ndata['inlet_mask'].bool()
        outmask = batch_data.ndata['outlet_mask'].bool()

        bccoeff = 100
        mask = torch.ones(ns[:,:,0].shape)
        mask[inmask,0] = mask[inmask,0] * bccoeff
        mask[outmask,0] = mask[outmask,0] * bccoeff
        mask[outmask,1] = mask[outmask,1] * bccoeff

        for istride in range(self.stride):
            nf = self.predictor.perform_timestep(batch_data_c, ns, istride)
            batch_data_c.ndata['nfeatures'][:,0:2] = nf
            coeff = 0.5
            if istride == 0:
                coeff = 1
            loss = loss + coeff * self.mse(nf, ns[:,:,istride], mask)
        return {'loss': loss}

class LossDiffusion():
    def __init__(self, sde_x=None, sde_adj=None, score_x=None, score_adj=None, edge_to_adj=None,
                 symmetric: bool=True, mask_eps: bool=False, reduce_mean: bool=True,
                 likelihood_weighting: bool=False, eps: float=1e-5) -> None:
        """ Loss function for training diffusion model.
        Args:
            sde_x: sde describing the evolution of node features in diffusion process
            sde_adj: sde describing the evolution of adjacency matrices in diffusion process
            score_x: target score function related to node featrues
            score_adj: target score function related to adjacency matrices
            edge_to_adj: function convert edge index list to adjacency matrices
            symmetric: whether to keep adjacency matrix symmetric
            mask_eps: whether to add a mask for features with small value
            reduce_mean: whether to reduce residuals by taking the mean
            likelihood_weighting: whether to multiply the residuals by weights
            eps: tolerance value
        """
        self.sde_x = sde_x
        self.sde_adj = sde_adj
        self.score_x = score_x
        self.score_adj = score_adj
        self.edge_to_adj = edge_to_adj
        self.symmetric = symmetric
        self.mask_eps = mask_eps
        self.reduce_mean = reduce_mean
        self.likelihood_weighting = likelihood_weighting
        self.eps = eps

        # self.reduce_op = (torch.mean if self.reduce_mean else 
        #     lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs))
    
    def symmetrization(self, x):
        x = x.tril(-1)
        x = x + x.transpose(-1,-2)
        return x
    
    def get_mask_eps(self, adj, eps=1e-5):
        mask_eps = torch.abs(adj).sum(-1).gt(eps).to(dtype=torch.float32)
        if len(mask_eps.shape)==3:
            mask_eps = mask_eps[:,0,:]
        return mask_eps

    def add_x_mask_eps(self, x, mask_eps):
        if mask_eps is None:
            mask_eps = torch.ones((x.shape[0], x.shape[1]), device=x.device)
        return x * mask_eps[:,:,None]
    
    def add_adj_mask_eps(self, adj, mask_eps):
        """
        Args:
            adj: [batch_size,nodes_num_max,nodes_num_max] or 
                 [batch_size,channels_num,nodes_num_max,nodes_num_max]
            mask_eps: [batch_size,nodes_num_max]
        Returns:
            adj
        """
        if mask_eps is None:
            mask_eps = torch.ones((adj.shape[0], adj.shape[-1]), device=adj.device)

        if len(adj.shape) == 4:
            mask_eps = mask_eps.unsqueeze(1)  # B x 1 x N
        adj = adj * mask_eps.unsqueeze(-1)
        adj = adj * mask_eps.unsqueeze(-2)
        return adj

    def loss_term(self, sde, z, mask, std, score, t):
        """ Loss term related to node or adjacency matrix.
        Args:
            sde: stochastic differential equation describing the diffusion process
            z: node or adjacency matrix with noise
            mask: mask
            std: standard deviation of the noise distribution
            score: target score function
            t: time
        """
        if mask is None:
            mask = torch.ones(z.shape, device=z.device)
        mask = mask.reshape(mask.shape[0], -1)

        if not self.likelihood_weighting:
            loss = torch.square(score * std + z)
            loss = loss.reshape(loss.shape[0], -1)
            # loss = self.reduce_op(loss*mask, dim=-1)
            if self.reduce_mean:
                loss = torch.sum(loss*mask, dim=-1) / torch.sum(mask, dim=-1)
            else:
                loss = 0.5 * torch.sum(loss*mask, dim=-1)
            loss = loss.mean()
        else:
            g2 = sde.sde(torch.zeros_like(z), t)[1] ** 2
            loss = torch.square(score + z / std)
            loss = loss.reshape(loss.shape[0], -1)
            if self.reduce_mean:
                loss = torch.sum(loss*mask, dim=-1) / torch.sum(mask, dim=-1)
            else:
                loss = 0.5 * torch.sum(loss*mask, dim=-1)
            loss = (loss*g2).mean()
        return loss

    def __call__(self, batch_data):
        """ Compute the value of loss function
        Args:
            batch_data: a mini-batch of training data, including node features, adjacency matrices and mask.
        Returns:
            loss: value of loss function
        """
        x, adj, mask_x, mask_adj = self.edge_to_adj(batch_data)
        # x, adj, mask_x, mask_adj = batch_data
        
        t = torch.rand(adj.shape[0], device=adj.device) * (self.sde_adj.time_end - self.eps) + self.eps

        mask_eps = None
        if self.mask_eps:
            mask_eps = self.get_mask_eps(adj)
        
        if x is not None:
            z_x = torch.randn_like(x)
            z_x = self.add_x_mask_eps(z_x, mask_eps)
            mean_x, std_x = self.sde_x.marginal_prob(x, t)
            perturbed_x = mean_x + std_x * z_x
            perturbed_x = self.add_x_mask_eps(perturbed_x, mask_eps)
        else:
            perturbed_x = None

        if adj is not None:
            z_adj = torch.randn_like(adj)
            if self.symmetric:
                z_adj = self.symmetrization(z_adj)
            z_adj = self.add_adj_mask_eps(z_adj, mask_eps)

            mean_adj, std_adj = self.sde_adj.marginal_prob(adj, t)
            if self.symmetric:
                mean_adj = self.symmetrization(mean_adj)
            perturbed_adj = mean_adj + std_adj * z_adj
            perturbed_adj = self.add_adj_mask_eps(perturbed_adj, mask_eps)
        else:
            perturbed_adj = None
        
        if x is not None:
            # gdss: mask_eps; graphgdp: mask_x
            score_x = self.score_x(perturbed_x, perturbed_adj, mask_x, t)
            loss_x = self.loss_term(self.sde_x, z_x, mask_x, std_x, score_x, t)
        else:
            loss_x = None

        if adj is not None:
            score_adj = self.score_adj(perturbed_x, perturbed_adj, mask_adj, t)
            if self.symmetric and mask_adj is not None:
                mask_adj = self.symmetrization(mask_adj)
            loss_adj = self.loss_term(self.sde_adj, z_adj, mask_adj, std_adj, score_adj, t)
        else:
            loss_adj = None
        
        loss = {}
        if loss_x is not None:
            loss['loss_x'] = loss_x
        if loss_adj is not None:
            loss['loss_adj'] = loss_adj
        
        return loss

class LossScoreProduct():
    def __init__(self, model, t_to_sigma, tra_weight: float=1.0, rot_weight: float=1.0, tor_weight: float=1.0,
                 no_torsion: bool=False) -> None:
        """ Loss function related to the score in product space.
        Args:
            model: model
            t_to_sigma: function for calculate the standard deviation 
                of the prior distribution according to the time
            tra_weight: weight realated to translation term
            rot_weight: weight realated to rotation term
            tor_weight: weight realated to torsion term
            no_torsion: whether to consider torsion
        """
        self.model = model
        self.t_to_sigma = t_to_sigma
        self.tra_weight = tra_weight
        self.rot_weight = rot_weight
        self.tor_weight = tor_weight
        self.no_torsion = no_torsion
                 
    def __call__(self, data):
        """ Calculate the value of loss function.
        Args:
            data: heterogeneous graph containing receptor and ligand information. 
        """
        # predict the score
        tra_pred, rot_pred, tor_pred = self.model(data)

        # calculate the the standard deviation of the prior distribution according to the time
        device = tra_pred.device
        tra_sigma, rot_sigma, tor_sigma = self.t_to_sigma(
            *[torch.cat([d.complex_t[noise_type] for d in data]) if device.type == 'cuda' else data.complex_t[noise_type]
            for noise_type in ['tr', 'rot', 'tor']])
        mean_dims = (0, 1)

        # translation component
        tra_score = torch.cat([d.tr_score for d in data], dim=0) if device.type == 'cuda' else data.tr_score
        tra_sigma = tra_sigma.unsqueeze(-1)
        tra_loss = ((tra_pred.cpu() - tra_score) ** 2 * tra_sigma ** 2).mean(dim=mean_dims)
        
        # rotation component
        rot_score = torch.cat([d.rot_score for d in data], dim=0) if device.type == 'cuda' else data.rot_score
        rot_score_norm = gnn4s.data.so3.score_norm(rot_sigma.cpu()).unsqueeze(-1)
        rot_loss = (((rot_pred.cpu() - rot_score) / rot_score_norm) ** 2).mean(dim=mean_dims)
        
        # torsion component
        if not self.no_torsion:
            edge_tor_sigma = torch.from_numpy(
                np.concatenate([d.tor_sigma_edge for d in data] if device.type == 'cuda' else data.tor_sigma_edge))
            tor_score = torch.cat([d.tor_score for d in data], dim=0) if device.type == 'cuda' else data.tor_score
            tor_score_norm = torch.tensor(gnn4s.data.torus.score_norm(edge_tor_sigma.cpu().numpy())).float()
            tor_loss = ((tor_pred.cpu() - tor_score) ** 2 / tor_score_norm)
            tor_loss = tor_loss.mean() * torch.ones(1, dtype=torch.float)
        else:
            tor_loss = torch.zeros(1, dtype=torch.float)
        
        loss = self.tra_weight*tra_loss + self.rot_weight*rot_loss + self.tor_weight*tor_loss
        return loss