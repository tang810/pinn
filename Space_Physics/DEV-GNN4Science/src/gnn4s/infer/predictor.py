import torch
import numpy as np
import copy

import gnn4s

class PredictorEuler():
    """ predictor of euler system """
    def __init__(self, model, device):
        """ initialization
        Args:
            model: model
            device: computing device
        """
        self.model = model
        self.device = device
        
        self.model.eval()
    
    def __call__(self, data):
        """ Predict the solution from the initial time step to the final time step
        Args:
            data: data
        """
        predict = []

        vel_cur = data[0].x[:,1:]
        for step in range(len(data)):
            # copy graph
            g = copy.deepcopy(data[step])
            if step>0:
                g.x[:,1:] = vel_next
                vel_cur = vel_next
            g = g.to(self.device)

            # forward propagation
            acceleration = self.model(g)
            vel_next = vel_cur.to(self.device) + acceleration
            
            # add mask
            mask = torch.logical_not(g.mask)
            vel_next[mask] = g.y[mask]
            
            # store
            vel_next = vel_next.cpu().detach()
            predict.append(vel_next)
        
        return predict

class PredictorLagrange():
    """ predictor of lagrange system """
    def __init__(self, model, device):
        """ initialization
        Args:
            model: model
            device: computing device
        """
        self.model = model
        self.device = device

        self.model.eval()
    
    def __call__(self, data):
        """ predicting the solution from the initial time step to the final time step
        Args:
            data: data
        """
        predict = []

        pos_prev = data[0].world_pos_prev
        pos_cur = data[0].world_pos
        for step in range(len(data)):
            # copy graph
            g = copy.deepcopy(data[step])
            if step>0:
                g.world_pos = pos_next
                g.x[:,1:] = pos_next - pos_cur
                pos_prev = pos_cur
                pos_cur = pos_next
            g = g.to(self.device)

            # forward propagation
            acceleration = self.model(g)
            pos_next = (2*pos_cur - pos_prev).to(self.device) + acceleration
            
            # add mask
            mask = torch.logical_not(g.mask)
            pos_next[mask] = (pos_cur.to(self.device))[mask]
            
            # store
            pos_next = pos_next.cpu().detach()
            predict.append(pos_next)
            
        return predict

class PredictorCardiovascular():
    def __init__(self, model, bc_type):
        """ Predictor of lagrange system
        Args:
            model: model
            bc_type: boundary condition type
        """
        self.model = model
        self.bc_type = bc_type

        self.model.eval()

    def get_bc_mask(self, graph, time_index):
        """ get boundary conditions mask
        Args:
            graph: DGL graph
            time index: timestep index
        """
        mask = (torch.zeros(graph.ndata['outlet_mask'].shape[0],2)==1)
        if self.bc_type == 'realistic_dirichlet':
            mask[graph.ndata['outlet_mask'].bool(),0] = True
            mask[graph.ndata['inlet_mask'].bool(),1] = True
        elif self.bc_type == 'full_dirichlet':
            mask[graph.ndata['inlet_mask'].bool(),0] = True
            mask[graph.ndata['outlet_mask'].bool(),0] = True
            mask[graph.ndata['inlet_mask'].bool(),1] = True
            mask[graph.ndata['outlet_mask'].bool(),1] = True
        return mask
        
    def perform_timestep(self, graph, bc, time_index, set_bc=True):
        """ Performs a single timestep of the rollout phase.
        Args:
            graph: DGL graph
            bc: 3D array containing the boundary conditions. 
                dim 1: node index, dim 2: pressure (0) and flow rate (1),
                dim 3: timesteps
            time index: index of timestep where we have to take the boundary
                        conditions from
            set_bc: set boundary conditions.
        Returns:
            2D array where dim 1 corresponds to node indices, 
            and dim 2 corresponds to pressure (0) and flow rate (1)
        """
        gnf = graph.ndata['nfeatures'][:,0:2]
        graph.ndata['next_flowrate'] = bc[:,1,time_index]

        if 'dirichlet' in self.bc_type:
            mask = self.get_bc_mask(graph, time_index)
            gnf[mask] = bc[:,:,time_index][mask]
        elif self.bc_type == 'physiological':
            mask = graph.ndata['inlet_mask'].bool()

        gnf = gnf + self.model(graph)

        if set_bc:
            if 'dirichlet' in self.bc_type:
                gnf[mask] = bc[:,:,time_index][mask]
            elif self.bc_type == 'physiological':
                gnf[mask,1] = bc[:,:,time_index][mask,1]

        return gnf

    def compute_average_branches(self, graph, flowrate):
        """ Average flowrate over branch nodes.
        Args:
            graph: DGL graph
            flowrate: nodal flow rate
        """
        branch_id = graph.ndata['branch_id'].detach().numpy()
        bmax = np.max(branch_id)
        for i in range(bmax + 1):
            idxs = np.where(branch_id==i)[0]
            rflowrate = torch.mean(flowrate[idxs])
            flowrate[idxs] = rflowrate

    def __call__(self, graph, average_branches=True):
        """ Performs rollout phase.
        Args:
            graph: DGL graph
            average_branches: if Trues, averages flowrate over branch nodes.
        Returns:
            2D array of reconstructed features, where dim 1 corresponds to node 
                indices and dim 2 corresponds to pressure (0) and flow rate (1),
            2D array containing normalized pressure and flow rate relative errors
            2D array containing pressure and flow rate relative errors
            2D array containing the difference of reconstructed and actual features
        """
        times = graph.ndata['nfeatures'].shape[2]
        graph = copy.deepcopy(graph)
        graph_truth = copy.deepcopy(graph)

        gnf_truth = graph_truth.ndata['nfeatures'].clone()
        graph.ndata['nfeatures'] = gnf_truth[:,:,0].clone()
        graph.edata['efeatures'] = graph_truth.edata['efeatures'].squeeze().clone()

        gnf = graph.ndata['nfeatures'][:,0:2].unsqueeze(axis=2).clone()
        for it in range(times-1):
            graph.ndata['nfeatures'][:,-1] = gnf_truth[:,-1,it]
            gf = self.perform_timestep(graph, gnf_truth, it + 1)

            if average_branches:
                self.compute_average_branches(graph, gf[:,1])

            graph.ndata['nfeatures'][:,0:2] = gf
            gnf = torch.cat((gnf, gf.unsqueeze(axis = 2)), axis = 2)
        
        return gnf.detach().numpy()