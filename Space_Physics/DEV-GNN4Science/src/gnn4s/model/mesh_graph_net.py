import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import dgl
from torch_geometric.data import Data as PyGData
from torch_scatter import scatter_add
from types import SimpleNamespace

from .mlp import MLP, Activation
from .normalizer import Normalizer

class NodeBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(NodeBlock, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.mlp = MLP(self.input_dim, self.hidden_dim, self.hidden_dim,
                       layer_num=2, norm=True, activation=nn.ReLU())

    def forward(self, graph):
        edge_feat = graph.edge_feat
        
        _, recv_idx = graph.edge_index
        agg_recv_feat = scatter_add(edge_feat, recv_idx, dim=0)

        node_feat_all = torch.cat([graph.node_feat,agg_recv_feat], dim=-1)
        node_feat = self.mlp(node_feat_all)
        return PyGData(node_feat=node_feat, edge_feat=edge_feat,
                       edge_index=graph.edge_index)

class EdgeBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(EdgeBlock, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.mlp = MLP(self.input_dim, self.hidden_dim, self.hidden_dim,
                       layer_num=2, norm=True, activation=nn.ReLU())

    def forward(self, graph):
        send_idx, recv_idx = graph.edge_index
        node_send_feat = graph.node_feat[send_idx]
        node_recv_feat = graph.node_feat[recv_idx]

        edge_feat_all = torch.cat([node_send_feat,node_recv_feat,
                                   graph.edge_feat], dim=-1)
        edge_feat = self.mlp(edge_feat_all)
        return PyGData(node_feat=graph.node_feat, edge_feat=edge_feat,
                       edge_index=graph.edge_index)

class Encoder(nn.Module):
    def __init__(self, edge_input_dim=128, node_input_dim=128, hidden_dim=128):
        super(Encoder, self).__init__()
        self.edge_input_dim = edge_input_dim
        self.node_input_dim = node_input_dim
        self.hidden_dim = hidden_dim
        self.node_embd = MLP(self.node_input_dim, self.hidden_dim, self.hidden_dim,
                             layer_num=2, norm=True, activation=nn.ReLU())
        self.edge_embd = MLP(self.edge_input_dim, self.hidden_dim, self.hidden_dim,
                             layer_num=2, norm=True, activation=nn.ReLU())
        
    def forward(self, graph):
        node_feat = self.node_embd(graph.node_feat)
        edge_feat = self.edge_embd(graph.edge_feat)
        return PyGData(node_feat=node_feat, edge_feat=edge_feat, 
                       edge_index=graph.edge_index)

class Processer(nn.Module):
    def __init__(self, hidden_dim=128):
        super(Processer, self).__init__()
        self.hidden_dim = hidden_dim

        self.nb_input_dim = 2 * self.hidden_dim
        self.eb_input_dim = 3 * self.hidden_dim
        self.node_embd = NodeBlock(self.nb_input_dim, self.hidden_dim)
        self.edge_embd = EdgeBlock(self.eb_input_dim, self.hidden_dim)

    def forward(self, graph):
        node_feat = graph.node_feat
        edge_feat = graph.edge_feat
        graph = self.edge_embd(graph)
        graph = self.node_embd(graph)
        node_feat = node_feat + graph.node_feat
        edge_feat = edge_feat + graph.edge_feat
        return PyGData(node_feat=node_feat, edge_feat=edge_feat, 
                       edge_index=graph.edge_index)

class Decoder(nn.Module):
    def __init__(self, hidden_dim=128, output_dim=2):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.mlp = MLP(self.hidden_dim, self.output_dim, self.hidden_dim,
                       layer_num=2, activation=nn.ReLU())

    def forward(self, graph):
        return self.mlp(graph.node_feat)

class MeshGraphNet(nn.Module):
    """ MeshGraphNet model """
    def __init__(self, node_input_dim: int=1, edge_input_dim: int=2, output_dim: int=2,
                 hidden_dim: int=128, block_num: int=1, use_output_normalizer: bool=False
                 ) -> None:
        super(MeshGraphNet, self).__init__()
        """ initialization
        Args:
            node_input_dim: dimension of node input features
            edge_input_dim: dimension of edge input features
            output_dim: dimension of output
            hidden_dim: dimension of hidden block
            block_num: number of hidden block
        """
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.block_num = block_num
        self.use_output_normalizer = use_output_normalizer

        self.encoder = Encoder(edge_input_dim=self.edge_input_dim,
                               node_input_dim=self.node_input_dim, 
                               hidden_dim=self.hidden_dim)
        
        self.processer = nn.ModuleList()
        for _ in range(self.block_num):
            self.processer.append(Processer(hidden_dim=self.hidden_dim))
        
        self.decoder = Decoder(hidden_dim=self.hidden_dim, output_dim=output_dim)
        
        self._node_normalizer = Normalizer(size=self.node_input_dim)
        self._edge_normalizer = Normalizer(size=self.edge_input_dim)
        if use_output_normalizer:
            self._output_normalizer = Normalizer(size=self.output_dim)
        else:
            self._output_normalizer = None
        
    def update_node_feat(self, node_feat, node_type: torch.Tensor):
        node_type = torch.squeeze(node_type.long())
        node_type_one_hot = torch.nn.functional.one_hot(node_type, 9)
        
        node_feat = torch.cat([node_feat,node_type_one_hot], dim=-1)
        node_feat = self._node_normalizer(node_feat, self.training)
        return node_feat

    def forward(self, graph: PyGData):
        """ Forward propogation
        Args:
            graph: graph data
        """
        node_type = graph.x[:,:1]
        node_attr = graph.x[:,1:]
        
        # construct graph node
        graph.node_feat = self.update_node_feat(node_attr, node_type)
        
        send_idx, recv_idx = graph.edge_index
        if hasattr(graph, 'world_pos'):
            relative_world_pos = graph.world_pos[send_idx,:] - graph.world_pos[recv_idx,:]
            relative_mesh_pos = graph.mesh_pos[send_idx,:] - graph.mesh_pos[recv_idx,:]
            edge_attr = torch.cat([relative_world_pos, torch.norm(relative_world_pos, dim=1, keepdim=True),
                                   relative_mesh_pos, torch.norm(relative_mesh_pos, dim=1, keepdim=True)],-1)
        else:
            relative_mesh_pos = graph.mesh_pos[send_idx,:] - graph.mesh_pos[recv_idx,:]
            edge_attr = torch.cat([relative_mesh_pos, torch.norm(relative_mesh_pos, dim=1, keepdim=True)],-1)

        graph.edge_feat = self._edge_normalizer(edge_attr, self.training)

        graph = self.encoder(graph)
        for processer in self.processer:
            graph = processer(graph)
        node_update = self.decoder(graph)

        if self._output_normalizer is None:
            return node_update
        else:
            return self._output_normalizer.inverse(node_update)

    def to(self, device):
        self.encoder.to(device)
        self.processer.to(device)
        self.decoder.to(device)
        self._node_normalizer.to(device)
        self._edge_normalizer.to(device)
        if self.use_output_normalizer:
            self._output_normalizer.to(device)
    
    def get_model_config(self):
        """ get model configiration """
        model_config_dict = {}
        model_config_dict['node_input_dim'] = self.node_input_dim
        model_config_dict['edge_input_dim'] = self.edge_input_dim
        model_config_dict['output_dim'] = self.output_dim
        model_config_dict['hidden_dim'] = self.hidden_dim
        model_config_dict['block_num'] = self.block_num
        model_config_dict['use_output_normalizer'] = self.use_output_normalizer
        model_config_namespace = SimpleNamespace(**model_config_dict)
        return model_config_namespace
    
    def state_dict(self):
        state_dict = {}
        state_dict['encoder'] = self.encoder.state_dict()
        state_dict['processer'] = self.processer.state_dict()
        state_dict['decoder'] = self.decoder.state_dict()
        state_dict['_node_normalizer'] = self._node_normalizer.state_dict()
        state_dict['_edge_normalizer'] = self._edge_normalizer.state_dict()
        if self.use_output_normalizer:
            state_dict['_output_normalizer'] = self._output_normalizer.state_dict()
        return state_dict
    
    def load_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict['encoder'])
        self.processer.load_state_dict(state_dict['processer'])
        self.decoder.load_state_dict(state_dict['decoder'])
        self._node_normalizer.load_state_dict(state_dict['_node_normalizer'])
        self._edge_normalizer.load_state_dict(state_dict['_edge_normalizer'])
        if self.use_output_normalizer:
            self._output_normalizer.load_state_dict(state_dict['_output_normalizer'])
'''
class DglEncodeProcessDecode(nn.Module):
    """ EncodeProcessDecodeNetwork
        This class computes applies a decode-process-decode sequence to 
        node and edge features in a graph
    Args:
    """
    def __init__(self, node_input_dim, edge_input_dim, output_dim, hidden_dim_gnn, hidden_dim_mlp, 
                 block_num_gnn, layer_num_mlp):
        super(DglEncodeProcessDecode, self).__init__()
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.output_dim = output_dim
        self.hidden_dim_gnn = hidden_dim_gnn
        self.hidden_dim_mlp = hidden_dim_mlp
        self.block_num_gnn = block_num_gnn
        self.layer_num_mlp = layer_num_mlp

        self.encoder_node = MLP(self.node_input_dim, self.hidden_dim_gnn,
            self.hidden_dim_mlp, self.layer_num_mlp, activation=F.leaky_relu, norm=True)
        self.encoder_edge = MLP(self.edge_input_dim, self.hidden_dim_gnn,
            self.hidden_dim_mlp, self.layer_num_mlp, activation=F.leaky_relu, norm=True)

        self.processor_node = torch.nn.ModuleList()
        self.processor_edge = torch.nn.ModuleList()
        for i in range(self.block_num_gnn):
            self.processor_node.append(MLP(2*self.hidden_dim_gnn, self.hidden_dim_gnn,
                self.hidden_dim_mlp, self.layer_num_mlp, activation=F.leaky_relu, norm=True))
            self.processor_edge.append(MLP(3*self.hidden_dim_gnn, self.hidden_dim_gnn,
                self.hidden_dim_mlp, self.layer_num_mlp, activation=F.leaky_relu, norm=True))

        self.output = MLP(self.hidden_dim_gnn, self.output_dim,
            self.hidden_dim_mlp, self.layer_num_mlp, activation=F.leaky_relu, norm=False)

    def encode_edge(self, edge):
        """ Encode graph edge
        Args:
            edge: graph edge
        Returns:
            encoded features
        """
        proc_edge = self.encoder_edge(edge.data['efeatures'])
        return {'proc_edge': proc_edge}

    def process_edge(self, edge, index):
        """ Process graph edge
        Args:
            edge: graph edge
            index: iteration index
        Returns:
            processed features
        """
        f1 = edge.data['proc_edge']
        f2 = edge.src['proc_node']
        f3 = edge.dst['proc_node']
        proc_edge = self.processor_edge[index](torch.cat((f1, f2, f3), 1))
        proc_edge = proc_edge + f1
        return {'proc_edge': proc_edge}

    def process_node(self, node, index):
        """ Process graph node
        Args:
            node: graph node
            index: iteration index
        Returns:
            processed features
        """
        f1 = node.data['proc_node']
        f2 = node.data['pe_sum']
        proc_node = self.processor_node[index](torch.cat((f1, f2), 1))
        proc_node = proc_node + f1
        return {'proc_node': proc_node}

    def decode_node(self, node):
        """ Decode graph node
        Args:
            node: graph node
        Returns:
            decoded features
        """
        h = self.output(node.data['proc_node'])
        return {'pred_labels': h}

class DglMeshGraphNet(DglEncodeProcessDecode):
    """ MeshGraphNet """
    def __init__(self, node_input_dim, edge_input_dim, output_dim, hidden_dim_gnn, hidden_dim_mlp, 
                 layer_num_mlp, block_num_gnn):
        super(DglMeshGraphNet, self).__init__(node_input_dim, edge_input_dim, output_dim, 
            hidden_dim_gnn, hidden_dim_mlp, layer_num_mlp, block_num_gnn)

    def encode_node(self, node):
        """ Encode graph node
        Args:
            edge: graph node
        Returns:
            encoded features
        """
        inmask = node.data['inlet_mask'].bool()
        nnode = inmask.shape[0]
        nf = torch.zeros((nnode,1))
        nf[inmask] = torch.unsqueeze(node.data['next_flowrate'][inmask],1)

        features = torch.cat((node.data['nfeatures'], nf), 1)
        enc_features = self.encoder_node(features)
        return {'proc_node': enc_features}

    def continuity_loss(self, g, flowrate, take_mean = True):
        """ Compute contiuity loss
            Continuity loss as the mass loss occurring at junctions.
        Args:
            g: graph
            flowrate: tensor containing nodal values of flowrate
            take_mean: if True, take mean of junction losses. If False, take sum.
        Returns: 
            sum of mass loss occurring at branches and at junctions
        """
        g.ndata['next_flowrate'] = flowrate.clone()

        # we zero-out inlet and outlet flowrate (otherwise they would send
        # their flowrate to branch and junction node)
        g.ndata['next_flowrate'][g.ndata['inlet_mask'].bool()] = 0
        g.ndata['next_flowrate'][g.ndata['outlet_mask'].bool()] = 0

        # we keep flowrate at inlet and outlets of junctions
        g.ndata['flow_junction'] = g.ndata['next_flowrate'] * g.ndata['jun_mask']

        g.update_all(dgl.function.copy_u('flow_junction', 'm'), 
                     dgl.function.sum('m', 'sum_flowrate'))

        # we use the inlet to compute the difference
        diff = torch.abs(g.ndata['sum_flowrate'] - g.ndata['next_flowrate'])
        diff = diff * g.ndata['jun_inlet_mask']

        if take_mean:
            junction_continuity = torch.sum(diff) / torch.sum(g.ndata['jun_inlet_mask'])
        else:
            junction_continuity = torch.sum(diff)

        return junction_continuity

    def forward(self, g):
        """ Forward step
        Args:
            g: the graph
        Returns:
            the update for pressure and flowrate 
        """
        g.apply_nodes(self.encode_node)
        g.apply_edges(self.encode_edge)
        
        for index in range(self.block_num_gnn):
            def process_edge(edge):
                return self.process_edge(edge, index)
            def process_node(node):
                return self.process_node(node, index)
            # compute junction-branch interactions
            g.apply_edges(process_edge)
            g.update_all(dgl.function.copy_e('proc_edge', 'm'), 
                         dgl.function.sum('m', 'pe_sum'))
            g.apply_nodes(process_node)

        g.apply_nodes(self.decode_node)

        return g.ndata['pred_labels']
'''