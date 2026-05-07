import os
import torch
import numpy as np
import networkx as nx
import scipy.sparse as sp
import copy
from collections import OrderedDict
from torch.utils.data import Subset
from torch_geometric.data import Data as PyGData
from torch_geometric.data import HeteroData as PyGHeteroData
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_scatter import scatter
from rdkit import Chem
from tqdm import tqdm

from .molecule import Mol
# from .protein import Protein

def to_pyg_data(data, var_dict, dtype=torch.float32):
    node_pos = []
    for key in var_dict['node_pos']:
        node_pos.append(torch.tensor(data[key], dtype=dtype))
    node_pos = torch.cat(node_pos,-1)

    node_attr = []
    for key in var_dict['node_attr']:
        if data[key].shape!=():
            node_attr.append(data[key])
        else:
            node_attr.append(torch.tensor(data[key], dtype=dtype
                             )*torch.ones(node_pos.shape[0],1, dtype=dtype))
    node_attr = torch.cat(node_attr,-1)

    edge_index = torch.tensor(data['edge_index'], dtype=torch.long)
    edge_index = torch.cat([edge_index,edge_index.flip(dims=[0])], -1)
    edge_index[edge_index==-1] = node_pos.shape[0]-1
    
    node_label = {}
    for key in var_dict['node_label']:
        node_label[key] = torch.tensor(data[key], dtype=dtype)
    
    name = data['name'] if 'name' in data else None

    return PyGData(node_pos=node_pos, node_attr=node_attr, edge_index=edge_index,
                   node_label=node_label, name=name)

def face_to_edge(face_idx):
    """ Transform face index to edge index.
    Args:
        face_idx: face index
    Returns:
        edge index
    """
    # Collect edges from triangles
    face_idx = face_idx.T
    edge_idx = torch.cat([face_idx[:,0:2], face_idx[:,1:3],
                          torch.stack([face_idx[:,2], face_idx[:,0]], dim=1)], dim=0)

    # Sort and pack edges as single torch.int64
    recv_idx, _ = torch.min(edge_idx, dim=1)
    send_idx, _ = torch.max(edge_idx, dim=1)
    packed_edge = torch.stack([send_idx, recv_idx], dim=1).to(torch.int64)

    # Remove duplicates and unpack
    unique_edge_dict = OrderedDict.fromkeys(map(tuple, packed_edge.numpy()))
    unique_edge = torch.tensor(list(unique_edge_dict.keys()))

    send_idx, recv_idx = torch.unbind(unique_edge.T, dim=0)
    
    # Create two-way connectivity
    return torch.cat([torch.cat([send_idx, recv_idx], dim=0).unsqueeze(0),
                      torch.cat([recv_idx, send_idx], dim=0).unsqueeze(0)], dim=0)

def euler_lagrange_to_pyg_data(data, sys_type, mask_fn, mask_noise_fn=None,
                               noise_std: float=None, noise_gamma: float=None, 
                               add_noise: bool=False, label_update: bool=False,
                               dtype=torch.float32):
    """ Transform numpy data to pygdata
    Args:
        data: numpy data
        sys_type: type of dynamical system, euler or lagrange
        mask_fn: mask adding to computing node
        mask_noise_fn: mask adding to node requiring noise addition
        noise_std: standard deviation of the noise distribution
        noise_gamma: ratio of noise
        add_noise: whether to add noise
        label_update: whether the label is an update quantity
    """
    pyg_data = []
    time_step_num = data['mesh_pos'].shape[0]
    for i in range(1,time_step_num-1):
        # convert face index to edge index
        face_index = torch.as_tensor(data['cells'][i].T, dtype=torch.long)
        edge_index = face_to_edge(face_index)
        
        # mesh_pos, world_pos and volectiy
        mesh_pos = torch.as_tensor(data['mesh_pos'][i], dtype=dtype)
        if sys_type=='euler':
            world_pos, world_pos_prev = None, None
            velocity = torch.as_tensor(data['velocity'][i], dtype=dtype)
        if sys_type=='lagrange':
            world_pos = torch.as_tensor(data['world_pos'][i], dtype=dtype)
            world_pos_prev = torch.as_tensor(data['world_pos'][i-1], dtype=dtype)
            velocity = world_pos - world_pos_prev

        # node attribute and label
        node_type = torch.as_tensor(data['node_type'][i], dtype=dtype)
        x = velocity
        if sys_type=='euler':
            y = torch.as_tensor(data['velocity'][i+1], dtype=dtype)
        if sys_type=='lagrange':
            y = torch.as_tensor(data['world_pos'][i+1], dtype=dtype)
        
        # add noise
        if add_noise:
            noise = torch.normal(mean=0.0, std=noise_std, size=x.shape).to(x.device)
            mask_noise = mask_noise_fn(node_type[:,0])
            noise[~mask_noise] = 0
            # x += noise
            # y += (1.0-noise_gamma) * noise
        
        # calculate the update quantity (if need)
        if label_update:
            if sys_type=='euler':
                y = y - x
            if sys_type=='lagrange':
                y = y - 2*world_pos + world_pos_prev
        x = torch.cat([node_type,x],1)

        # transform to pygdata
        mask = mask_fn(node_type[:,0])
        pyg_data.append(PyGData(edge_index=edge_index, face_index=face_index,
                                mesh_pos=mesh_pos, world_pos=world_pos, 
                                world_pos_prev=world_pos_prev, x=x, y=y, mask=mask))
    return pyg_data

def transform_qm9(data):
    x, adj = data
    # the last place is for virtual nodes
    # 6: C, 7: N, 8: O, 9: F
    x_ = np.zeros((9, 5))
    indices = np.where(x >= 6, x - 6, 4)
    x_[np.arange(9), indices] = 1
    x = torch.tensor(x_).to(torch.float32)
    # single, double, triple and no-bond; the last channel is for virtual edges
    adj = np.concatenate([adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)],
                         axis=0).astype(np.float32)

    x = x[:, :-1]                               # 9, 5 (the last place is for vitual nodes) -> 9, 4 (38, 9)
    adj = torch.tensor(adj.argmax(axis=0))      # 4, 9, 9 (the last place is for vitual edges) -> 9, 9 (38, 38)
    # 0, 1, 2, 3 -> 1, 2, 3, 0; now virtual edges are denoted as 0
    adj = torch.where(adj == 3, 0, adj + 1).to(torch.float32)

    adj = adj.triu()
    edge = from_scipy_sparse_matrix(sp.coo_matrix(adj))
    edge_index = edge[0]
    edge_weight = edge[1]
    return PyGData(x=x, edge_index=edge_index, edge_weight=edge_weight)

def edge_idx_to_adj(node_num, edge_index, edge_weight):
    adj = np.zeros([node_num,node_num])
    if edge_index.shape[1]!=0:
        row = np.array(edge_index[0,:])
        col = np.array(edge_index[1,:])
        adj = sp.coo_matrix((edge_weight, (row,col)), shape=(node_num,node_num)).tocsr().toarray()
    return adj

def mol_to_smiles(mols):
    return [Chem.MolToSmiles(mol) for mol in mols]

def smiles_to_mol(smiles):
    return [Chem.MolFromSmiles(s) for s in smiles]

def canonicalize_smiles(smiles):
    return [Chem.MolToSmiles(Chem.MolFromSmiles(smile)) for smile in smiles]

def mol_to_nx(mols):
    nx_graphs = []
    for mol in mols:
        g = nx.Graph()
        for atom in mol.GetAtoms():
            g.add_node(atom.GetIdx(), label=atom.GetSymbol())
            #          atomic_num=atom.GetAtomicNum(),
            #          formal_charge=atom.GetFormalCharge(),
            #          chiral_tag=atom.GetChiralTag(),
            #          hybridization=atom.GetHybridization(),
            #          num_explicit_hs=atom.GetNumExplicitHs(),
            #          is_aromatic=atom.GetIsAromatic())
        for bond in mol.GetBonds():
            g.add_edge(bond.GetBeginAtomIdx(),
                       bond.GetEndAtomIdx(),
                       label=int(bond.GetBondTypeAsDouble()))
            #          bond_type=bond.GetBondType())
        nx_graphs.append(g)
    return nx_graphs
'''
class EdgeToAdj():
    def __init__(self, node_num_max: int=125, mask: bool=True, channelization: bool=True,
                 centralization: bool=True, dequantization: bool=False,
                 remove_diagonal: bool=False) -> None:
        self.node_num_max = node_num_max
        self.mask = mask
        self.channelization = channelization
        self.centralization = centralization
        self.dequantization = dequantization
        self.remove_diagonal = remove_diagonal

    def __call__(self, batch_data):
        """ Convert batched PyG data to adjacency matrices.
        Args:
            batch_data: batched PyG data.
            node_num_max: maximum number of node in each graph.
        Returns:
            adj: adjacency matrices.
            adj_mask: mask for adjacency matrices.
        """
        batch, x, edge_index = batch_data.batch, batch_data.x, batch_data.edge_index
        if hasattr(batch_data, 'edge_weight'):
            edge_weight = batch_data.edge_weight
        else:
            edge_weight = None
        
        x, adj, x_mask, adj_mask = self.to_adj_matrices(
            batch, x, edge_index, edge_weight, self.node_num_max)
        
        if self.channelization:
            if x is not None:
                x = x.unsqueeze(1)
                x_mask = x_mask.unsqueeze(1)
            adj = adj.unsqueeze(1)
            adj_mask = adj_mask.unsqueeze(1)
        
        if self.dequantization:
            noise = torch.rand_like(adj)
            noise = torch.tril(noise, -1)
            noise = noise + noise.transpose(-1,-2)
            adj = (noise + adj) / 2.
        
        if self.centralization:
            adj = 2*adj-1.
        
        if self.remove_diagonal:
            adj_mask = torch.tril(adj_mask, -1)
            adj_mask = adj_mask + adj_mask.transpose(-1,-2)

        return x, adj, x_mask, adj_mask
    
    @torch.no_grad()
    def to_adj_matrices(self, batch=None, node=None, edge_index=None, edge_weight=None,
                        node_num_max=None):
        """ Converts batched PyG data given by edge indices and edge attributes to 
            batched adjacency matrices.
        Args:
            edge_index: the edge indices.
            batch: batch vector assigning each node to a specific graph
            edge_weight: edge weights or edge features.
            node_num_max: maximum number of node in each graph.
        Returns:
            adj: adjacency matrices, [batch_size, node_num_max, node_num_max].
            adj_mask: mask for adjacency matrices.
        """
        # node_num: number of node in each graph
        # node_cum: cumulative number of node
        if batch is None:
            batch = edge_index.new_zeros(edge_index.max().item() + 1)
        batch_size = batch.max().item() + 1
        ones = batch.new_ones(batch.size(0))
        node_num = scatter(ones, batch, dim=0, dim_size=batch_size, reduce='add')
        node_cum = torch.cat([batch.new_zeros(1), node_num.cumsum(dim=0)])
        
        # idx0: index of the graph to which the edge belongs
        # idx1, idx2: index of edge in each graph
        idx0 = batch[edge_index[0]]
        idx1 = edge_index[0] - node_cum[batch][edge_index[0]]
        idx2 = edge_index[1] - node_cum[batch][edge_index[1]]

        # truncate node and edge
        if node_num_max is None:
            node_num_max = node_num.max().item()
        elif idx1.max() >= node_num_max or idx2.max() >= node_num_max:
            mask = (idx1 < node_num_max) & (idx2 < node_num_max)
            idx0 = idx0[mask]
            idx1 = idx1[mask]
            idx2 = idx2[mask]
            edge_weight = None if edge_weight is None else edge_weight[mask]

        # generate adjacency matrices
        # add edge weight to adj through scatter
        # adj: size -> flattened_size -> size
        if edge_weight is None:
            edge_weight = torch.ones(idx0.numel(), device=edge_index.device)
        
        adj_size = [batch_size, node_num_max, node_num_max]
        adj_size += list(edge_weight.size())[1:]
        adj_size_flat = batch_size * node_num_max * node_num_max
        
        adj = torch.zeros(adj_size, dtype=edge_weight.dtype, device=edge_index.device)
        adj = adj.view([adj_size_flat] + list(adj.size())[3:])
        adj_idx = idx0 * node_num_max * node_num_max + idx1 * node_num_max + idx2
        scatter(edge_weight, adj_idx, dim=0, out=adj, reduce='add')
        adj = adj.view(adj_size)
        adj = adj + adj.transpose(-1,-2)
        
        # generate node
        # x_idx: index of node in extended graph
        x_idx = torch.arange(batch.size(0), dtype=torch.long, device=edge_index.device)
        x_idx = (x_idx - node_cum[batch]) + (batch * node_num_max)
        
        if node is not None:
            x_size = [batch_size, node_num_max]
            x_size += list(node.size())[1:]
            x_size_flat = batch_size * node_num_max
        
            x = torch.zeros(x_size, dtype=node.dtype, device=node.device)
            x = x.view([x_size_flat] + list(x.size())[2:])
            scatter(node, x_idx, dim=0, out=x, reduce='add')
            x = x.reshape(x_size)
        else:
            x = None
        
        # generate mask
        x_mask = torch.zeros(batch_size*node_num_max, dtype=adj.dtype, device=adj.device)
        x_mask[x_idx] = 1
        x_mask = x_mask.view(batch_size, node_num_max)
        adj_mask = x_mask[:,None,:] * x_mask[:,:,None]
        if node is None or not self.mask:
            x_mask = None
        if not self.mask:
            adj_mask = None

        return x, adj, x_mask, adj_mask

def process_complex(root, name, rec, rec_embd, ligs,
                    lig_size_max, popsize, maxiter, matching, keep_original,
                    conformers_num, remove_hs, rec_cutoff_radius, ca_nb_num_max,
                    all_atom, atom_cutoff_radius, atom_nb_num_max):
    if not os.path.exists(os.path.join(root, name)):
        print("Folder not found", name)
        return [], []

    if rec is None:
        print(f'Skipping {name} because rec is None')
        return [], []
        
    if ligs is None:
        print(f'Skipping {name} because ligs is None')
        return [], []
    
    complex_graphs = []
    failed_indices = []
    for i, lig in enumerate(ligs):
        # Skip ligand with large size
        if lig_size_max is not None and lig.GetNumHeavyAtoms() > lig_size_max:
            print(f'Ligand with {lig.GetNumHeavyAtoms()} heavy atoms is larger than '+
                  f'lig_size_max {lig_size_max}. Not including {name} in preprocessed data.')
            continue

        complex_graph = PyGHeteroData()
        complex_graph['name'] = name
        try:
            lig_model = Mol(lig)
            (rmsd_matching, node_pos_ori, node_fea, node_pos, edge_idx, edge_attr, edge_mask,
             mask_rotate) = lig_model.get_lig_graph(
                lig, matching, remove_hs, keep_original, conformers_num, popsize, maxiter)
            complex_graph.rmsd_matching = rmsd_matching
            complex_graph['ligand'].orig_pos = node_pos_ori
            complex_graph['ligand'].x = node_fea
            complex_graph['ligand'].pos = node_pos
            complex_graph['ligand','lig_bond','ligand'].edge_index = edge_idx
            complex_graph['ligand','lig_bond','ligand'].edge_attr = edge_attr
            complex_graph['ligand'].edge_mask = edge_mask
            complex_graph['ligand'].mask_rotate = mask_rotate
            
            rec_model = Protein(rec)
            (fail, node_fea, node_pos, node_mu_r_norm, node_side_chain_vec, edge_idx,
             atom_x, atom_pos, atom_atom_edge_idx, rec_atom_edge_idx) = rec_model.get_rec_graph(
                lig, rec_embd, rec_cutoff_radius=rec_cutoff_radius,
                ca_nb_num_max=ca_nb_num_max, all_atom=all_atom,
                atom_cutoff_radius=atom_cutoff_radius, atom_nb_num_max=atom_nb_num_max,
                remove_hs=remove_hs)
            if fail:
                print(f'LM embeddings for complex {name} did not have the right length for the protein. Skipping {name}.')
                failed_indices.append(i)
                continue
            
            complex_graph['receptor'].x = node_fea
            complex_graph['receptor'].pos = node_pos
            complex_graph['receptor'].mu_r_norm = node_mu_r_norm
            complex_graph['receptor'].side_chain_vecs = node_side_chain_vec
            complex_graph['receptor', 'rec_contact', 'receptor'].edge_index = edge_idx

            complex_graph['atom'].x = atom_x
            complex_graph['atom'].pos = atom_pos
            complex_graph['atom', 'atom_contact', 'atom'].edge_index = atom_atom_edge_idx
            complex_graph['atom', 'atom_rec_contact', 'receptor'].edge_index = rec_atom_edge_idx
        
        except Exception as e:
            print(f'Skipping {name} because of the error:')
            print(e)
            failed_indices.append(i)
            continue

        protein_center = torch.mean(complex_graph['receptor'].pos, dim=0, keepdim=True)
        complex_graph['receptor'].pos -= protein_center
        if all_atom:
            complex_graph['atom'].pos -= protein_center

        if (not matching) or conformers_num == 1:
            complex_graph['ligand'].pos -= protein_center
        else:
            for p in complex_graph['ligand'].pos:
                p -= protein_center

        complex_graph.original_center = protein_center
        complex_graphs.append(complex_graph)
        
    for idx_to_delete in sorted(failed_indices, reverse=True):
        del ligs[idx_to_delete]
    
    return complex_graphs, ligs
'''
def split_dataset(dataset, split_ratio: float=1.0, train_idx=None, test_idx=None):
    
    # data iterators
    if test_idx is not None:
        train_dataset = Subset(dataset, train_idx)
        test_dataset = Subset(dataset, test_idx)
        return train_dataset, test_dataset
    else:
        num_train = int(len(dataset) * split_ratio)
        train_dataset = dataset[:num_train]
        test_dataset = dataset[num_train:]

        num_test = len(dataset) - num_train
        eval_dataset = dataset[:num_test]
    
        return train_dataset, eval_dataset, test_dataset

def create_index_map(times, stride):
    """ Create index map.
        Index map is a n x 2 array (n is the total number of timesteps in the dataset) 
        mapping a graph index (first column) to the timestep index (second column).
    """
    i, st = 0, 0
    graph_num = len(times)
    total_times = np.sum(times)
    index_map = np.zeros((int(total_times - graph_num*stride), 2))
    for t in times:
        at = t - stride # actual time (minus stride)
        graph_index = i*np.ones((at,1))
        time_index = np.expand_dims(np.arange(0,at), axis=1)
        index_map[st:st+at,:] = (
            np.concatenate([graph_index,time_index], axis=1))
        i += 1
        st += at
    index_map = np.array(index_map, dtype=int)
    return index_map

def graph_to_lightgraph(data_list, stride):
    """ Process Dataset.
        Creates lightgraphs, the index map, and collects all times from the graphs.
    """
    times = []
    lightgraphs = []
    for graph in tqdm(data_list, desc='Processing dataset', colour='green'):

        lightgraph = copy.deepcopy(graph)

        node_data = [ndata for ndata in lightgraph.ndata]
        edge_data = [edata for edata in lightgraph.edata]
        for ndata in node_data:
            if 'mask' not in ndata:
                del lightgraph.ndata[ndata]
        for edata in edge_data:
            del lightgraph.edata[edata]

        times.append(graph.ndata['nfeatures'].shape[2])
        lightgraphs.append(lightgraph)

    times = np.array(times)
    index_map = create_index_map(times, stride)

    return lightgraphs, times, index_map