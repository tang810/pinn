import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius, radius_graph
from torch_scatter import scatter, scatter_mean
from e3nn import o3
from e3nn.nn import BatchNorm
from rdkit.Chem.rdchem import BondType as BT

import gnn4s
from .mlp import MLP

class AtomEncoder(nn.Module):
    def __init__(self, embd_dim: int, feat_dim: tuple, sigma_embd_dim: int, lm_embd_type: str=None) -> None:
        super(AtomEncoder, self).__init__()
        # first element of feat_dim tuple is a list with the length of each categorical feature and 
        # the second is the number of scalar features
        self.embd_dim = embd_dim
        self.feat_dim = feat_dim
        self.sigma_embd_dim = sigma_embd_dim
        self.lm_embd_type = lm_embd_type

        # number of categorical features
        self.catg_feat_num = len(self.feat_dim[0])
        self.scalar_feat_num = self.feat_dim[1] + self.sigma_embd_dim
        
        # feature embedding layer
        self.atom_embd_list = torch.nn.ModuleList()
        for i, dim in enumerate(self.feat_dim[0]):
            embd = torch.nn.Embedding(dim, self.embd_dim)
            torch.nn.init.xavier_uniform_(embd.weight.data)
            self.atom_embd_list.append(embd)

        # scalar embedding layer
        if self.scalar_feat_num > 0:
            self.scalar_embd_layer = torch.nn.Linear(self.scalar_feat_num, self.embd_dim)
        
        # language model embedding layer
        if self.lm_embd_type is not None:
            if self.lm_embd_type == 'esm':
                self.lm_embd_dim = 1280
            else: 
                raise ValueError('LM Embedding type was not correctly determined. LM embedding type: ',
                                 self.lm_embd_type)
            self.lm_embd_layer = torch.nn.Linear(self.lm_embd_dim+self.embd_dim, self.embd_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.lm_embd_type is not None:
            assert x.shape[1] == self.catg_feat_num + self.scalar_feat_num + self.lm_embd_dim
        else:
            assert x.shape[1] == self.catg_feat_num + self.scalar_feat_num
        
        x_embd = 0
        # feature embeddings
        for i in range(self.catg_feat_num):
            x_embd += self.atom_embd_list[i](x[:,i].long())
        
        # scalar embeddings
        if self.scalar_feat_num > 0:
            x_embd += self.scalar_embd_layer(x[:, self.catg_feat_num: self.catg_feat_num+self.scalar_feat_num])
        
        # language model embeddings
        if self.lm_embd_type is not None:
            x_embd = self.lm_embd_layer(torch.cat([x_embd, x[:,-self.lm_embd_dim:]], axis=1))
        return x_embd

class GaussianSmearing(nn.Module):
    """ Embedding the edge distances
    """
    def __init__(self, start: float=0.0, stop: float=5.0, gaussian_num: int=50) -> None:
        super().__init__()
        offset = torch.linspace(start, stop, gaussian_num)
        self.coef = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coef * torch.pow(dist, 2))

class TensorProductConvLayer(nn.Module):
    def __init__(self, in_irreps, sh_irreps, ou_irreps, edge_feat_num: int,
                 use_residual: bool=True, use_batch_norm: bool=True, dropout_rate: float=0.0,
                 hidden_feat_num: int=None) -> None:
        super(TensorProductConvLayer, self).__init__()
        # irreducible representations
        self.in_irreps = in_irreps
        self.ou_irreps = ou_irreps
        self.sh_irreps = sh_irreps
        self.edge_feat_num = edge_feat_num
        self.use_residual = use_residual
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.hidden_feat_num = hidden_feat_num

        if self.hidden_feat_num is None:
            self.hidden_feat_num = self.edge_feat_num

        # tensor product layer
        self.tp = o3.FullyConnectedTensorProduct(self.in_irreps, self.sh_irreps, self.ou_irreps, shared_weights=False)

        # fully connected layer for weights
        self.fc = MLP(input_dim=self.edge_feat_num, output_dim=self.tp.weight_numel, 
                      hidden_dim=self.hidden_feat_num, layer_num=1, 
                      activation=nn.ReLU(), dropout_rate=self.dropout_rate)
        self.batch_norm = BatchNorm(self.ou_irreps) if self.use_batch_norm else None

    def forward(self, node_attr: torch.Tensor, edge_idx: torch.Tensor, edge_attr: torch.Tensor, edge_sh: torch.Tensor,
                node_out_dim: int=None, reduce: str='mean') -> torch.Tensor:
        # calulate tensor product
        edge_src, edge_dst = edge_idx
        tp = self.tp(node_attr[edge_dst], edge_sh, self.fc(edge_attr))
        
        # scatter
        node_out_dim = node_out_dim or node_attr.shape[0]
        node_out = scatter(tp, edge_src, dim=0, dim_size=node_out_dim, reduce=reduce)
        
        # residual connection
        if self.use_residual:
            node_out += F.pad(node_attr, (0, node_out.shape[-1]-node_attr.shape[-1]))
        
        # batch normalization
        if self.use_batch_norm:
            node_out = self.batch_norm(node_out)
        return node_out

class TensorProductModel(nn.Module):
    def __init__(self, t_to_sigma, timestep_embd_func, ns: int=16, nv: int=4,
                 lig_edge_in_feat: int=4, sigma_embd_dim: int=32, dist_embd_dim: int=32,
                 cro_dist_embd_dim: int=32, lm_embd_type=None, 
                 lig_radius_max: float=5.0, rec_radius_max: float=30.0, cro_dist_max: float=250.0,
                 cen_dist_max: float=30, dropout_rate=0.0, dynamic_cro_max: bool=False, 
                 use_second_order_repr=False, sh_lmax: int=2, use_batch_norm: bool=True, conv_layer_num: int=2,
                 no_torsion: bool=False, scale_by_sigma: bool=True, 
                 conf_mode: bool=False, conf_dropout_rate: float=0.0, conf_no_batch_norm: bool=False,
                 conf_output_num: int=1, device: str='cpu') -> None:
        super(TensorProductModel, self).__init__()
        """ Tensor product model.
        Args:
            t_to_sigma:
            timestep_embd_func: time embedding function
            ns: number of hidden features per node of order 0
            nv: number of hidden features per node of order >0
            lig_edge_in_feat: 
            sigma_embd_dim: size of the embedding of the diffusion time
            dist_embd_dim: embedding size for the distance 
            cro_dist_embd_dim: embeddings size for the cross distance
            lm_embd_type: type of language model embedding
            lig_radius_max: cutoff radius for ligand graph 
            rec_radius_max: cutoff radius for recptor graph 
            cro_dist_max: maximum cross distance in case not dynamic
            cen_dist_max: maximum center distance in case not dynamic
            dropout_rate: mlp dropout rate
            dynamic_cro_max: whether to use the dynamic distance cutoff
            use_second_order_repr: whether to use only up to first order representations or also second
            sh_lmax: maximum spherical harmonic degree
            use_batch_norm: whether to use the batch norm
            conv_layer_num: number of interaction layers
            no_torsion: whether to conside torsion
            scale_by_sigma: whether to normalize the score 
            conf_mode: whether the model is a confidence model
            conf_dropout_rate: mlp dropout rate
            conf_no_batch_norm: whether to remove the batch norm
            conf_output_num: number of output
            device: computing device
        """
        self.t_to_sigma = t_to_sigma
        self.timestep_embd_func = timestep_embd_func
        self.ns, self.nv = ns, nv
        self.lig_edge_in_feat = lig_edge_in_feat
        self.sigma_embd_dim = sigma_embd_dim
        self.dist_embd_dim = dist_embd_dim
        self.cro_dist_embd_dim = cro_dist_embd_dim
        self.lm_embd_type = lm_embd_type
        self.lig_radius_max = lig_radius_max
        self.rec_radius_max = rec_radius_max
        self.cro_dist_max = cro_dist_max
        self.cen_dist_max = cen_dist_max
        self.dropout_rate = dropout_rate
        self.dynamic_cro_max = dynamic_cro_max
        self.use_second_order_repr = use_second_order_repr
        self.sh_lmax = sh_lmax
        self.use_batch_norm = use_batch_norm
        self.conv_layer_num = conv_layer_num
        self.no_torsion = no_torsion
        self.scale_by_sigma = scale_by_sigma
        self.conf_mode = conf_mode
        self.conf_dropout_rate = conf_dropout_rate
        self.conf_no_batch_norm = conf_no_batch_norm
        self.conf_output_num = conf_output_num
        self.device = device

        self.allowable_features = {
            'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
            'possible_chirality_list': ['CHI_UNSPECIFIED','CHI_TETRAHEDRAL_CW','CHI_TETRAHEDRAL_CCW','CHI_OTHER'],
            'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
            'possible_numring_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
            'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
            'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
            'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
            'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
            'possible_hybridization_list': ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'],
            'possible_is_aromatic_list': [False, True],
            'possible_is_in_ring3_list': [False, True],
            'possible_is_in_ring4_list': [False, True],
            'possible_is_in_ring5_list': [False, True],
            'possible_is_in_ring6_list': [False, True],
            'possible_is_in_ring7_list': [False, True],
            'possible_is_in_ring8_list': [False, True],
            'possible_amino_acids': ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                                    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
                                    'HIP', 'HIE', 'TPO', 'HID', 'LEV', 'MEU', 'PTR', 'GLV', 'CYT', 'SEP',
                                    'HIZ', 'CYM', 'GLM', 'ASQ', 'TYS', 'CYX', 'GLZ', 'misc'],
            'possible_atom_type_2': ['C*', 'CA', 'CB', 'CD', 'CE', 'CG', 'CH', 'CZ', 'N*', 'ND',
                                    'NE', 'NH', 'NZ', 'O*', 'OD', 'OE', 'OG', 'OH', 'OX', 'S*',
                                    'SD', 'SG', 'misc'],
            'possible_atom_type_3': ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3',
                                    'CG', 'CG1', 'CG2', 'CH2', 'CZ', 'CZ2', 'CZ3', 'N', 'ND1', 'ND2',
                                    'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'O', 'OD1', 'OD2', 'OE1',
                                    'OE2', 'OG', 'OG1', 'OH', 'OXT', 'SD', 'SG', 'misc'],
        }
        self.bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        self.lig_feat_dim = (list(map(len, [
            self.allowable_features['possible_atomic_num_list'],
            self.allowable_features['possible_chirality_list'],
            self.allowable_features['possible_degree_list'],
            self.allowable_features['possible_formal_charge_list'],
            self.allowable_features['possible_implicit_valence_list'],
            self.allowable_features['possible_numH_list'],
            self.allowable_features['possible_number_radical_e_list'],
            self.allowable_features['possible_hybridization_list'],
            self.allowable_features['possible_is_aromatic_list'],
            self.allowable_features['possible_numring_list'],
            self.allowable_features['possible_is_in_ring3_list'],
            self.allowable_features['possible_is_in_ring4_list'],
            self.allowable_features['possible_is_in_ring5_list'],
            self.allowable_features['possible_is_in_ring6_list'],
            self.allowable_features['possible_is_in_ring7_list'],
            self.allowable_features['possible_is_in_ring8_list'],
        ])), 0)  # number of scalar features

        self.rec_res_feat_dim = (list(map(len, [
            self.allowable_features['possible_amino_acids']
        ])), 0)

        # embedding layer
        self.lig_node_embd = AtomEncoder(embd_dim=self.ns, feat_dim=self.lig_feat_dim,
                                         sigma_embd_dim=self.sigma_embd_dim)
        self.lig_edge_embd = MLP(input_dim=self.lig_edge_in_feat+self.sigma_embd_dim+self.dist_embd_dim,
                                 output_dim=self.ns, hidden_dim=self.ns, layer_num=1, 
                                 activation=nn.ReLU(), dropout_rate=self.dropout_rate)
        
        self.rec_node_embd = AtomEncoder(embd_dim=self.ns, feat_dim=self.rec_res_feat_dim, 
                                         sigma_embd_dim=self.sigma_embd_dim, lm_embd_type=self.lm_embd_type)
        self.rec_edge_embd = MLP(input_dim=self.sigma_embd_dim+self.dist_embd_dim,
                                 output_dim=self.ns, hidden_dim=self.ns, layer_num=1, 
                                 activation=nn.ReLU(), dropout_rate=self.dropout_rate)

        self.cro_edge_embd = MLP(input_dim=self.sigma_embd_dim+self.cro_dist_embd_dim,
                                 output_dim=self.ns, hidden_dim=self.ns, layer_num=1, 
                                 activation=nn.ReLU(), dropout_rate=self.dropout_rate)
        
        self.lig_dist_exps = GaussianSmearing(0.0, self.lig_radius_max, self.dist_embd_dim)
        self.rec_dist_exps = GaussianSmearing(0.0, self.rec_radius_max, self.dist_embd_dim)
        self.cro_dist_exps = GaussianSmearing(0.0, self.cro_dist_max, self.cro_dist_embd_dim)

        # convolutional layer
        # irreducible representations
        if self.use_second_order_repr:
            irrep_seq = [
                f'{self.ns}x0e',
                f'{self.ns}x0e + {self.nv}x1o + {self.nv}x2e',
                f'{self.ns}x0e + {self.nv}x1o + {self.nv}x2e + {self.nv}x1e + {self.nv}x2o',
                f'{self.ns}x0e + {self.nv}x1o + {self.nv}x2e + {self.nv}x1e + {self.nv}x2o + {self.ns}x0o'
            ]
        else:
            irrep_seq = [
                f'{self.ns}x0e',
                f'{self.ns}x0e + {self.nv}x1o',
                f'{self.ns}x0e + {self.nv}x1o + {self.nv}x1e',
                f'{self.ns}x0e + {self.nv}x1o + {self.nv}x1e + {self.ns}x0o'
            ]

        lig_conv_layer, rec_conv_layer, lig_to_rec_conv_layer, rec_to_lig_conv_layer = [], [], [], []
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=self.sh_lmax)
        for i in range(self.conv_layer_num):
            in_irreps = irrep_seq[min(i, len(irrep_seq)-1)]
            ou_irreps = irrep_seq[min(i+1, len(irrep_seq)-1)]
            parameters = {
                'in_irreps': in_irreps,
                'sh_irreps': self.sh_irreps,
                'ou_irreps': ou_irreps,
                'edge_feat_num': 3*self.ns,
                'hidden_feat_num': 3*self.ns,
                'use_residual': False,
                'use_batch_norm': self.use_batch_norm,
                'dropout_rate': self.dropout_rate
            }
            lig_conv_layer.append(TensorProductConvLayer(**parameters))
            rec_conv_layer.append(TensorProductConvLayer(**parameters))
            lig_to_rec_conv_layer.append(TensorProductConvLayer(**parameters))
            rec_to_lig_conv_layer.append(TensorProductConvLayer(**parameters))

        self.lig_conv_layer = nn.ModuleList(lig_conv_layer)
        self.rec_conv_layer = nn.ModuleList(rec_conv_layer)
        self.lig_to_rec_conv_layer = nn.ModuleList(lig_to_rec_conv_layer)
        self.rec_to_lig_conv_layer = nn.ModuleList(rec_to_lig_conv_layer)
        
        if self.conf_mode:
            # confidence layer
            self.conf_predictor = nn.Sequential(
                nn.Linear(2*self.ns if self.conv_layer_num >= 3 else self.ns, self.ns),
                nn.BatchNorm1d(self.ns) if not self.conf_no_batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(self.conf_dropout_rate),
                nn.Linear(self.ns, self.ns),
                nn.BatchNorm1d(self.ns) if not self.conf_no_batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(self.conf_dropout_rate),
                nn.Linear(self.ns, self.conf_output_num)
            )
        else:
            # center of mass translation and rotation components
            self.cen_dist_exps = GaussianSmearing(0.0, self.cen_dist_max, self.dist_embd_dim)
            self.cen_edge_embd = MLP(input_dim=self.dist_embd_dim+self.sigma_embd_dim,
                                     output_dim=self.ns, hidden_dim=self.ns, layer_num=1, 
                                     activation=nn.ReLU(), dropout_rate=self.dropout_rate)
            # final layer
            self.final_conv = TensorProductConvLayer(
                in_irreps=self.lig_conv_layer[-1].ou_irreps,
                sh_irreps=self.sh_irreps,
                ou_irreps=f'2x1o + 2x1e',
                edge_feat_num=2*self.ns,
                use_residual=False,
                dropout_rate=self.dropout_rate,
                use_batch_norm=self.use_batch_norm
            )
            self.tra_final_layer = MLP(input_dim=1+self.sigma_embd_dim,
                                       output_dim=1, hidden_dim=self.ns, layer_num=1, 
                                       activation=nn.ReLU(), dropout_rate=self.dropout_rate)
            self.rot_final_layer = MLP(input_dim=1+self.sigma_embd_dim,
                                       output_dim=1, hidden_dim=self.ns, layer_num=1, 
                                       activation=nn.ReLU(), dropout_rate=self.dropout_rate)
            
            if not self.no_torsion:
                # torsion angles components
                self.final_edge_embd = MLP(input_dim=self.dist_embd_dim,
                                           output_dim=self.ns, hidden_dim=self.ns, layer_num=1, 
                                           activation=nn.ReLU(), dropout_rate=self.dropout_rate)
                self.final_tp_tor = o3.FullTensorProduct(self.sh_irreps, "2e")
                self.tor_bond_conv = TensorProductConvLayer(
                    in_irreps=self.lig_conv_layer[-1].ou_irreps,
                    sh_irreps=self.final_tp_tor.irreps_out,
                    ou_irreps=f'{self.ns}x0o + {self.ns}x0e',
                    edge_feat_num=3*self.ns,
                    use_residual=False,
                    dropout_rate=self.dropout_rate,
                    use_batch_norm=self.use_batch_norm
                )
                self.tor_final_layer = MLP(input_dim=2*self.ns, output_dim=1, hidden_dim=self.ns, 
                                           layer_num=1, bias=False, activation=nn.Tanh(), 
                                           dropout_rate=self.dropout_rate)

    def forward(self, data):
        """ Forward propagation.
        Args:
            data: heterogeneous graph containing receptor and ligand information.
        """
        if not self.conf_mode:
            tr_sigma, rot_sigma, tor_sigma = self.t_to_sigma(
                *[data.complex_t[noise_type] for noise_type in ['tr', 'rot', 'tor']])
        else:
            tr_sigma, rot_sigma, tor_sigma = [
                data.complex_t[noise_type] for noise_type in ['tr', 'rot', 'tor']]
        
        # build ligand graph
        lig_node_attr, lig_edge_idx, lig_edge_attr, lig_edge_sh = self.build_lig_conv_graph(data)
        lig_src, lig_dst = lig_edge_idx
        lig_node_attr = self.lig_node_embd(lig_node_attr)
        lig_edge_attr = self.lig_edge_embd(lig_edge_attr)

        # build receptor graph
        rec_node_attr, rec_edge_idx, rec_edge_attr, rec_edge_sh = self.build_rec_conv_graph(data)
        rec_src, rec_dst = rec_edge_idx
        rec_node_attr = self.rec_node_embd(rec_node_attr)
        rec_edge_attr = self.rec_edge_embd(rec_edge_attr)

        # build cross graph
        if self.dynamic_cro_max:
            cross_cutoff = (tr_sigma * 3 + 20).unsqueeze(1)
        else:
            cross_cutoff = self.cro_dist_max
        cro_edge_idx, cro_edge_attr, cro_edge_sh = self.build_cross_conv_graph(data, cross_cutoff)
        cross_lig, cross_rec = cro_edge_idx
        cro_edge_attr = self.cro_edge_embd(cro_edge_attr)

        for i in range(len(self.lig_conv_layer)):
            # intra graph message passing
            lig_edge_attr_ = torch.cat([lig_edge_attr, lig_node_attr[lig_src,:self.ns],
                                        lig_node_attr[lig_dst,:self.ns]], -1)
            
            lig_intra_update = self.lig_conv_layer[i](
                lig_node_attr, lig_edge_idx, lig_edge_attr_, lig_edge_sh)
            
            # inter graph message passing
            rec_to_lig_edge_attr_ = torch.cat([cro_edge_attr, lig_node_attr[cross_lig,:self.ns],
                                               rec_node_attr[cross_rec,:self.ns]], -1)
            lig_inter_update = self.rec_to_lig_conv_layer[i](
                rec_node_attr, cro_edge_idx, rec_to_lig_edge_attr_, cro_edge_sh,
                node_out_dim=lig_node_attr.shape[0])

            if i != len(self.lig_conv_layer) - 1:
                rec_edge_attr_ = torch.cat([rec_edge_attr, rec_node_attr[rec_src,:self.ns],
                                            rec_node_attr[rec_dst,:self.ns]], -1)
                rec_intra_update = self.rec_conv_layer[i](
                    rec_node_attr, rec_edge_idx, rec_edge_attr_, rec_edge_sh)

                lig_to_rec_edge_attr_ = torch.cat([cro_edge_attr, lig_node_attr[cross_lig,:self.ns],
                                                   rec_node_attr[cross_rec,:self.ns]], -1)
                rec_inter_update = self.lig_to_rec_conv_layer[i](
                    lig_node_attr, torch.flip(cro_edge_idx, dims=[0]), lig_to_rec_edge_attr_,
                    cro_edge_sh, node_out_dim=rec_node_attr.shape[0])

            # padding original features
            lig_node_attr = F.pad(lig_node_attr, (0, lig_intra_update.shape[-1] - lig_node_attr.shape[-1]))

            # update features with residual updates
            lig_node_attr = lig_node_attr + lig_intra_update + lig_inter_update

            if i != len(self.lig_conv_layer) - 1:
                rec_node_attr = F.pad(rec_node_attr, (0, rec_intra_update.shape[-1] - rec_node_attr.shape[-1]))
                rec_node_attr = rec_node_attr + rec_intra_update + rec_inter_update

        # compute confidence score
        if self.conf_mode:
            scalar_lig_attr = torch.cat([lig_node_attr[:,:self.ns],lig_node_attr[:,-self.ns:] ], 
                                        dim=1) if self.conv_layer_num >= 3 else lig_node_attr[:,:self.ns]
            confidence = self.conf_predictor(
                scatter_mean(scalar_lig_attr, data['ligand'].batch, dim=0)).squeeze(dim=-1)
            return confidence

        # compute translational and rotational score vectors
        center_edge_idx, center_edge_attr, center_edge_sh = self.build_center_conv_graph(data)
        center_edge_attr = self.cen_edge_embd(center_edge_attr)
        center_edge_attr = torch.cat([center_edge_attr, lig_node_attr[center_edge_idx[1], :self.ns]], -1)
        global_pred = self.final_conv(lig_node_attr, center_edge_idx, center_edge_attr, center_edge_sh, 
                                      node_out_dim=data.num_graphs)

        tr_pred = global_pred[:,:3] + global_pred[:,6:9]
        rot_pred = global_pred[:,3:6] + global_pred[:,9:]
        data.graph_sigma_emb = self.timestep_embd_func(data.complex_t['tr'])

        # fix the magnitude of translational and rotational score vectors
        tr_norm = torch.linalg.vector_norm(tr_pred, dim=1).unsqueeze(1)
        tr_pred = tr_pred / tr_norm * self.tra_final_layer(torch.cat([tr_norm, data.graph_sigma_emb], dim=1))
        rot_norm = torch.linalg.vector_norm(rot_pred, dim=1).unsqueeze(1)
        rot_pred = rot_pred / rot_norm * self.rot_final_layer(torch.cat([rot_norm, data.graph_sigma_emb], dim=1))

        if self.scale_by_sigma:
            tr_pred = tr_pred / tr_sigma.unsqueeze(1)
            rot_pred = rot_pred * gnn4s.data.so3.score_norm(rot_sigma.cpu()).unsqueeze(1).to(data['ligand'].x.device)

        if self.no_torsion or data['ligand'].edge_mask.sum() == 0:
            return tr_pred, rot_pred, torch.empty(0, device=self.device)

        # torsional components
        tor_bonds, tor_edge_idx, tor_edge_attr, tor_edge_sh = self.build_bond_conv_graph(data)
        tor_bond_vec = data['ligand'].pos[tor_bonds[1]] - data['ligand'].pos[tor_bonds[0]]
        tor_bond_attr = lig_node_attr[tor_bonds[0]] + lig_node_attr[tor_bonds[1]]

        tor_bonds_sh = o3.spherical_harmonics("2e", tor_bond_vec, normalize=True, normalization='component')
        tor_edge_sh = self.final_tp_tor(tor_edge_sh, tor_bonds_sh[tor_edge_idx[0]])

        tor_edge_attr = torch.cat([tor_edge_attr, lig_node_attr[tor_edge_idx[1], :self.ns],
                                   tor_bond_attr[tor_edge_idx[0], :self.ns]], -1)
        tor_pred = self.tor_bond_conv(lig_node_attr, tor_edge_idx, tor_edge_attr, tor_edge_sh,
                                      node_out_dim=data['ligand'].edge_mask.sum(), reduce='mean')
        tor_pred = self.tor_final_layer(tor_pred).squeeze(1)
        edge_sigma = tor_sigma[data['ligand'].batch][data['ligand', 'ligand'].edge_index[0]][data['ligand'].edge_mask]

        if self.scale_by_sigma:
            tor_pred = tor_pred * torch.sqrt(torch.tensor(gnn4s.data.torus.score_norm(edge_sigma.cpu().numpy())).float()
                                             .to(data['ligand'].x.device))
        return tr_pred, rot_pred, tor_pred

    def build_lig_conv_graph(self, data):
        # builds the ligand graph edges and initial node and edge features
        data['ligand'].node_sigma_emb = self.timestep_embd_func(data['ligand'].node_t['tr'])

        # compute edges
        radius_edges = radius_graph(data['ligand'].pos, self.lig_radius_max, data['ligand'].batch)
        edge_idx = torch.cat([data['ligand', 'ligand'].edge_index, radius_edges], 1).long()
        edge_attr = torch.cat([
            data['ligand', 'ligand'].edge_attr,
            torch.zeros(radius_edges.shape[-1], self.lig_edge_in_feat, device=data['ligand'].x.device)
        ], 0)

        # compute initial features
        edge_sigma_emb = data['ligand'].node_sigma_emb[edge_idx[0].long()]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
        node_attr = torch.cat([data['ligand'].x, data['ligand'].node_sigma_emb], 1)

        src, dst = edge_idx
        edge_vec = data['ligand'].pos[dst.long()] - data['ligand'].pos[src.long()]
        edge_length_emb = self.lig_dist_exps(edge_vec.norm(dim=-1))

        edge_attr = torch.cat([edge_attr, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return node_attr, edge_idx, edge_attr, edge_sh

    def build_rec_conv_graph(self, data):
        # builds the receptor initial node and edge embeddings
        data['receptor'].node_sigma_emb = self.timestep_embd_func(data['receptor'].node_t['tr']) # tr rot and tor noise is all the same
        node_attr = torch.cat([data['receptor'].x, data['receptor'].node_sigma_emb], 1)

        # this assumes the edges were already created in preprocessing since protein's structure is fixed
        edge_idx = data['receptor', 'receptor'].edge_index
        src, dst = edge_idx
        edge_vec = data['receptor'].pos[dst.long()] - data['receptor'].pos[src.long()]

        edge_length_emb = self.rec_dist_exps(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['receptor'].node_sigma_emb[edge_idx[0].long()]
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return node_attr, edge_idx, edge_attr, edge_sh

    def build_cross_conv_graph(self, data, cross_distance_cutoff):
        # builds the cross edges between ligand and receptor
        if torch.is_tensor(cross_distance_cutoff):
            # different cutoff for every graph (depends on the diffusion time)
            edge_idx = radius(data['receptor'].pos / cross_distance_cutoff[data['receptor'].batch],
                              data['ligand'].pos / cross_distance_cutoff[data['ligand'].batch], 1,
                              data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)
        else:
            edge_idx = radius(data['receptor'].pos, data['ligand'].pos, cross_distance_cutoff,
                              data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)

        src, dst = edge_idx
        edge_vec = data['receptor'].pos[dst.long()] - data['ligand'].pos[src.long()]

        edge_length_emb = self.cro_dist_exps(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['ligand'].node_sigma_emb[src.long()]
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return edge_idx, edge_attr, edge_sh

    def build_center_conv_graph(self, data):
        # builds the filter and edges for the convolution generating translational and rotational scores
        edge_idx = torch.cat([data['ligand'].batch.unsqueeze(0),
                              torch.arange(len(data['ligand'].batch)).to(data['ligand'].x.device).unsqueeze(0)], dim=0)

        center_pos = torch.zeros((data.num_graphs, 3)).to(data['ligand'].x.device)
        center_pos.index_add_(0, index=data['ligand'].batch, source=data['ligand'].pos)
        center_pos = center_pos / torch.bincount(data['ligand'].batch).unsqueeze(1)

        edge_vec = data['ligand'].pos[edge_idx[1]] - center_pos[edge_idx[0]]
        edge_attr = self.cen_dist_exps(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['ligand'].node_sigma_emb[edge_idx[1].long()]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return edge_idx, edge_attr, edge_sh

    def build_bond_conv_graph(self, data):
        # builds the graph for the convolution between the center of the rotatable bonds and the neighbouring nodes
        bonds = data['ligand', 'ligand'].edge_index[:, data['ligand'].edge_mask].long()
        bond_pos = (data['ligand'].pos[bonds[0]] + data['ligand'].pos[bonds[1]]) / 2
        bond_batch = data['ligand'].batch[bonds[0]]
        edge_idx = radius(data['ligand'].pos, bond_pos, self.lig_radius_max, batch_x=data['ligand'].batch, batch_y=bond_batch)

        edge_vec = data['ligand'].pos[edge_idx[1]] - bond_pos[edge_idx[0]]
        edge_attr = self.lig_dist_exps(edge_vec.norm(dim=-1))

        edge_attr = self.final_edge_embd(edge_attr)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return bonds, edge_idx, edge_attr, edge_sh

class TensorProductModelALL(torch.nn.Module):
    def __init__(self, t_to_sigma, timestep_embd_func, ns: int=16, nv: int=4,
                 lig_edge_in_feat: int=4, sigma_embd_dim: int=32, dist_embd_dim: int=32,
                 cro_dist_embd_dim: int=32, lm_embd_type=None, 
                 lig_radius_max: float=5.0, rec_radius_max: float=30.0, cro_dist_max: float=250.0,
                 cen_dist_max: float=30, dropout_rate=0.0, dynamic_cro_max: bool=False, 
                 use_second_order_repr=False, sh_lmax: int=2, use_batch_norm: bool=True, conv_layer_num: int=2,
                 no_torsion: bool=False, scale_by_sigma: bool=True, 
                 conf_mode: bool=False, conf_dropout_rate: float=0.0, conf_no_batch_norm: bool=False,
                 conf_output_num: int=1, device: str='cpu') -> None:
        super(TensorProductModelALL, self).__init__()
        self.t_to_sigma = t_to_sigma
        self.timestep_embd_func = timestep_embd_func
        self.ns, self.nv = ns, nv
        self.lig_edge_in_feat = lig_edge_in_feat
        self.sigma_embd_dim = sigma_embd_dim
        self.dist_embd_dim = dist_embd_dim
        self.cro_dist_embd_dim = cro_dist_embd_dim
        self.lm_embd_type = lm_embd_type
        self.lig_radius_max = lig_radius_max
        self.rec_radius_max = rec_radius_max
        self.cro_dist_max = cro_dist_max
        self.cen_dist_max = cen_dist_max
        self.dropout_rate = dropout_rate
        self.dynamic_cro_max = dynamic_cro_max
        self.use_second_order_repr = use_second_order_repr
        self.sh_lmax = sh_lmax
        self.use_batch_norm = use_batch_norm
        self.conv_layer_num = conv_layer_num
        self.no_torsion = no_torsion
        self.scale_by_sigma = scale_by_sigma
        self.conf_mode = conf_mode
        self.conf_dropout_rate = conf_dropout_rate
        self.conf_no_batch_norm = conf_no_batch_norm
        self.conf_output_num = conf_output_num
        self.device = device

        self.allowable_features = {
            'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
            'possible_chirality_list': ['CHI_UNSPECIFIED','CHI_TETRAHEDRAL_CW','CHI_TETRAHEDRAL_CCW','CHI_OTHER'],
            'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
            'possible_numring_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
            'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
            'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
            'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
            'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
            'possible_hybridization_list': ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'],
            'possible_is_aromatic_list': [False, True],
            'possible_is_in_ring3_list': [False, True],
            'possible_is_in_ring4_list': [False, True],
            'possible_is_in_ring5_list': [False, True],
            'possible_is_in_ring6_list': [False, True],
            'possible_is_in_ring7_list': [False, True],
            'possible_is_in_ring8_list': [False, True],
            'possible_amino_acids': ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                                    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
                                    'HIP', 'HIE', 'TPO', 'HID', 'LEV', 'MEU', 'PTR', 'GLV', 'CYT', 'SEP',
                                    'HIZ', 'CYM', 'GLM', 'ASQ', 'TYS', 'CYX', 'GLZ', 'misc'],
            'possible_atom_type_2': ['C*', 'CA', 'CB', 'CD', 'CE', 'CG', 'CH', 'CZ', 'N*', 'ND',
                                    'NE', 'NH', 'NZ', 'O*', 'OD', 'OE', 'OG', 'OH', 'OX', 'S*',
                                    'SD', 'SG', 'misc'],
            'possible_atom_type_3': ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3',
                                    'CG', 'CG1', 'CG2', 'CH2', 'CZ', 'CZ2', 'CZ3', 'N', 'ND1', 'ND2',
                                    'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'O', 'OD1', 'OD2', 'OE1',
                                    'OE2', 'OG', 'OG1', 'OH', 'OXT', 'SD', 'SG', 'misc'],
        }
        self.bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        self.lig_feat_dim = (list(map(len, [
            self.allowable_features['possible_atomic_num_list'],
            self.allowable_features['possible_chirality_list'],
            self.allowable_features['possible_degree_list'],
            self.allowable_features['possible_formal_charge_list'],
            self.allowable_features['possible_implicit_valence_list'],
            self.allowable_features['possible_numH_list'],
            self.allowable_features['possible_number_radical_e_list'],
            self.allowable_features['possible_hybridization_list'],
            self.allowable_features['possible_is_aromatic_list'],
            self.allowable_features['possible_numring_list'],
            self.allowable_features['possible_is_in_ring3_list'],
            self.allowable_features['possible_is_in_ring4_list'],
            self.allowable_features['possible_is_in_ring5_list'],
            self.allowable_features['possible_is_in_ring6_list'],
            self.allowable_features['possible_is_in_ring7_list'],
            self.allowable_features['possible_is_in_ring8_list'],
        ])), 0)  # number of scalar features

        self.rec_res_feat_dim = (list(map(len, [
            self.allowable_features['possible_amino_acids']
        ])), 0)

        self.rec_atom_feat_dim = (list(map(len, [
            self.allowable_features['possible_amino_acids'],
            self.allowable_features['possible_atomic_num_list'],
            self.allowable_features['possible_atom_type_2'],
            self.allowable_features['possible_atom_type_3'],
        ])), 0)

        # embedding layers
        self.lig_node_embd = AtomEncoder(embd_dim=self.ns, feat_dim=self.lig_feat_dim,
                                         sigma_embd_dim=self.sigma_embd_dim)
        self.lig_edge_embd = MLP(input_dim=self.lig_edge_in_feat+self.sigma_embd_dim+self.dist_embd_dim,
                                 output_dim=self.ns, hidden_dim=self.ns, layer_num=1, 
                                 activation=nn.ReLU(), dropout_rate=self.dropout_rate)

        self.rec_node_embd = AtomEncoder(embd_dim=self.ns, feat_dim=self.rec_res_feat_dim, 
                                         sigma_embd_dim=self.sigma_embd_dim, lm_embd_type=self.lm_embd_type)
        self.rec_edge_embd = MLP(input_dim=self.sigma_embd_dim+self.dist_embd_dim,
                                 output_dim=self.ns, hidden_dim=self.ns, layer_num=1, 
                                 activation=nn.ReLU(), dropout_rate=self.dropout_rate)

        self.atom_node_embd = AtomEncoder(embd_dim=self.ns, feat_dim=self.rec_atom_feat_dim, 
                                          sigma_embd_dim=self.sigma_embd_dim)
        self.atom_edge_embd = MLP(input_dim=self.sigma_embd_dim+self.dist_embd_dim,
                                  output_dim=self.ns, hidden_dim=self.ns, layer_num=1, 
                                  activation=nn.ReLU(), dropout_rate=self.dropout_rate)

        self.lr_edge_embd = MLP(input_dim=self.sigma_embd_dim+self.cro_dist_embd_dim,
                                output_dim=self.ns, hidden_dim=self.ns, layer_num=1, 
                                activation=nn.ReLU(), dropout_rate=self.dropout_rate)
        self.ar_edge_embd = MLP(input_dim=self.sigma_embd_dim+self.dist_embd_dim,
                                output_dim=self.ns, hidden_dim=self.ns, layer_num=1, 
                                activation=nn.ReLU(), dropout_rate=self.dropout_rate)
        self.la_edge_embd = MLP(input_dim=self.sigma_embd_dim+self.cro_dist_embd_dim,
                                output_dim=self.ns, hidden_dim=self.ns, layer_num=1, 
                                activation=nn.ReLU(), dropout_rate=self.dropout_rate)
        
        self.lig_dist_exps = GaussianSmearing(0.0, self.lig_radius_max, self.dist_embd_dim)
        self.rec_dist_exps = GaussianSmearing(0.0, self.rec_radius_max, self.dist_embd_dim)
        self.cro_dist_exps = GaussianSmearing(0.0, self.cro_dist_max, self.cro_dist_embd_dim)

        # convolutional layer
        # irreducible representations
        if self.use_second_order_repr:
            irrep_seq = [
                f'{self.ns}x0e',
                f'{self.ns}x0e + {self.nv}x1o + {self.nv}x2e',
                f'{self.ns}x0e + {self.nv}x1o + {self.nv}x2e + {self.nv}x1e + {self.nv}x2o',
                f'{self.ns}x0e + {self.nv}x1o + {self.nv}x2e + {self.nv}x1e + {self.nv}x2o + {self.ns}x0o'
            ]
        else:
            irrep_seq = [
                f'{self.ns}x0e',
                f'{self.ns}x0e + {self.nv}x1o',
                f'{self.ns}x0e + {self.nv}x1o + {self.nv}x1e',
                f'{self.ns}x0e + {self.nv}x1o + {self.nv}x1e + {self.ns}x0o'
            ]
        
        # convolutional layers
        conv_layer = []
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=self.sh_lmax)
        for i in range(self.conv_layer_num):
            in_irreps = irrep_seq[min(i, len(irrep_seq)-1)]
            ou_irreps = irrep_seq[min(i+1, len(irrep_seq)-1)]
            parameters = {
                'in_irreps': in_irreps,
                'sh_irreps': self.sh_irreps,
                'ou_irreps': ou_irreps,
                'edge_feat_num': 3*self.ns,
                'hidden_feat_num': 3*self.ns,
                'use_residual': False,
                'use_batch_norm': self.use_batch_norm,
                'dropout_rate': self.dropout_rate
            }
            for _ in range(9): # 3 intra & 6 inter per each layer
                conv_layer.append(TensorProductConvLayer(**parameters))

        self.conv_layer = nn.ModuleList(conv_layer)

        if self.conf_mode:
            # confidence layer
            self.conf_predictor = nn.Sequential(
                nn.Linear(2*self.ns if self.conv_layer_num >= 3 else self.ns, self.ns),
                nn.BatchNorm1d(self.ns) if not self.conf_no_batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(self.conf_dropout_rate),
                nn.Linear(self.ns, self.ns),
                nn.BatchNorm1d(self.ns) if not self.conf_no_batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(self.conf_dropout_rate),
                nn.Linear(self.ns, self.conf_output_num)
            )
        else:
            # convolution for translational and rotational scores
            self.cen_dist_exps = GaussianSmearing(0.0, cen_dist_max, dist_embd_dim)
            self.cen_edge_embd = MLP(input_dim=self.dist_embd_dim+self.sigma_embd_dim,
                                     output_dim=self.ns, hidden_dim=self.ns, layer_num=1, 
                                     activation=nn.ReLU(), dropout_rate=self.dropout_rate)
            # final layer
            self.final_conv = TensorProductConvLayer(
                in_irreps=self.lig_conv_layer[-1].ou_irreps,
                sh_irreps=self.sh_irreps,
                ou_irreps=f'2x1o + 2x1e',
                edge_feat_num=2*self.ns,
                use_residual=False,
                dropout_rate=self.dropout_rate,
                use_batch_norm=self.use_batch_norm
            )
            self.tra_final_layer = MLP(input_dim=1+self.sigma_embd_dim,
                                       output_dim=1, hidden_dim=self.ns, layer_num=1, 
                                       activation=nn.ReLU(), dropout_rate=self.dropout_rate)
            self.rot_final_layer = MLP(input_dim=1+self.sigma_embd_dim,
                                       output_dim=1, hidden_dim=self.ns, layer_num=1, 
                                       activation=nn.ReLU(), dropout_rate=self.dropout_rate)
            
            if not self.no_torsion:
                # torsion angles components
                self.final_edge_embd = MLP(input_dim=self.dist_embd_dim,
                                           output_dim=self.ns, hidden_dim=self.ns, layer_num=1, 
                                           activation=nn.ReLU(), dropout_rate=self.dropout_rate)
                self.final_tp_tor = o3.FullTensorProduct(self.sh_irreps, "2e")
                self.tor_bond_conv = TensorProductConvLayer(
                    in_irreps=self.lig_conv_layer[-1].ou_irreps,
                    sh_irreps=self.final_tp_tor.irreps_out,
                    ou_irreps=f'{self.ns}x0o + {self.ns}x0e',
                    edge_feat_num=3*self.ns,
                    use_residual=False,
                    dropout_rate=self.dropout_rate,
                    use_batch_norm=self.use_batch_norm
                )
                self.tor_final_layer = MLP(input_dim=2*self.ns, output_dim=1, hidden_dim=self.ns, 
                                           layer_num=1, bias=False, activation=nn.Tanh(), 
                                           dropout_rate=self.dropout_rate)
    
    def forward(self, data):
        if not self.conf_mode:
            tr_sigma, rot_sigma, tor_sigma = self.t_to_sigma(
                *[data.complex_t[noise_type] for noise_type in ['tr', 'rot', 'tor']])
        else:
            tr_sigma, rot_sigma, tor_sigma = [
                data.complex_t[noise_type] for noise_type in ['tr', 'rot', 'tor']]

        # build ligand graph
        lig_node_attr, lig_edge_idx, lig_edge_attr, lig_edge_sh = self.build_lig_conv_graph(data)
        lig_src, lig_dst = lig_edge_idx
        lig_node_attr = self.lig_node_embd(lig_node_attr)
        lig_edge_attr = self.lig_edge_embd(lig_edge_attr)

        # build receptor graph
        rec_node_attr, rec_edge_idx, rec_edge_attr, rec_edge_sh = self.build_rec_conv_graph(data)
        rec_src, rec_dst = rec_edge_idx
        rec_node_attr = self.rec_node_embd(rec_node_attr)
        rec_edge_attr = self.rec_edge_embd(rec_edge_attr)

        # build atom graph
        atom_node_attr, atom_edge_idx, atom_edge_attr, atom_edge_sh = self.build_atom_conv_graph(data)
        atom_src, atom_dst = atom_edge_idx
        atom_node_attr = self.atom_node_embd(atom_node_attr)
        atom_edge_attr = self.atom_edge_embd(atom_edge_attr)

        # build cross graph
        if self.dynamic_cro_max:
            cross_cutoff = (tr_sigma * 3 + 20).unsqueeze(1)
        else:
            cross_cutoff = self.cro_dist_max
        lr_edge_idx, lr_edge_attr, lr_edge_sh, la_edge_idx, la_edge_attr, \
            la_edge_sh, ar_edge_idx, ar_edge_attr, ar_edge_sh = self.build_cross_conv_graph(data, cross_cutoff)
        lr_edge_attr = self.lr_edge_embd(lr_edge_attr)
        la_edge_attr = self.la_edge_embd(la_edge_attr)
        ar_edge_attr = self.ar_edge_embd(ar_edge_attr)

        for i in range(self.conv_layer_num):
            # ligand updates
            lig_edge_attr_ = torch.cat([lig_edge_attr, lig_node_attr[lig_edge_idx[0], :self.ns],
                                        lig_node_attr[lig_edge_idx[1], :self.ns]], -1)
            lig_update = self.conv_layer[9*i](lig_node_attr, lig_edge_idx, lig_edge_attr_, lig_edge_sh)

            lr_edge_attr_ = torch.cat([lr_edge_attr, lig_node_attr[lr_edge_idx[0], :self.ns],
                                       rec_node_attr[lr_edge_idx[1], :self.ns]], -1)
            lr_update = self.conv_layer[9*i+1](rec_node_attr, lr_edge_idx, lr_edge_attr_, lr_edge_sh,
                                               node_out_dim=lig_node_attr.shape[0])

            la_edge_attr_ = torch.cat([la_edge_attr, lig_node_attr[la_edge_idx[0], :self.ns],
                                       atom_node_attr[la_edge_idx[1], :self.ns]], -1)
            la_update = self.conv_layer[9*i+2](atom_node_attr, la_edge_idx, la_edge_attr_, la_edge_sh,
                                               node_out_dim=lig_node_attr.shape[0])

            if i != self.conv_layer_num-1:  # last layer optimisation
                # atom updates
                atom_edge_attr_ = torch.cat([atom_edge_attr, atom_node_attr[atom_edge_idx[0], :self.ns],
                                             atom_node_attr[atom_edge_idx[1], :self.ns]], -1)
                atom_update = self.conv_layer[9*i+3](atom_node_attr, atom_edge_idx, atom_edge_attr_, atom_edge_sh)

                al_edge_attr_ = torch.cat([la_edge_attr, atom_node_attr[la_edge_idx[1], :self.ns],
                                           lig_node_attr[la_edge_idx[0], :self.ns]], -1)
                al_update = self.conv_layer[9*i+4](lig_node_attr, torch.flip(la_edge_idx, dims=[0]), al_edge_attr_,
                                                   la_edge_sh, node_out_dim=atom_node_attr.shape[0])

                ar_edge_attr_ = torch.cat([ar_edge_attr, atom_node_attr[ar_edge_idx[0], :self.ns],
                                           rec_node_attr[ar_edge_idx[1], :self.ns]],-1)
                ar_update = self.conv_layer[9*i+5](rec_node_attr, ar_edge_idx, ar_edge_attr_, ar_edge_sh,
                                                   node_out_dim=atom_node_attr.shape[0])

                # recptor updates
                rec_edge_attr_ = torch.cat([rec_edge_attr, rec_node_attr[rec_edge_idx[0], :self.ns],
                                            rec_node_attr[rec_edge_idx[1], :self.ns]], -1)
                rec_update = self.conv_layer[9*i+6](rec_node_attr, rec_edge_idx, rec_edge_attr_, rec_edge_sh)

                rl_edge_attr_ = torch.cat([lr_edge_attr, rec_node_attr[lr_edge_idx[1], :self.ns],
                                           lig_node_attr[lr_edge_idx[0], :self.ns]], -1)
                rl_update = self.conv_layer[9*i+7](lig_node_attr, torch.flip(lr_edge_idx, dims=[0]), rl_edge_attr_,
                                                   lr_edge_sh, node_out_dim=rec_node_attr.shape[0])

                ra_edge_attr_ = torch.cat([ar_edge_attr, rec_node_attr[ar_edge_idx[1], :self.ns],
                                           atom_node_attr[ar_edge_idx[0], :self.ns]], -1)
                ra_update = self.conv_layer[9*i+8](atom_node_attr, torch.flip(ar_edge_idx, dims=[0]), ra_edge_attr_,
                                                   ar_edge_sh, node_out_dim=rec_node_attr.shape[0])

            # padding original features and update features with residual updates
            lig_node_attr = F.pad(lig_node_attr, (0, lig_update.shape[-1] - lig_node_attr.shape[-1]))
            lig_node_attr = lig_node_attr + lig_update + la_update + lr_update

            if i != self.conv_layer_num - 1:  # last layer optimisation
                atom_node_attr = F.pad(atom_node_attr, (0, atom_update.shape[-1] - rec_node_attr.shape[-1]))
                atom_node_attr = atom_node_attr + atom_update + al_update + ar_update
                rec_node_attr = F.pad(rec_node_attr, (0, rec_update.shape[-1] - rec_node_attr.shape[-1]))
                rec_node_attr = rec_node_attr + rec_update + ra_update + rl_update

        # confidence and affinity prediction
        if self.conf_mode:
            if self.conv_layer_num >= 3:
                scalar_lig_attr = torch.cat([lig_node_attr[:,:self.ns],lig_node_attr[:,-self.ns:]], dim=1)
            else:
                scalar_lig_attr = lig_node_attr[:,:self.ns]
            confidence = self.conf_predictor(scatter_mean(scalar_lig_attr, data['ligand'].batch, dim=0)).squeeze(dim=-1)
            return confidence

        # compute translational and rotational score vectors
        center_edge_idx, center_edge_attr, center_edge_sh = self.build_center_conv_graph(data)
        center_edge_attr = self.cen_edge_embd(center_edge_attr)
        center_edge_attr = torch.cat([center_edge_attr, lig_node_attr[center_edge_idx[1], :self.ns]], -1)
        global_pred = self.final_conv(lig_node_attr, center_edge_idx, center_edge_attr, center_edge_sh, 
                                      node_out_dim=data.num_graphs)

        tr_pred = global_pred[:,:3] + global_pred[:,6:9]
        rot_pred = global_pred[:,3:6] + global_pred[:,9:]
        data.graph_sigma_emb = self.timestep_embd_func(data.complex_t['tr'])

        # fix the magnitude of translational and rotational score vectors
        tr_norm = torch.linalg.vector_norm(tr_pred, dim=1).unsqueeze(1)
        tr_pred = tr_pred / tr_norm * self.tra_final_layer(torch.cat([tr_norm, data.graph_sigma_emb], dim=1))
        rot_norm = torch.linalg.vector_norm(rot_pred, dim=1).unsqueeze(1)
        rot_pred = rot_pred / rot_norm * self.rot_final_layer(torch.cat([rot_norm, data.graph_sigma_emb], dim=1))

        if self.scale_by_sigma:
            tr_pred = tr_pred / tr_sigma.unsqueeze(1)
            rot_pred = rot_pred * gnn4s.data.so3.score_norm(rot_sigma.cpu()).unsqueeze(1).to(data['ligand'].x.device)

        if self.no_torsion or data['ligand'].edge_mask.sum() == 0:
            return tr_pred, rot_pred, torch.empty(0, device=self.device)

        # torsional components
        tor_bonds, tor_edge_idx, tor_edge_attr, tor_edge_sh = self.build_bond_conv_graph(data)
        tor_bond_vec = data['ligand'].pos[tor_bonds[1]] - data['ligand'].pos[tor_bonds[0]]
        tor_bond_attr = lig_node_attr[tor_bonds[0]] + lig_node_attr[tor_bonds[1]]

        tor_bonds_sh = o3.spherical_harmonics("2e", tor_bond_vec, normalize=True, normalization='component')
        tor_edge_sh = self.final_tp_tor(tor_edge_sh, tor_bonds_sh[tor_edge_idx[0]])

        tor_edge_attr = torch.cat([tor_edge_attr, lig_node_attr[tor_edge_idx[1], :self.ns],
                                   tor_bond_attr[tor_edge_idx[0], :self.ns]], -1)
        tor_pred = self.tor_bond_conv(lig_node_attr, tor_edge_idx, tor_edge_attr, tor_edge_sh,
                                      node_out_dim=data['ligand'].edge_mask.sum(), reduce='mean')
        tor_pred = self.tor_final_layer(tor_pred).squeeze(1)
        edge_sigma = tor_sigma[data['ligand'].batch][data['ligand', 'ligand'].edge_index[0]][data['ligand'].edge_mask]

        if self.scale_by_sigma:
            tor_pred = tor_pred * torch.sqrt(torch.tensor(gnn4s.data.torus.score_norm(edge_sigma.cpu().numpy())).float()
                                             .to(data['ligand'].x.device))
        return tr_pred, rot_pred, tor_pred

    def build_lig_conv_graph(self, data):
        # builds the ligand graph edges and initial node and edge features
        data['ligand'].node_sigma_emb = self.timestep_embd_func(data['ligand'].node_t['tr'])

        # compute edges
        radius_edges = radius_graph(data['ligand'].pos, self.lig_radius_max, data['ligand'].batch)
        edge_idx = torch.cat([data['ligand', 'ligand'].edge_index, radius_edges], 1).long()
        edge_attr = torch.cat([
            data['ligand', 'ligand'].edge_attr,
            torch.zeros(radius_edges.shape[-1], self.lig_edge_in_feat, device=data['ligand'].x.device)
        ], 0)

        # compute initial features
        edge_sigma_emb = data['ligand'].node_sigma_emb[edge_idx[0].long()]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
        node_attr = torch.cat([data['ligand'].x, data['ligand'].node_sigma_emb], 1)

        src, dst = edge_idx
        edge_vec = data['ligand'].pos[dst.long()] - data['ligand'].pos[src.long()]
        edge_length_emb = self.lig_dist_exps(edge_vec.norm(dim=-1))

        edge_attr = torch.cat([edge_attr, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return node_attr, edge_idx, edge_attr, edge_sh

    def build_rec_conv_graph(self, data):
        # builds the receptor initial node and edge embeddings
        # tr rot and tor noise is all the same
        data['receptor'].node_sigma_emb = self.timestep_embd_func(data['receptor'].node_t['tr'])
        node_attr = torch.cat([data['receptor'].x, data['receptor'].node_sigma_emb], 1)

        # this assumes the edges were already created in preprocessing since protein's structure is fixed
        edge_idx = data['receptor', 'receptor'].edge_index
        src, dst = edge_idx
        edge_vec = data['receptor'].pos[dst.long()] - data['receptor'].pos[src.long()]

        edge_length_emb = self.rec_dist_exps(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['receptor'].node_sigma_emb[edge_idx[0].long()]
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return node_attr, edge_idx, edge_attr, edge_sh

    def build_atom_conv_graph(self, data):
        # build the graph between receptor atoms
        data['atom'].node_sigma_emb = self.timestep_embd_func(data['atom'].node_t['tr'])
        node_attr = torch.cat([data['atom'].x, data['atom'].node_sigma_emb], 1)

        # this assumes the edges were already created in preprocessing since protein's structure is fixed
        edge_idx = data['atom', 'atom'].edge_index
        src, dst = edge_idx
        edge_vec = data['atom'].pos[dst.long()] - data['atom'].pos[src.long()]

        edge_length_emb = self.lig_dist_exps(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['atom'].node_sigma_emb[edge_idx[0].long()]
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return node_attr, edge_idx, edge_attr, edge_sh

    def build_cross_conv_graph(self, data, lr_cross_distance_cutoff):
        # build the cross edges between ligan atoms, receptor residues and receptor atoms

        # LIGAND to RECEPTOR
        if torch.is_tensor(lr_cross_distance_cutoff):
            # different cutoff for every graph
            lr_edge_idx = radius(data['receptor'].pos / lr_cross_distance_cutoff[data['receptor'].batch],
                                data['ligand'].pos / lr_cross_distance_cutoff[data['ligand'].batch], 1,
                                data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)
        else:
            lr_edge_idx = radius(data['receptor'].pos, data['ligand'].pos, lr_cross_distance_cutoff,
                            data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)

        lr_edge_vec = data['receptor'].pos[lr_edge_idx[1].long()] - data['ligand'].pos[lr_edge_idx[0].long()]
        lr_edge_length_emb = self.cro_dist_exps(lr_edge_vec.norm(dim=-1))
        lr_edge_sigma_emb = data['ligand'].node_sigma_emb[lr_edge_idx[0].long()]
        lr_edge_attr = torch.cat([lr_edge_sigma_emb, lr_edge_length_emb], 1)
        lr_edge_sh = o3.spherical_harmonics(self.sh_irreps, lr_edge_vec, normalize=True, normalization='component')

        # LIGAND to ATOM
        la_edge_idx = radius(data['atom'].pos, data['ligand'].pos, self.lig_radius_max,
                               data['atom'].batch, data['ligand'].batch, max_num_neighbors=10000)

        la_edge_vec = data['atom'].pos[la_edge_idx[1].long()] - data['ligand'].pos[la_edge_idx[0].long()]
        la_edge_length_emb = self.cro_dist_exps(la_edge_vec.norm(dim=-1))
        la_edge_sigma_emb = data['ligand'].node_sigma_emb[la_edge_idx[0].long()]
        la_edge_attr = torch.cat([la_edge_sigma_emb, la_edge_length_emb], 1)
        la_edge_sh = o3.spherical_harmonics(self.sh_irreps, la_edge_vec, normalize=True, normalization='component')

        # ATOM to RECEPTOR
        ar_edge_idx = data['atom', 'receptor'].edge_index
        ar_edge_vec = data['receptor'].pos[ar_edge_idx[1].long()] - data['atom'].pos[ar_edge_idx[0].long()]
        ar_edge_length_emb = self.rec_dist_exps(ar_edge_vec.norm(dim=-1))
        ar_edge_sigma_emb = data['atom'].node_sigma_emb[ar_edge_idx[0].long()]
        ar_edge_attr = torch.cat([ar_edge_sigma_emb, ar_edge_length_emb], 1)
        ar_edge_sh = o3.spherical_harmonics(self.sh_irreps, ar_edge_vec, normalize=True, normalization='component')

        return lr_edge_idx, lr_edge_attr, lr_edge_sh, la_edge_idx, la_edge_attr, \
               la_edge_sh, ar_edge_idx, ar_edge_attr, ar_edge_sh

    def build_center_conv_graph(self, data):
        # builds the filter and edges for the convolution generating translational and rotational scores
        edge_idx = torch.cat([data['ligand'].batch.unsqueeze(0),
                              torch.arange(len(data['ligand'].batch)).to(data['ligand'].x.device).unsqueeze(0)], dim=0)

        center_pos = torch.zeros((data.num_graphs, 3)).to(data['ligand'].x.device)
        center_pos.index_add_(0, index=data['ligand'].batch, source=data['ligand'].pos)
        center_pos = center_pos / torch.bincount(data['ligand'].batch).unsqueeze(1)

        edge_vec = data['ligand'].pos[edge_idx[1]] - center_pos[edge_idx[0]]
        edge_attr = self.cen_dist_exps(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['ligand'].node_sigma_emb[edge_idx[1].long()]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return edge_idx, edge_attr, edge_sh

    def build_bond_conv_graph(self, data):
        # builds the graph for the convolution between the center of the rotatable bonds and the neighbouring nodes
        bonds = data['ligand', 'ligand'].edge_index[:, data['ligand'].edge_mask].long()
        bond_pos = (data['ligand'].pos[bonds[0]] + data['ligand'].pos[bonds[1]]) / 2
        bond_batch = data['ligand'].batch[bonds[0]]
        edge_idx = radius(data['ligand'].pos, bond_pos, self.lig_radius_max, batch_x=data['ligand'].batch, batch_y=bond_batch)

        edge_vec = data['ligand'].pos[edge_idx[1]] - bond_pos[edge_idx[0]]
        edge_attr = self.lig_dist_exps(edge_vec.norm(dim=-1))

        edge_attr = self.final_edge_embd(edge_attr)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return bonds, edge_idx, edge_attr, edge_sh
