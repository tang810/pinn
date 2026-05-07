import copy
import torch
import numpy as np
import scipy.spatial as spa
from rdkit.Chem import GetPeriodicTable
from torch_cluster import radius_graph
from scipy.special import softmax

class Protein:
    def __init__(self, protein):
        self.protein = protein
        self.extract_structure()

        self.allowable_features = {
            'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
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

    def extract_structure(self):
        # cord: [chain_num, res_num, atom_num, dim]
        self.cord, self.a_cord, self.n_cord, self.c_cord = [], [], [], []
        self.valid_chain_id, self.valid_res_id = [], []
        self.invalid_chain_id, self.invalid_res_id = [], []
        self.length = []
        
        for chain_idx, chain in enumerate(self.protein):
            chain_cord, chain_a_cord, chain_n_cord, chain_c_cord = [], [], [], []
            chain_valid_res_id, chain_invalid_res_id = [], []
            chain_length = 0
        
            for res_idx, residue in enumerate(chain):
                if residue.get_resname() == 'HOH':
                    chain_invalid_res_id.append(residue.get_id())
                    continue
        
                res_cord = []
                res_a_cord, res_n_cord, res_c_cord = None, None, None
                for atom in residue:
                    if atom.name == 'CA':
                        res_a_cord = list(atom.get_vector())
                    if atom.name == 'N':
                        res_n_cord = list(atom.get_vector())
                    if atom.name == 'C':
                        res_c_cord = list(atom.get_vector())
                    res_cord.append(list(atom.get_vector()))

                # only append residue if it is an amino acid and not some weird molecule that is part of the complex
                if res_a_cord != None and res_n_cord != None and res_c_cord != None:
                    chain_cord.append(np.array(res_cord))
                    chain_a_cord.append(res_a_cord)
                    chain_n_cord.append(res_n_cord)
                    chain_c_cord.append(res_c_cord)
                    chain_valid_res_id.append(residue.get_id())
                    chain_length += 1
                else:
                    chain_invalid_res_id.append(residue.get_id())

            self.cord.append(chain_cord)
            self.a_cord.append(np.array(chain_a_cord))
            self.n_cord.append(np.array(chain_n_cord))
            self.c_cord.append(np.array(chain_c_cord))
            self.valid_res_id.append(chain_valid_res_id)
            self.invalid_res_id.append(chain_invalid_res_id)
            if chain_length != 0:
                self.valid_chain_id.append(chain.get_id())
            else:
                self.invalid_chain_id.append(chain.get_id())
            self.length.append(chain_length)

    def get_valid_coordinate(self, lig, lm_embd_chain=None):
        rec = copy.deepcopy(self.protein)
        lig_cord = lig.GetConformer().GetPositions()

        min_distances = []
        for chain_idx, chain in enumerate(rec):
            if self.length[chain_idx] > 0:
                distance = spa.distance.cdist(lig_cord, np.concatenate(self.cord[chain_idx], axis=0))
                min_distance = distance.min()
            else:
                min_distance = np.inf
            min_distances.append(min_distance)
        
        min_distance = np.array(min_distances)
        valid_chain_id = copy.deepcopy(self.valid_chain_id)
        if len(valid_chain_id) == 0:
            valid_chain_id.append(np.argmin(min_distance))
        invalid_chain_id = []

        valid_cord, valid_a_cord, valid_n_cord, valid_c_cord = [], [], [], []
        valid_length = []
        valid_lm_embd = []
        for chain_idx, chain in enumerate(rec):
            if chain.get_id() in valid_chain_id:
                valid_cord.append(self.cord[chain_idx])
                valid_a_cord.append(self.a_cord[chain_idx])
                valid_n_cord.append(self.n_cord[chain_idx])
                valid_c_cord.append(self.c_cord[chain_idx])
                valid_length.append(self.length[chain_idx])

                if lm_embd_chain is not None:
                    if chain_idx >= len(lm_embd_chain):
                        raise ValueError('Encountered valid chain id that was not present in the LM embeddings')
                    valid_lm_embd.append(lm_embd_chain[chain_idx])
                
                for res_id in self.invalid_res_id[chain_idx]:
                    chain.detach_child(res_id)
            else:
                invalid_chain_id.append(chain.get_id())
        
        for invalid_id in invalid_chain_id:
            rec.detach_child(invalid_id)

        cord = [item for sublist in valid_cord for item in sublist]  # list with n_residues arrays: [n_atoms, 3]
        a_cord = np.concatenate(valid_a_cord, axis=0)  # [n_residues, 3]
        n_cord = np.concatenate(valid_n_cord, axis=0)  # [n_residues, 3]
        c_cord = np.concatenate(valid_c_cord, axis=0)  # [n_residues, 3]
        lm_embd = np.concatenate(valid_lm_embd, axis=0) if lm_embd_chain is not None else None
        
        assert len(a_cord) == len(n_cord)
        assert len(a_cord) == len(c_cord)
        assert sum(valid_length) == len(a_cord)
        return rec, cord, a_cord, n_cord, c_cord, lm_embd

    def safe_index(self, list, elem):
        """ Return index of element in list.
            If elem is not present, return the last index
        """
        try:
            return list.index(elem)
        except:
            return len(list) - 1
    
    def get_residue_feature(self, rec):
        feature_list = []
        for residue in rec.get_residues():
            feature_list.append([self.safe_index(self.allowable_features['possible_amino_acids'], residue.get_resname())])
        return torch.tensor(feature_list, dtype=torch.float32)  # (N_res, 1)
    
    def get_atom_feature(self, rec):
        atom_feats = []
        for i, atom in enumerate(rec.get_atoms()):
            atom_name, element = atom.name, atom.element
            if element == 'CD':
                element = 'C'
            assert not element == ''
            try:
                atomic_num = GetPeriodicTable().GetAtomicNumber(element)
            except:
                atomic_num = -1
            atom_feat = [self.safe_index(self.allowable_features['possible_amino_acids'], atom.get_parent().get_resname()),
                         self.safe_index(self.allowable_features['possible_atomic_num_list'], atomic_num),
                         self.safe_index(self.allowable_features['possible_atom_type_2'], (atom_name + '*')[:2]),
                         self.safe_index(self.allowable_features['possible_atom_type_3'], atom_name)]
            atom_feats.append(atom_feat)

        return atom_feats

    def get_calpha_graph(self, rec, a_cord: np.ndarray, n_cord: np.ndarray, c_cord: np.ndarray,
                         cutoff: float=20, nb_num_max: int=None, lm_embd: np.ndarray=None):
        n_rel_pos = n_cord - a_cord
        c_rel_pos = c_cord - a_cord
        residue_num = len(a_cord)
        if residue_num <= 1:
            raise ValueError(f"rec contains only 1 residue!")

        # Build the k-NN graph
        distance = spa.distance.cdist(a_cord, a_cord)
        src_list, dst_list = [], []
        mean_norm_list = []
        for i in range(residue_num):
            dst = list(np.where(distance[i, :] < cutoff)[0])
            dst.remove(i)
            if nb_num_max != None and len(dst) > nb_num_max:
                dst = list(np.argsort(distance[i, :]))[1: nb_num_max + 1]
            if len(dst) == 0:
                dst = list(np.argsort(distance[i, :]))[1:2]  # choose second because first is i itself
                print(f'The c_alpha_cutoff {cutoff} was too small for one c_alpha such that it had no neighbors. '
                    f'So we connected it to the closest other c_alpha')
            assert i not in dst
            src = [i] * len(dst)
            src_list.extend(src)
            dst_list.extend(dst)

            valid_dist_np = distance[i, dst]
            sigma = np.array([1., 2., 5., 10., 30.]).reshape((-1, 1))
            weight = softmax(- valid_dist_np.reshape((1, -1)) ** 2 / sigma, axis=1)  # (sigma_num, neigh_num)
            assert weight[0].sum() > 1 - 1e-2 and weight[0].sum() < 1.01

            diff_vec = a_cord[src, :] - a_cord[dst, :]  # (neigh_num, 3)
            mean_vec = weight.dot(diff_vec)  # (sigma_num, 3)
            denominator = weight.dot(np.linalg.norm(diff_vec, axis=1))  # (sigma_num,)
            mean_vec_ratio_norm = np.linalg.norm(mean_vec, axis=1) / denominator  # (sigma_num,)
            mean_norm_list.append(mean_vec_ratio_norm)
        assert len(src_list) == len(dst_list)

        node_fea = self.get_residue_feature(rec)
        node_mu_r_norm = torch.from_numpy(np.array(mean_norm_list).astype(np.float32))
        node_side_chain_vec = torch.from_numpy(
            np.concatenate([np.expand_dims(n_rel_pos, axis=1), np.expand_dims(c_rel_pos, axis=1)], axis=1))
        
        node_fea = torch.cat([node_fea, torch.tensor(lm_embd)], axis=1) if lm_embd is not None else node_fea
        node_pos = torch.from_numpy(a_cord).float()
        node_side_chain_vec = node_side_chain_vec.float()
        edge_idx = torch.from_numpy(np.asarray([src_list, dst_list]))

        return node_fea, node_pos, node_mu_r_norm, node_side_chain_vec, edge_idx
    
    def get_atom_graph(self, rec, rec_cord, atom_cutoff=5, atom_nb_num_max=None, remove_hs=False):
        src_c_alpha_idx = np.concatenate([np.asarray([i]*len(l)) for i, l in enumerate(rec_cord)])
        atom_feat = torch.from_numpy(np.asarray(self.get_atom_feature(rec)))
        atom_cord = torch.from_numpy(np.concatenate(rec_cord, axis=0)).float()

        if remove_hs:
            not_hs = (atom_feat[:,1] != 0)
            src_c_alpha_idx = src_c_alpha_idx[not_hs]
            atom_feat = atom_feat[not_hs]
            atom_cord = atom_cord[not_hs]

        atoms_edge_idx = radius_graph(atom_cord, atom_cutoff, max_num_neighbors=
                                      atom_nb_num_max if atom_nb_num_max else 1000)
        atom_res_edge_idx = torch.from_numpy(np.asarray([np.arange(len(atom_feat)), src_c_alpha_idx])).long()

        atom_x = atom_feat
        atom_pos = atom_cord
        atom_atom_edge_idx = atoms_edge_idx
        rec_atom_edge_idx = atom_res_edge_idx
        
        return atom_x, atom_pos, atom_atom_edge_idx, rec_atom_edge_idx
        
    def get_rec_graph(self, lig, rec_embd, rec_cutoff_radius, ca_nb_num_max: int=None, 
                      all_atom: bool=False, atom_cutoff_radius: float=5.0,
                      atom_nb_num_max: int=None, remove_hs: bool=False):
        node_fea, node_pos, node_mu_r_norm, node_side_chain_vec, edge_idx = None, None, None, None, None
        atom_x, atom_pos, atom_atom_edge_idx, rec_atom_edge_idx = None, None, None, None

        rec, rec_cord, a_cord, n_cord, c_cord, lm_embd = self.get_valid_coordinate(lig, rec_embd)
        
        fail = False
        if lm_embd is not None and len(a_cord) != len(lm_embd):
            fail = True
            return (fail, node_fea, node_pos, node_mu_r_norm, node_side_chain_vec, edge_idx,
                    atom_x, atom_pos, atom_atom_edge_idx, rec_atom_edge_idx)
        
        if all_atom:
            atom_x, atom_pos, atom_atom_edge_idx, rec_atom_edge_idx = self.get_atom_graph(
                rec, rec_cord, atom_cutoff=atom_cutoff_radius,
                atom_nb_num_max=atom_nb_num_max, remove_hs=remove_hs)
        
        node_fea, node_pos, node_mu_r_norm, node_side_chain_vec, edge_idx = self.get_calpha_graph(
            rec, a_cord, n_cord, c_cord, rec_cutoff_radius, ca_nb_num_max, lm_embd=lm_embd)

        return (fail, node_fea, node_pos, node_mu_r_norm, node_side_chain_vec, edge_idx,
                atom_x, atom_pos, atom_atom_edge_idx, rec_atom_edge_idx)
