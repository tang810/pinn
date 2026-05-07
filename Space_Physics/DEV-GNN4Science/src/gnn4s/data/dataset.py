import os
import pickle
import copy
import binascii
import torch
import numpy as np
import networkx as nx
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.data import Dataset as PyGDataset
from torch_geometric.data import InMemoryDataset as PyGInMemoryDataset
from torch_geometric.utils import from_networkx
from torch_geometric.loader import DataLoader as PyGDataLoader
# from dgl.data import DGLDataset
# from dgl.dataloading import GraphDataLoader
from tqdm import tqdm
from rdkit.Chem import MolFromSmiles, AddHs, AllChem
from multiprocessing import Pool

from .load import read_molecule, load_lm_embeddings_inf

class Dataset(PyGDataset):
    """ graph data class """
    def __init__(self, name, dirt, load_fn, process_fn, normalizer=None, split_time=False):
        super(Dataset, self).__init__()
        """ initialization
        Args:
            name: name of the dataset
            dirt: directory for storing the dataset
            load_fn: function for loading the dataset
            process_fn: function for processing the dataset
            normalizer: class for normalize data
            split_time: whether to split data according to time
        """
        self.name = name
        self.dirt = dirt
        self.load_fn = load_fn
        self.process_fn = process_fn
        self.normalizer = normalizer
        self.split_time = split_time

        # load data
        self.data_list = self.load_fn(self.name, self.dirt)

        # process data
        data_list = []
        with tqdm(total=len(self.data_list), desc='processing data') as pbar:
            for data in map(self.process_fn, self.data_list):
                if not self.split_time:
                    data_list.append(data)
                else:
                    data_list += data
                pbar.update()
        self.data_list = data_list

        # normalize data
        if self.normalizer is not None:
            if not hasattr(self.normalizer, 'mean'):
                self.normalizer.cal_mean_and_std(self.data_list)
            with tqdm(total=len(self.data_list), desc='normalizing data') as pbar:
                for data in data_list:
                    self.normalizer.normalize(data)
                    pbar.update()
            
    def len(self):
        """ return the number of graph data in the dataset """
        return len(self.data_list)
    
    def get(self, idx):
        """ retrieve an individual graph data based on the index """
        return self.data_list[idx]
'''
class DglDataset(DGLDataset):
    """
    Class to store and traverse a DGL dataset.

    Attributes:
        graphs: list of graphs in the dataset
        params: dictionary containing parameters of the problem
        times: array containing number of times for each graph in the dataset
        lightgraphs: list of graphs, without edge and node features
        data_name: n x 2 array (n is the total number of timesteps in the
                     dataset) mapping a graph index (first column) to the
                     timestep index (second column).

    """
    def __init__(self, name, dirt, split_fn=None, load_fn=None, normalize_fn=None, process_fn=None, noise_fn=None):
        """
        Init Dataset with list of graphs, dictionary of parameters, and list of
        graph names.

        Args:
            graphs: lift of graphs
            params: dictionary of parameters
            data_name: list of graph names
            index_map:
        """
        self.dirt = dirt
        self.load_fn = load_fn
        self.normalize_fn = normalize_fn
        self.process_fn = process_fn
        self.noise_fn = noise_fn
        
        self.times = []
        self.lightgraphs = []

        self.graphs = self.load_fn(self.dirt)
        self.data_list, self.bc_type, self.statistics = self.normalize_fn(self.graphs)

        graphs_name = [graph_name for graph_name in self.data_list]
        self.split_idx = split_fn(graphs_name)
        self.data_list = [self.data_list[idx] for idx in self.split_idx]
        
        super().__init__(name)
    
    def process(self):
        """ Process Dataset.
            Creates lightgraphs, the index map, and collects all times from the graphs.
        """
        self.lightgraphs, self.times, self.index_map = self.process_fn(self.data_list)
    
    def __getitem__(self, i):
        """ Get ith data. """    
        return self.noise_fn(self.data_list, self.lightgraphs, self.index_map, statistics=self.statistics, i=i)

    def __len__(self):
        """ Get length of the dataset. """
        return self.index_map.shape[0]

    def __str__(self):
        """ Get dataset name. """
        print('Total number of graphs: {:}'.format(self.__len__()))
        return 'Dataset = ' + ', '.join(self.split_idx)
'''
class PDBDataset(PyGDataset):
    def __init__(self, dirt, load_fn, process_fn, transform=None, cache_path: str='data_files/processed/cache',
                 split_path: str='data_files/processed/', esm_embd_path: str=None, complex_limit: int=0, worker_num: int=1,
                 lig_description=None, require_lig: bool=False, lig_size_max: int=None, popsize: int=15, maxiter: int=15,
                 matching: bool=True, keep_original: bool=False, conformer_num: int=1,
                 remove_hs: bool=False, keep_local_structure: bool=False,
                 rec_path_list=None, rec_cutoff_radius: float=30.0, ca_nb_num_max: int=None,
                 all_atom: bool=False, atom_cutoff_radius: float=5.0, atom_nb_num_max: int=None):
        super(PDBDataset, self).__init__(dirt, transform)
        """ PDBBind dataset
        Args:
            dirt: directory for storing dataset
            load_fn: function for loading the dataset
            process_fn: function for processing the dataset
            transform: noise transformation function
            cache_path: directory for storing cached dataset
            split_dirt: directory storing index for spliting dataset
            esm_embd_path: path for storing LM embeddings for the receptor features
            complex_limit: the number of training and validation complexes is capped
            worker_num: number of workers for preprocessing
            lig_description: description of ligand
            require_lig: whether to require ligand
            lig_size_max: maximum number of heavy atoms in ligand
            popsize: differential evolution popsize parameter in matching
            maxiter: differential evolution maxiter parameter in matching
            matching: whether to matching
            keep_original: whether to keep the original structure
            conformer_num: number of conformers to match to each ligand
            remove_hs: whether to remove Hs
            keep_local_structure: keeps the local structure when specifying an input with 3D coordinates instead of generating them with RDKit
            rec_path_list: list of path of receptor
            rec_cutoff_radius: cutoff on distances for receptor edges
            ca_nb_num_max: maximum number of neighbors for each residue
            all_atom: whether to use the all atoms model
            atom_cutoff_radius: cutoff on distances for atom connections
            atom_nb_num_max: maximum number of atom neighbours for receptor
        """
        self.dirt = dirt
        self.load_fn = load_fn
        self.process_fn = process_fn
        self.cache_path = cache_path
        self.split_path = split_path
        self.esm_embd_path = esm_embd_path
        self.complex_limit = complex_limit
        self.worker_num = worker_num

        self.lig_description = lig_description
        self.require_lig = require_lig
        self.lig_size_max = lig_size_max
        self.popsize = popsize
        self.maxiter = maxiter
        self.matching = matching
        self.keep_original = keep_original
        self.conformer_num = conformer_num
        self.remove_hs = remove_hs
        self.keep_local_structure = keep_local_structure

        self.rec_path_list = rec_path_list
        self.rec_cutoff_radius = rec_cutoff_radius
        self.ca_nb_num_max = ca_nb_num_max
        self.all_atom = all_atom
        self.atom_cutoff_radius = atom_cutoff_radius
        self.atom_nb_num_max = atom_nb_num_max

        if self.matching or self.rec_path_list is not None and self.lig_description is not None:
            self.cache_path += '_torsion'
        if self.all_atom:
            self.cache_path += '_allatoms'
        
        self.full_cache_path = os.path.join(self.cache_path, 
            f'limit{self.complex_limit}'
            f'_INDEX{os.path.splitext(os.path.basename(self.split_path))[0]}'
            f'_maxLigSize{self.lig_size_max}_H{int(not self.remove_hs)}'
            f'_recRad{self.rec_cutoff_radius}_recMax{self.ca_nb_num_max}'
            + ('' if not self.all_atom else f'_atomRad{self.atom_cutoff_radius}_atomMax{self.atom_nb_num_max}')
            + ('' if not self.matching or self.conformer_num == 1 else f'_confs{self.conformer_num}')
            + ('' if self.esm_embd_path is None else f'_esmEmbeddings')
            + ('' if not self.keep_local_structure else f'_keptLocalStruct')
            + ('' if self.rec_path_list is None or self.lig_description is None else str(
                binascii.crc32(''.join(self.lig_description + self.rec_path_list).encode()))))
        
        # Pre-process data
        if not os.path.exists(os.path.join(self.full_cache_path, "heterographs.pkl")) or (
            self.require_lig and not os.path.exists(os.path.join(self.full_cache_path, "rdkit_ligands.pkl"))):
            os.makedirs(self.full_cache_path, exist_ok=True)
            if self.rec_path_list is None or self.lig_description is None:
                self.pre_process()
            else:
                self.pre_process_inf()

        # Load processed data
        print('Loading data from memory: ', os.path.join(self.full_cache_path, "heterographs.pkl"))
        with open(os.path.join(self.full_cache_path, "heterographs.pkl"), 'rb') as f:
            self.complex_graphs = pickle.load(f)
        if self.require_lig:
            with open(os.path.join(self.full_cache_path, "rdkit_ligands.pkl"), 'rb') as f:
                self.rdkit_ligands = pickle.load(f)

        self.print_statistics(self.complex_graphs)

    def len(self):
        return len(self.complex_graphs)

    def get(self, idx):
        if self.require_lig:
            complex_graph = copy.deepcopy(self.complex_graphs[idx])
            complex_graph.mol = copy.deepcopy(self.rdkit_ligands[idx])
            return complex_graph
        else:
            return copy.deepcopy(self.complex_graphs[idx])

    def pre_process(self):
        print(f'Processing complexes from [{self.split_path}] and saving it to [{self.full_cache_path}]')
        
        complex_names, lm_embd_chains, rec_models, ligs = self.load_fn(
            self.dirt, self.split_path, self.esm_embd_path)

        complex_graphs, rdkit_ligands = [], []
        if self.worker_num > 1:
            p = Pool(self.worker_num, maxtasksperchild=1)
            p.__enter__()
        with tqdm(total=len(complex_names), desc='process complexes') as pbar:
            map_fn = p.imap_unordered if self.worker_num > 1 else map
            cl = len(complex_names)
            for t in map_fn(self.process_fn, [self.dirt]*cl, complex_names, 
                rec_models, lm_embd_chains, ligs, [self.lig_size_max]*cl, [self.popsize]*cl,
                [self.maxiter]*cl, [self.matching]*cl, [self.keep_original]*cl,
                [self.conformer_num]*cl, [self.remove_hs]*cl, [self.rec_cutoff_radius]*cl,
                [self.ca_nb_num_max]*cl, [self.all_atom]*cl, [self.atom_cutoff_radius]*cl,
                [self.atom_nb_num_max]*cl):
                complex_graphs.extend(t[0])
                rdkit_ligands.extend(t[1])
                pbar.update()
        if self.worker_num > 1: p.__exit__(None, None, None)

        with open(os.path.join(self.full_cache_path, "heterographs.pkl"), 'wb') as f:
            pickle.dump((complex_graphs), f)
        with open(os.path.join(self.full_cache_path, "rdkit_ligands.pkl"), 'wb') as f:
            pickle.dump((rdkit_ligands), f)

    def pre_process_inf(self):
        ligands_list = []
        print('Reading molecules and generating local structures with RDKit')
        for ligand_description in tqdm(self.lig_description):
            mol = MolFromSmiles(ligand_description)  # check if it is a smiles or a path
            if mol is not None:
                mol = AddHs(mol)
                
                ps = AllChem.ETKDGv2()
                id = AllChem.EmbedMolecule(mol, ps)
                if id == -1:
                    print('rdkit coords could not be generated without using random coords. using random coords now.')
                    ps.useRandomCoords = True
                    AllChem.EmbedMolecule(mol, ps)
                    AllChem.MMFFOptimizeMolecule(mol, confId=0)

                ligands_list.append(mol)
            else:
                mol = read_molecule(ligand_description, remove_hs=False, sanitize=True)
                if not self.keep_local_structure:
                    mol.RemoveAllConformers()
                    mol = AddHs(mol)

                    ps = AllChem.ETKDGv2()
                    id = AllChem.EmbedMolecule(mol, ps)
                    if id == -1:
                        print('rdkit coords could not be generated without using random coords. using random coords now.')
                        ps.useRandomCoords = True
                        AllChem.EmbedMolecule(mol, ps)
                        AllChem.MMFFOptimizeMolecule(mol, confId=0)

                ligands_list.append(mol)
        
        lm_embeddings_chains_all, rec_models, ligs = load_lm_embeddings_inf(
            self.dirt, self.esm_embd_path, self.rec_path_list, ligands_list, self.lig_description)
    
        print('Generating graphs for ligands and proteins')
        complex_graphs, rdkit_ligands = [], []
        if self.worker_num > 1:
            p = Pool(self.worker_num, maxtasksperchild=1)
            p.__enter__()

        with tqdm(total=len(self.rec_path_list), desc='process complexes') as pbar:
            map_fn = p.imap_unordered if self.worker_num > 1 else map
            cl = len(self.rec_path_list)
            for t in map_fn(self.process_fn, [self.dirt]*cl, self.rec_path_list, rec_models, 
                            lm_embeddings_chains_all, ligs,
                            [self.lig_size_max]*cl, [self.popsize]*cl, [self.maxiter]*cl,
                            [self.matching]*cl, [self.keep_original]*cl,
                            [self.conformer_num]*cl, [self.remove_hs]*cl, [self.rec_cutoff_radius]*cl, 
                            [self.ca_nb_num_max]*cl, [self.all_atom]*cl,
                            [self.atom_cutoff_radius]*cl, [self.atom_nb_num_max]*cl):
                complex_graphs.extend(t[0])
                rdkit_ligands.extend(t[1])
                pbar.update()
        if self.worker_num > 1: p.__exit__(None, None, None)

        with open(os.path.join(self.full_cache_path, "heterographs.pkl"), 'wb') as f:
            pickle.dump((complex_graphs), f)
        with open(os.path.join(self.full_cache_path, "rdkit_ligands.pkl"), 'wb') as f:
            pickle.dump((rdkit_ligands), f)

    def print_statistics(self, complex_graphs):
        statistics = ([], [], [], [])

        for complex_graph in complex_graphs:
            lig_pos = complex_graph['ligand'].pos if torch.is_tensor(complex_graph['ligand'].pos) else complex_graph['ligand'].pos[0]
            radius_protein = torch.max(torch.linalg.vector_norm(complex_graph['receptor'].pos, dim=1))
            molecule_center = torch.mean(lig_pos, dim=0)
            radius_molecule = torch.max(
                torch.linalg.vector_norm(lig_pos - molecule_center.unsqueeze(0), dim=1))
            distance_center = torch.linalg.vector_norm(molecule_center)
            statistics[0].append(radius_protein)
            statistics[1].append(radius_molecule)
            statistics[2].append(distance_center)
            if "rmsd_matching" in complex_graph:
                statistics[3].append(complex_graph.rmsd_matching)
            else:
                statistics[3].append(0)

        name = ['radius protein', 'radius molecule', 'distance protein-mol', 'rmsd matching']
        print('Number of complexes: ', len(complex_graphs))
        for i in range(4):
            array = np.asarray(statistics[i])
            print(f"{name[i]}: mean {np.mean(array)}, std {np.std(array)}, max {np.max(array)}")

class DataLoader(PyGDataLoader):
    """ dataLoader for loading graph dataset """
    def __init__(self, dataset, batch_size: int=1, shuffle: bool=True):
        super(DataLoader, self).__init__(dataset, batch_size, shuffle)
        """ initialization
        Args:
            dataset: dataset class
            batch_size: size of batch data
            shuffle: whether to shuffle the data
        """
'''
class DglDataLoader(GraphDataLoader):
    def __init__(self, dataset, sampler=None, batch_size: int=1, drop_last: bool=False):
        if sampler==None:
            sampler = SubsetRandomSampler(torch.arange(int(len(dataset))))
        super(DglDataLoader, self).__init__(
            dataset, sampler=sampler, batch_size=batch_size, drop_last=drop_last)
        """ DataLoader class for loading graph datasets.
        Args:
            dataset: dataset class
            batch_size: size of batch data
            shuffle: whether to shuffle the data
        """
'''