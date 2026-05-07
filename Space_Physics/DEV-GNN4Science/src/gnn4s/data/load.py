import os
import torch
import numpy as np
import h5py
import pickle as pk
import pandas as pd
import random
import glob
import json
import warnings
from tqdm import tqdm
# from dgl.data.utils import load_graphs as lg
from collections import defaultdict
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from rdkit import Chem
from rdkit.Chem import AllChem

def load_h5(name: str, dirt: str, keys: list):
    """ load .h5 format dataset 
    Args:
        name: name of dataset
        dirt: directory storing dataset
        keys: keys of data to be loaded
    Returns:
        data_list: list of dataset
    """
    with h5py.File(f'{dirt}/{name}.h5', 'r') as f:
        data_list = []
        for key in f:
            data = {}
            for k in keys:
                data[k] = f[key][k][:]
            data_list.append(data)
    return data_list

def load_h5_from_name_list(name: str, dirt: str):
    """ load .h5 format dataset 
    Args:
        name: name list of dataset
        dirt: directory storing dataset
    Returns:
        data_list: list of dataset
    """
    name_list = []
    with open(f'{dirt}/{name}_name_list.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            name_list.append(f'{line.strip()}')

    data_list = []
    for data_name in name_list:
        with h5py.File(f'{dirt}/{data_name}.h5', 'r') as f:
            data = {}
            data['name'] = data_name
            for key in f:
                data[key] = np.array(f[key])
            data_list.append(data)
    
    return data_list

def load_graphs(input_dir):
    """
    Load all graphs in directory.

    Arguments:
        input_dir (string): input directory path

    Returns:
        list of DGL graphs

    """
    files = os.listdir(input_dir)
    random.shuffle(files)

    graphs = {}
    for file in tqdm(files, desc = 'Loading graphs', colour='green'):
        if 'grph' in file:
            graphs[file] = lg(input_dir + '/' + file)[0][0]

    return graphs

def load_smiles(name: str, dirt: str):
    col = 'SMILES1'
    path = f'{dirt}/{name}.csv'
    data = pd.read_csv(path)
    return data[col]

def load_mol(name: str, dirt: str):
    path = f'{dirt}/{name}.npz'

    print(f'Loading file {path}')
    if not os.path.exists(path):
        raise ValueError(f'Invalid filepath {path} for dataset')
    load_data = np.load(path)
    result = []
    i = 0
    while True:
        key = f'arr_{i}'
        if key in load_data.keys():
            result.append(load_data[key])
            i += 1
        else:
            break
    return list(map(lambda x, a: (x, a), result[0], result[1]))

def load_idx(name: str, dirt: str, data_len: int):
    path = f'{dirt}/valid_idx_{name}.json'
    with open(path) as f:
        test_idx = json.load(f)
    test_idx = test_idx['valid_idxs']
    test_idx = [int(i) for i in test_idx]
    train_idx = [i for i in range(data_len) if i not in test_idx]
    return train_idx, test_idx

def load_lm_embeddings(root, split_path, esm_embeddings_path, limit_complexes=None):
    with open(split_path) as f:
        lines = f.readlines()
        complex_names = [line.rstrip() for line in lines]
    if limit_complexes is not None and limit_complexes != 0:
        complex_names = complex_names[:limit_complexes]
    print(f'Loading {len(complex_names)} complexes.')
    
    if esm_embeddings_path is not None:
        id_to_embeddings = torch.load(esm_embeddings_path)
        chain_embeddings_dictlist = defaultdict(list)
        for key, embedding in id_to_embeddings.items():
            key_name = key.split('_')[0]
            if key_name in complex_names:
                chain_embeddings_dictlist[key_name].append(embedding)
        lm_embeddings_chains = []
        for name in complex_names:
            lm_embeddings_chains.append(chain_embeddings_dictlist[name])
    else:
        lm_embeddings_chains = [None] * len(complex_names)
    
    rec_models, ligs = [], []
    for i in range(len(complex_names)):
        name = complex_names[i]
        try:
            rec_path = os.path.join(root, name, f'{name}_protein_processed.pdb')
            rec_models.append(parse_pdb_from_path(rec_path))
        except Exception as e:
            rec_models.append(None)
            print(f'Skipping {name} because of the error:')
            print(e)
            
        try:
            ligs.append(read_mols(root, name, remove_hs=False))
        except Exception as e:
            ligs.append(None)
            print(f'Skipping {name} because of the error:')
            print(e)
    
    return complex_names, lm_embeddings_chains, rec_models, ligs

def parse_receptor(pdbid, pdbbind_dir):
    rec_path = os.path.join(pdbbind_dir, pdbid, f'{pdbid}_protein_processed.pdb')
    return parse_pdb_from_path(rec_path)

def load_lm_embeddings_inf(root, esm_embeddings_path, protein_path_list, ligands_list, ligand_descriptions):
    if esm_embeddings_path is not None:
        print('Reading language model embeddings.')
        lm_embeddings_chains_all = []
        if not os.path.exists(esm_embeddings_path): 
            raise Exception('ESM embeddings path does not exist: ', esm_embeddings_path)
        for protein_path in protein_path_list:
            embeddings_paths = sorted(glob.glob(os.path.join(esm_embeddings_path, os.path.basename(protein_path)) + '*'))
            lm_embeddings_chains = []
            for embeddings_path in embeddings_paths:
                lm_embeddings_chains.append(torch.load(embeddings_path)['representations'][33])
            lm_embeddings_chains_all.append(lm_embeddings_chains)
    else:
        lm_embeddings_chains_all = [None] * len(protein_path_list)
    
    rec_models, lig_models = [], []
    for i in range(len(protein_path_list)):
        name = protein_path_list[i]
        ligs = ligands_list[i]
        ligand_description = ligand_descriptions[i]
        if ligs is not None:
            rec_model = parse_pdb_from_path(name)
            name = f'{name}____{ligand_description}'
            ligs = [ligs]
            rec_models.append(rec_model)
            lig_models.append(ligs)
        else:
            try:
                rec_model = parse_receptor(name, root)
            except Exception as e:
                print(f'Skipping {name} because of the error:')
                print(e)
                rec_model = None

            ligs = read_mols(root, name, remove_hs=False)
            rec_models.append(rec_model)
            lig_models.append(ligs)

    return lm_embeddings_chains_all, rec_models, lig_models

def load_rec_lig_csv(file_path):
    rec_lig = pd.read_csv(file_path)
    rec_path_list = rec_lig['protein_path'].tolist()
    lig_description = rec_lig['ligand'].tolist()
    return rec_path_list, lig_description

def parse_pdb_from_path(path):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        structure = PDBParser().get_structure('random_id', path)
        rec = structure[0]
    return rec

def read_mols(pdbbind_dir, name, remove_hs=False):
    ligs = []
    for file in os.listdir(os.path.join(pdbbind_dir, name)):
        if file.endswith(".sdf") and 'rdkit' not in file:
            lig = read_molecule(os.path.join(pdbbind_dir, name, file), remove_hs=remove_hs, sanitize=True)
            if lig is None and os.path.exists(os.path.join(pdbbind_dir, name, file[:-4] + ".mol2")):  
                # read mol2 file if sdf file cannot be sanitized
                print('Using the .sdf file failed. We found a .mol2 file instead and are trying to use that.')
                lig = read_molecule(os.path.join(pdbbind_dir, name, file[:-4] + ".mol2"),
                                    remove_hs=remove_hs, sanitize=True)
            if lig is not None:
                ligs.append(lig)
    return ligs

def read_molecule(molecule_file, sanitize=False, calc_charges=False, remove_hs=False):
    if molecule_file.endswith('.mol2'):
        mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.sdf'):
        supplier = Chem.SDMolSupplier(molecule_file, sanitize=False, removeHs=False)
        mol = supplier[0]
    elif molecule_file.endswith('.pdbqt'):
        with open(molecule_file) as file:
            pdbqt_data = file.readlines()
        pdb_block = ''
        for line in pdbqt_data:
            pdb_block += '{}\n'.format(line[:66])
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(molecule_file, sanitize=False, removeHs=False)
    else:
        raise ValueError('Expect the format of the molecule_file to be '
                         'one of .mol2, .sdf, .pdbqt and .pdb, got {}'.format(molecule_file))
    
    try:
        if sanitize or calc_charges:
            Chem.SanitizeMol(mol)
        if calc_charges:
            # Compute Gasteiger charges on the molecule.
            try:
                AllChem.ComputeGasteigerCharges(mol)
            except:
                warnings.warn('Unable to compute charges for the molecule.')
        if remove_hs:
            mol = Chem.RemoveHs(mol, sanitize=sanitize)
    except Exception as e:
        print(e)
        print("RDKit was unable to read the molecule.")
        return None

    return mol