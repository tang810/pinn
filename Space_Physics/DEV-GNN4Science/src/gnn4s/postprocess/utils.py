import os
import copy
import numpy as np
from rdkit.Chem import RemoveHs

import gnn4s

def sort_confidence(confidence, rmsd_classification_cutoff):
    if isinstance(rmsd_classification_cutoff, list):
        confidence = confidence[:,0]
    confidence = confidence.cpu().numpy()
    re_order = np.argsort(confidence)[::-1]
    confidence = confidence[re_order]
    return confidence, re_order

def cal_min_self_distances(pos):
    self_distances = np.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=-1)
    self_distances = np.where(np.eye(self_distances.shape[2]), np.inf, self_distances)
    return np.min(self_distances, axis=(1, 2))

def write_mol_with_cord(lig, lig_pos, remove_hs, confidence, dirt, idx, name, lig_pos_process=None):
    write_dir = f'{dirt}/{name.replace("/","-")}'
    os.makedirs(write_dir, exist_ok=True)
    for rank, pos in enumerate(lig_pos):
        mol = copy.deepcopy(lig)
        if remove_hs:
            mol = RemoveHs(mol)
        if rank==0: 
            gnn4s.data.write_mol_with_coords(mol, pos, f'{write_dir}/rank{rank+1}.sdf')
            if lig_pos_process is not None:
                for i in range(len(lig_pos_process)):
                    gnn4s.data.write_mol_with_coords(mol, lig_pos_process[i][rank], f'{write_dir}/rank{rank+1}_step{i}.sdf')
        
        gnn4s.data.write_mol_with_coords(mol, pos, f'{write_dir}/rank{rank+1}_confidence{confidence[rank]:.2f}.sdf')