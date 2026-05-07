import os
import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

def save_smiles(smiles, dir, name):
    if not(os.path.isdir(dir)):
        os.makedirs(dir)
    
    smiles = [smi for smi in smiles if len(smi)]
    path = f'{dir}/{name}.txt'
    with open(path, 'a') as f:
        for smi in smiles:
            f.write(f'{smi}\n')

def write_mol_with_coords(mol, new_coords, path):
    w = Chem.SDWriter(path)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        x,y,z = new_coords.astype(np.double)[i]
        conf.SetAtomPosition(i,Point3D(x,y,z))
    w.write(mol)
    w.close()
