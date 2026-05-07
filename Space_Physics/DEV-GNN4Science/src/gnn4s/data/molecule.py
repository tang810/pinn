import copy
import re
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem, RemoveHs, rdMolTransforms
from scipy.optimize import differential_evolution

class ChemMol:
    def __init__(self, x, adj, atomic_num_list, largest_connected_comp=True):
        self.x = x.detach().cpu().numpy()
        self.adj = adj.detach().cpu().numpy()
        self.atomic_num_list = atomic_num_list
        self.largest_connected_comp = largest_connected_comp
        
        self.atom_valency = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}
        self.bond_decoder = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE}
        self.gen_mol()
    
    def gen_mol(self):
        self.mol, self.no_correct_num = [], 0
        for x_elem, adj_elem in zip(self.x, self.adj):
            mol = self.construct_mol(x_elem, adj_elem, self.atomic_num_list)
            cmol, no_correct = self.correct_mol(mol)
            if no_correct:
                self.no_correct_num += 1
            vcmol = self.valid_mol_can_with_seg(cmol, self.largest_connected_comp)
            self.mol.append(vcmol)
        self.mol = [mol for mol in self.mol if mol is not None]
        self.mol_num = len(self.mol)
    
    def check_valency(self, mol):
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            return True, None
        except ValueError as e:
            e = str(e)
            p = e.find('#')
            e_sub = e[p:]
            atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
            return False, atomid_valence
        
    def construct_mol(self, x, adj, atomic_num_list): # x: 9, 5; adj: 4, 9, 9
        mol = Chem.RWMol()

        atoms = np.argmax(x, axis=1)
        atoms_exist = (atoms != len(atomic_num_list) - 1)
        atoms = atoms[atoms_exist]              # 9,
        for atom in atoms:
            mol.AddAtom(Chem.Atom(int(atomic_num_list[atom])))

        adj = np.argmax(adj, axis=0)            # 9, 9
        adj = adj[atoms_exist, :][:, atoms_exist]
        adj[adj == 3] = -1
        adj += 1                                # bonds 0, 1, 2, 3 -> 1, 2, 3, 0 (0 denotes the virtual bond)

        for start, end in zip(*np.nonzero(adj)):
            if start > end:
                mol.AddBond(int(start), int(end), self.bond_decoder[adj[start, end]])
                # add formal charge to atom: e.g. [O+], [N+], [S+]
                # not support [O-], [N-], [S-], [NH+] etc.
                flag, atomid_valence = self.check_valency(mol)
                if flag:
                    continue
                else:
                    assert len(atomid_valence) == 2
                    idx = atomid_valence[0]
                    v = atomid_valence[1]
                    an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                    if an in (7, 8, 16) and (v - self.atom_valency[an]) == 1:
                        mol.GetAtomWithIdx(idx).SetFormalCharge(1)
        return mol
    
    def correct_mol(self, m):
        mol = m
        no_correct = False
        flag, _ = self.check_valency(mol)
        if flag:
            no_correct = True

        while True:
            flag, atomid_valence = self.check_valency(mol)
            if flag:
                break
            else:
                assert len(atomid_valence) == 2
                idx = atomid_valence[0]
                v = atomid_valence[1]
                queue = []
                for b in mol.GetAtomWithIdx(idx).GetBonds():
                    queue.append((b.GetIdx(), int(b.GetBondType()), b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
                queue.sort(key=lambda tup: tup[1], reverse=True)
                if len(queue) > 0:
                    start = queue[0][2]
                    end = queue[0][3]
                    t = queue[0][1] - 1
                    mol.RemoveBond(start, end)
                    if t >= 1:
                        mol.AddBond(start, end, self.bond_decoder[t])
        return mol, no_correct
    
    def valid_mol_can_with_seg(self, m, largest_connected_comp=True):
        if m is None:
            return None
        sm = Chem.MolToSmiles(m, isomericSmiles=True)
        if largest_connected_comp and '.' in sm:
            vsm = [(s, len(s)) for s in sm.split('.')]  # 'C.CC.CCc1ccc(N)cc1CCC=O'.split('.')
            vsm.sort(key=lambda tup: tup[1], reverse=True)
            mol = Chem.MolFromSmiles(vsm[0][0])
        else:
            mol = Chem.MolFromSmiles(sm)
        return mol

class Mol:
    def __init__(self, mol):
        self.mol = mol
        self.bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
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
            'possible_is_in_ring8_list': [False, True]
        }

    def get_mol(self, remove_hs=False):
        mol = copy.deepcopy(self.mol)
        if remove_hs:
            mol = RemoveHs(mol, sanitize=True)
        return mol
    
    def get_original_pos(self, mol=None):
        if mol==None:
            mol = self.mol
        return mol.GetConformer().GetPositions()
        
    def get_torsion_angles(self, mol=None):
        if mol==None:
            mol = self.mol
        
        g = nx.Graph()
        for i, atom in enumerate(mol.GetAtoms()):
            g.add_node(i)
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            g.add_edge(start, end)
        
        torsions_list = []
        for e in g.edges():
            g2 = copy.deepcopy(g)
            g2.remove_edge(*e)
            if nx.is_connected(g2): continue
            l = list(sorted(nx.connected_components(g2), key=len)[0])
            if len(l) < 2: continue
            n0 = list(g2.neighbors(e[0]))
            n1 = list(g2.neighbors(e[1]))
            torsions_list.append((n0[0], e[0], e[1], n1[0]))
        return torsions_list
    
    def generate_conformer(self, mol=None, remove_hs=False):
        """ Generate the conformation of the molecule
        """
        if mol is None:
            mol = self.mol
        
        mol_rdkit = copy.deepcopy(mol)
        mol_rdkit.RemoveAllConformers()
        mol_rdkit = AllChem.AddHs(mol_rdkit)
        
        # Calculate the electrostatic potential of a molecule
        ps = AllChem.ETKDGv2()
        # Generate stable and reasonable molecular conformations
        id = AllChem.EmbedMolecule(mol_rdkit, ps)
        if id==-1:
            print('rdkit coords could not be generated without using random coords. using random coords now.')
            ps.useRandomCoords = True
            AllChem.EmbedMolecule(mol_rdkit, ps)
            # Optimize molecular structure using MMFF94 force field
            AllChem.MMFFOptimizeMolecule(mol_rdkit, confId=0)
        # else:
        #    AllChem.MMFFOptimizeMolecule(mol_rdkit, confId=0)

        if remove_hs:
            mol_rdkit = RemoveHs(mol_rdkit, sanitize=True)
        return mol_rdkit

    def set_dihedral(self, conf, atom_idx, new_value):
        rdMolTransforms.SetDihedralRad(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3], new_value)
    
    def apply_changes(self, mol, values, rotable_bonds, conf_id):
        opt_mol = copy.copy(mol)
        [self.set_dihedral(opt_mol.GetConformer(conf_id), rotable_bonds[r], values[r])
            for r in range(len(rotable_bonds))]
        return opt_mol
    
    def optimize_rotatable_bonds(self, mol, true_mol, rotable_bonds, probe_id=-1, ref_id=-1, seed=0,
                                 popsize=15, maxiter=500, mutation=(0.5,1.0), recombination=0.8):
        opt = OptimizeConformer(mol, true_mol, rotable_bonds, seed=seed, probe_id=probe_id, ref_id=ref_id)
        max_bound = [np.pi] * len(opt.rotable_bonds)
        min_bound = [-np.pi] * len(opt.rotable_bonds)
        bounds = (min_bound, max_bound)
        bounds = list(zip(bounds[0], bounds[1]))
        
        # Optimize conformations
        result = differential_evolution(opt.score_conformation, bounds,
                                        maxiter=maxiter, popsize=popsize,
                                        mutation=mutation, recombination=recombination, disp=False, seed=seed)
        opt_mol = self.apply_changes(opt.mol, result['x'], opt.rotable_bonds, conf_id=probe_id)

        return opt_mol
    
    def safe_index(self, list, elem):
        """ Return index of element in list.
            If elem is not present, return the last index
        """
        try:
            return list.index(elem)
        except:
            return len(list) - 1
    
    def get_atom_featurizer(self, mol=None):
        if mol is None:
            mol = self.mol
        
        ringinfo = mol.GetRingInfo()
        atom_features_list = []
        for idx, atom in enumerate(mol.GetAtoms()):
            atom_features_list.append([
                self.safe_index(self.allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
                self.allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag())),
                self.safe_index(self.allowable_features['possible_degree_list'], atom.GetTotalDegree()),
                self.safe_index(self.allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
                self.safe_index(self.allowable_features['possible_implicit_valence_list'], atom.GetImplicitValence()),
                self.safe_index(self.allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
                self.safe_index(self.allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
                self.safe_index(self.allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
                self.allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
                self.safe_index(self.allowable_features['possible_numring_list'], ringinfo.NumAtomRings(idx)),
                self.allowable_features['possible_is_in_ring3_list'].index(ringinfo.IsAtomInRingOfSize(idx, 3)),
                self.allowable_features['possible_is_in_ring4_list'].index(ringinfo.IsAtomInRingOfSize(idx, 4)),
                self.allowable_features['possible_is_in_ring5_list'].index(ringinfo.IsAtomInRingOfSize(idx, 5)),
                self.allowable_features['possible_is_in_ring6_list'].index(ringinfo.IsAtomInRingOfSize(idx, 6)),
                self.allowable_features['possible_is_in_ring7_list'].index(ringinfo.IsAtomInRingOfSize(idx, 7)),
                self.allowable_features['possible_is_in_ring8_list'].index(ringinfo.IsAtomInRingOfSize(idx, 8)),
            ])

        return torch.tensor(atom_features_list)
    
    def get_graph(self, mol=None):
        if mol is None:
            mol = self.mol
        
        atom_cord = torch.from_numpy(mol.GetConformer().GetPositions()).float()
        atom_feat = self.get_atom_featurizer(mol)
        
        row, col, edge_type = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_type += 2 * [self.bonds[bond.GetBondType()]] if bond.GetBondType() != BT.UNSPECIFIED else [0, 0]

        edge_idx = torch.tensor([row, col], dtype=torch.long)
        edge_type = torch.tensor(edge_type, dtype=torch.long)
        edge_attr = F.one_hot(edge_type, num_classes=len(self.bonds)).to(torch.float)
        
        return atom_feat, atom_cord, edge_idx, edge_attr

    def get_lig_graph(self, lig, matching, remove_hs, keep_original, conformers_num, popsize, maxiter):
    
        orig_pos = None
        if matching:
            mol_maybe_noh = self.get_mol(remove_hs=remove_hs)
            if keep_original:
                orig_pos = self.get_original_pos(mol_maybe_noh)
            
            rotable_bonds = self.get_torsion_angles(mol_maybe_noh)
            if not rotable_bonds: print("no_rotable_bonds but still using it")

            for i in range(conformers_num):
                # Generate a molecular conformation mol_rdkit
                mol_rdkit = self.generate_conformer(remove_hs=remove_hs)

                # twist the twistable bond
                mol = copy.deepcopy(mol_maybe_noh)
                if rotable_bonds:
                    self.optimize_rotatable_bonds(mol_rdkit, mol, rotable_bonds, popsize=popsize, maxiter=maxiter)
                mol.AddConformer(mol_rdkit.GetConformer())
                
                # Adding the molecular conformation after torsion
                rms_list = []
                # Align molecules, record root mean square
                AllChem.AlignMolConformers(mol, RMSlist=rms_list)
                mol_rdkit.RemoveAllConformers()
                # Take the conformation closest to the mol
                mol_rdkit.AddConformer(mol.GetConformers()[1])

                if i==0:
                    rmsd_matching = rms_list[0]
                    x, pos, edge_index, edge_attr = self.get_graph(mol_rdkit)
                else:
                    if torch.is_tensor(pos):
                        pos = [pos]
                    pos.append(torch.from_numpy(mol_rdkit.GetConformer().GetPositions()).float())
        else:
            rmsd_matching = 0
            if remove_hs: lig = RemoveHs(lig)
            x, pos, edge_index, edge_attr = self.get_graph(lig)

        node_nx_num = max(max(edge_index[0]),max(edge_index[1])) + 1
        node_nx = [i for i in range(node_nx_num)]
        edge_nx = [(int(edge_index[0,i]),int(edge_index[1,i])) for i in range(len(edge_index[0]))]

        edge_mask, mask_rotate = self.get_transformation_mask(node_nx, edge_nx)
        edge_mask = torch.tensor(edge_mask)

        return rmsd_matching, orig_pos, x, pos, edge_index, edge_attr, edge_mask, mask_rotate

    def get_transformation_mask(self, nodes_nx, edges_nx):
        g = nx.Graph()
        g.add_nodes_from(nodes_nx)
        g.add_edges_from(edges_nx)
        g = g.to_directed()

        to_rotate = []
        edges = np.array([[edges_nx[i][0],edges_nx[i][1]] for i in range(len(edges_nx))])
        
        for i in range(0, edges.shape[0], 2):
            assert edges[i, 0] == edges[i+1, 1]

            g2 = g.to_undirected()
            g2.remove_edge(*edges[i])
            if not nx.is_connected(g2):
                l = list(sorted(nx.connected_components(g2), key=len)[0])
                if len(l) > 1:
                    if edges[i, 0] in l:
                        to_rotate.append([])
                        to_rotate.append(l)
                    else:
                        to_rotate.append(l)
                        to_rotate.append([])
                    continue
            to_rotate.append([])
            to_rotate.append([])

        mask_edges = np.asarray([0 if len(l) == 0 else 1 for l in to_rotate], dtype=bool)
        mask_rotate = np.zeros((np.sum(mask_edges), len(g.nodes())), dtype=bool)
        idx = 0
        for i in range(len(g.edges())):
            if mask_edges[i]:
                mask_rotate[idx][np.asarray(to_rotate[i], dtype=int)] = True
                idx += 1

        return mask_edges, mask_rotate

class OptimizeConformer:
    def __init__(self, mol, true_mol, rotable_bonds, probe_id=-1, ref_id=-1, seed=None):
        super(OptimizeConformer, self).__init__()
        if seed:
            np.random.seed(seed)
        self.rotable_bonds = rotable_bonds
        self.mol = mol
        self.true_mol = true_mol
        self.probe_id = probe_id
        self.ref_id = ref_id

    def set_dihedral(self, conf, atom_idx, new_vale):
        rdMolTransforms.SetDihedralRad(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3], new_vale)

    def score_conformation(self, values):
        for i, r in enumerate(self.rotable_bonds):
            self.set_dihedral(self.mol.GetConformer(self.probe_id), r, values[i])
        return AllChem.AlignMol(self.mol, self.true_mol, self.probe_id, self.ref_id)
