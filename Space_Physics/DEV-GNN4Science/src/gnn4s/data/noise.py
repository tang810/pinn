import torch
import numpy as np
import math
import copy
import random
from torch_geometric.transforms import BaseTransform
from scipy.spatial.transform import Rotation as R

from .so3 import sample_vec, score_vec
from .torus import score
# from .normalization import invert_normalize

class TimeToSigma():
    def __init__(self, tr_sigma_min: float, tr_sigma_max: float, 
                 rot_sigma_min: float, rot_sigma_max: float, 
                 tor_sigma_min: float, tor_sigma_max: float) -> None:
        """ Standard deviation of the prior distribution. 
        Args:
            tr_sigma_min: minimum value of the standard deviation related to translation.
            tr_sigma_max: maximum value of the standard deviation related to translation.
            rot_sigma_min: minimum value of the standard deviation related to rotation.
            rot_sigma_max: maximum value of the standard deviation related to rotation.
            tor_sigma_min: minimum value of the standard deviation related to torsion.
            tor_sigma_max: maximum value of the standard deviation related to torsion.
        """
        self.tr_sigma_min = tr_sigma_min
        self.tr_sigma_max = tr_sigma_max
        self.rot_sigma_min = rot_sigma_min
        self.rot_sigma_max = rot_sigma_max
        self.tor_sigma_min = tor_sigma_min
        self.tor_sigma_max = tor_sigma_max

    def __call__(self, t_tr, t_rot, t_tor):
        """ Calculate the standard deviation of the prior distribution. 
        Args:
            t_tr, t_rot, t_tor: time
        """
        tr_sigma = self.tr_sigma_min ** (1-t_tr) * self.tr_sigma_max ** t_tr
        rot_sigma = self.rot_sigma_min ** (1-t_rot) * self.rot_sigma_max ** t_rot
        tor_sigma = self.tor_sigma_min ** (1-t_tor) * self.tor_sigma_max ** t_tor
        return tr_sigma, rot_sigma, tor_sigma

class NoiseTransform(BaseTransform):
    def __init__(self, t_to_sigma, no_torsion, all_atom):
        """ Add noise to data.
        Args:
            t_to_sigma: function for calculating the standard deviation of the prior distribution. 
            no_torsion: whether to consider torsion
            all_atom: whether to consider all atom
        """
        self.t_to_sigma = t_to_sigma
        self.no_torsion = no_torsion
        self.all_atom = all_atom

    def __call__(self, data):
        """ Add noise to data.
        Args:
            data: heterogeneous graph with ligand information
        """
        t = np.random.uniform()
        t_tr, t_rot, t_tor = t, t, t
        return self.apply_noise(data, t_tr, t_rot, t_tor)

    def apply_noise(self, data, t_tr, t_rot, t_tor, tr_update=None, rot_update=None, torsion_update=None):
        """ Add noise to data.
        Args:
            data: heterogeneous graph with ligand information
            t_tr, t_rot, t_tor: time
            tr_update: update of the translation
            rot_update: update of the rotation angle
            torsion_update: update of the torsion angle
        """
        if not torch.is_tensor(data['ligand'].pos):
            data['ligand'].pos = random.choice(data['ligand'].pos)

        # Calculate the standard deviation of the prior distribution.
        tr_sigma, rot_sigma, tor_sigma = self.t_to_sigma(t_tr, t_rot, t_tor)
        self.set_time(data, t_tr, t_rot, t_tor, 1, self.all_atom, device=None)

        # Sampling update magnitude from the distribution.
        tr_update = torch.normal(mean=0, std=tr_sigma, size=(1, 3)) if tr_update is None else tr_update
        rot_update = sample_vec(eps=rot_sigma) if rot_update is None else rot_update
        torsion_update = np.random.normal(loc=0.0, scale=tor_sigma, size=data['ligand'].edge_mask.sum()
                                           ) if torsion_update is None else torsion_update
        torsion_update = None if self.no_torsion else torsion_update
        
        # Modify the conformation of the ligand.
        self.modify_conformer(data, tr_update, torch.from_numpy(rot_update).float(), torsion_update)

        # Calculate score.
        data.tr_score = -tr_update / tr_sigma ** 2
        data.rot_score = torch.from_numpy(score_vec(vec=rot_update, eps=rot_sigma)).float().unsqueeze(0)
        data.tor_score = None if self.no_torsion else torch.from_numpy(score(torsion_update, tor_sigma)).float()
        data.tor_sigma_edge = None if self.no_torsion else np.ones(data['ligand'].edge_mask.sum()) * tor_sigma
        return data

    def set_time(self, complex_graphs, t_tr, t_rot, t_tor, batchsize, all_atoms, device):
        complex_graphs['ligand'].node_t = {
            'tr': t_tr * torch.ones(complex_graphs['ligand'].num_nodes).to(device),
            'rot': t_rot * torch.ones(complex_graphs['ligand'].num_nodes).to(device),
            'tor': t_tor * torch.ones(complex_graphs['ligand'].num_nodes).to(device)}
        complex_graphs['receptor'].node_t = {
            'tr': t_tr * torch.ones(complex_graphs['receptor'].num_nodes).to(device),
            'rot': t_rot * torch.ones(complex_graphs['receptor'].num_nodes).to(device),
            'tor': t_tor * torch.ones(complex_graphs['receptor'].num_nodes).to(device)}
        complex_graphs.complex_t = {'tr': t_tr * torch.ones(batchsize).to(device),
                                    'rot': t_rot * torch.ones(batchsize).to(device),
                                    'tor': t_tor * torch.ones(batchsize).to(device)}
        if all_atoms:
            complex_graphs['atom'].node_t = {
                'tr': t_tr * torch.ones(complex_graphs['atom'].num_nodes).to(device),
                'rot': t_rot * torch.ones(complex_graphs['atom'].num_nodes).to(device),
                'tor': t_tor * torch.ones(complex_graphs['atom'].num_nodes).to(device)}

    def modify_conformer(self, data, tr_update, rot_update, torsion_update):
        """ Modify the conformation of the ligand according to the update. 
        Args:
            data: heterogeneous graph with ligand information
            tr_update: update of the translation
            rot_update: update of the rotation angle
            torsion_update: update of the torsion angle
        """
        # Translate and rotate the ligand.
        lig_center = torch.mean(data['ligand'].pos, dim=0, keepdim=True)
        rot_mat = self.axis_angle_to_matrix(rot_update.squeeze())
        rigid_new_pos = (data['ligand'].pos - lig_center) @ rot_mat.T + tr_update + lig_center

        # Twist the rotatable bonds in the molecule.
        if torsion_update is not None:
            flexible_new_pos = self.modify_conformer_torsion_angles(
                rigid_new_pos, data['ligand','ligand'].edge_index.T[data['ligand'].edge_mask],
                data['ligand'].mask_rotate if isinstance(
                    data['ligand'].mask_rotate, np.ndarray) else data['ligand'].mask_rotate[0],
                torsion_update).to(rigid_new_pos.device)
            R, t = self.rigid_transform_Kabsch_3D_torch(flexible_new_pos.T, rigid_new_pos.T)
            aligned_flexible_pos = flexible_new_pos @ R.T + t.T
            data['ligand'].pos = aligned_flexible_pos
        else:
            data['ligand'].pos = rigid_new_pos
        return data

    def axis_angle_to_matrix(self, axis_angle):
        """ Convert rotations given as axis/angle to rotation matrices.
        Args:
            axis_angle: rotations given as a vector in axis angle form,
                as a tensor of shape (..., 3), where the magnitude is
                the angle turned anticlockwise in radians around the
                vector's direction.
        Returns:
            rotation matrices as tensor of shape (..., 3, 3).
        """
        # Convert rotations given as axis/angle to quaternions.
        angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
        half_angles = 0.5 * angles
        eps = 1e-6
        small_angles = angles.abs() < eps
        sin_half_angles_over_angles = torch.empty_like(angles)
        sin_half_angles_over_angles[~small_angles] = (
                torch.sin(half_angles[~small_angles]) / angles[~small_angles]
        )
        # for x small, sin(x/2) is about x/2 - (x/2)^3/6
        # so sin(x/2)/x is about 1/2 - (x*x)/48
        sin_half_angles_over_angles[small_angles] = (
                0.5 - (angles[small_angles] * angles[small_angles]) / 48
        )
        quaternions = torch.cat(
            [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
        )
        
        # Convert rotations given as quaternions to rotation matrices.
        r, i, j, k = torch.unbind(quaternions, -1)
        two_s = 2.0 / (quaternions * quaternions).sum(-1)

        o = torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            -1,
        )
        return o.reshape(quaternions.shape[:-1] + (3, 3))

    def modify_conformer_torsion_angles(self, pos, edge_index, mask_rotate, torsion_update, as_numpy=False):
        pos = copy.deepcopy(pos)
        if type(pos) != np.ndarray: pos = pos.cpu().numpy()

        for idx_edge, e in enumerate(edge_index.cpu().numpy()):
            if torsion_update[idx_edge] == 0:
                continue
            u, v = e[0], e[1]

            # check if need to reverse the edge, v should be connected to the part that gets rotated
            assert not mask_rotate[idx_edge, u]
            assert mask_rotate[idx_edge, v]

            rot_vec = pos[u] - pos[v]  # convention: positive rotation if pointing inwards
            rot_vec = rot_vec * torsion_update[idx_edge] / np.linalg.norm(rot_vec) # idx_edge!
            rot_mat = R.from_rotvec(rot_vec).as_matrix()

            pos[mask_rotate[idx_edge]] = (pos[mask_rotate[idx_edge]] - pos[v]) @ rot_mat.T + pos[v]

        if not as_numpy: pos = torch.from_numpy(pos.astype(np.float32))
        return pos

    def rigid_transform_Kabsch_3D_torch(self, A, B):
        # R = 3x3 rotation matrix, t = 3x1 column vector
        # This already takes residue identity into account.

        assert A.shape[1] == B.shape[1]
        num_rows, num_cols = A.shape
        if num_rows != 3:
            raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
        num_rows, num_cols = B.shape
        if num_rows != 3:
            raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")


        # find mean column wise: 3 x 1
        centroid_A = torch.mean(A, axis=1, keepdims=True)
        centroid_B = torch.mean(B, axis=1, keepdims=True)

        # subtract mean
        Am = A - centroid_A
        Bm = B - centroid_B

        H = Am @ Bm.T

        # find rotation
        U, S, Vt = torch.linalg.svd(H)

        R = Vt.T @ U.T
        # special reflection case
        if torch.linalg.det(R) < 0:
            # print("det(R) < R, reflection detected!, correcting for it ...")
            SS = torch.diag(torch.tensor([1.,1.,-1.], device=A.device))
            R = (Vt.T @ SS) @ U.T
        assert math.fabs(torch.linalg.det(R) - 1) < 3e-3  # note I had to change this error bound to be higher

        t = -R @ centroid_A + centroid_B
        return R, t
'''
def add_noise_to_lightgraph(graphs, lightgraphs, index_map, statistics, noise_rate, noise_rate_feat, 
                            stride, i):
    """ Get ith lightgraph
        Add noise to node features of the graph (pressure and flowrate).
    Args:
        i: index of the graph
    Returns:
        graph
    """
    graph_idx, time_idx = index_map[i,0], index_map[i,1]
    
    # add regular noise to the node features to prevent overfitting
    features = graphs[graph_idx].ndata['nfeatures'].clone()
    nf = features[:,:,time_idx].clone()

    dt = invert_normalize(graphs[graph_idx].ndata['dt'][0], 'dt', statistics, 'features')
    noise = np.random.normal(0, noise_rate*dt, nf[:,:2].shape)
    nf[:,:2] += noise

    nfnoise = np.random.normal(0, noise_rate_feat, nf[:,2:].shape)
    # flowrate at inlet is exact
    nfnoise[graphs[graph_idx].ndata['inlet_mask'].bool(),1] = 0
    nf[:,2:] += nfnoise

    lightgraphs[graph_idx].ndata['nfeatures'] = nf

    ns = features[:,0:2,time_idx+1:time_idx+1+stride].clone()
    lightgraphs[graph_idx].ndata['next_steps'] = ns

    # add regular noise to the edge features to prevent overfitting
    ef = graphs[graph_idx].edata['efeatures']
    efnoise = np.random.normal(0, noise_rate_feat, ef[:,2:].shape)
    ef[:,2:] += efnoise
    lightgraphs[graph_idx].edata['efeatures'] = ef.squeeze()

    return lightgraphs[graph_idx]
'''