import copy
import torch
import numpy as np
from tqdm import trange, tqdm
from torch_geometric.loader import DataLoader
from scipy.spatial.transform import Rotation as R

import gnn4s
from .predictor_corrector import ReverseDiffusionPredictor, LangevinCorrector

class PCSampler():
    def __init__(self, model_x, model_a, sde_x, sde_a, continuous=False, probability_flow=False,
                 snr=0.1, scale_eps=1.0, time_steps_num=1, eps=1e-3, denoise=True, device='cuda'):
        self.model_x = model_x
        self.model_a = model_a
        self.sde_x = sde_x
        self.sde_a = sde_a
        self.score_x = gnn4s.train.Score(model=self.model_x, sde=self.sde_x, train=False, continuous=continuous)
        self.score_a = gnn4s.train.Score(model=self.model_a, sde=self.sde_a, train=False, continuous=continuous)
        self.predictor = ReverseDiffusionPredictor
        self.corrector = LangevinCorrector

        self.predictor_x = self.predictor(self.sde_x, self.score_x, probability_flow, is_adj=False)
        self.corrector_x = self.corrector(self.sde_x, self.score_x, snr, scale_eps, time_steps_num, is_adj=False)

        self.predictor_a = self.predictor(self.sde_a, self.score_a, probability_flow, is_adj=True)
        self.corrector_a = self.corrector(self.sde_a, self.score_a, snr, scale_eps, time_steps_num, is_adj=True)

        self.time_steps_num = time_steps_num
        self.eps = eps
        self.denoise = denoise
        self.device = device

    def sample(self, init_mask, shape_x, shape_a):

        with torch.no_grad():
            # -------- Initial sample --------
            x = self.sde_x.prior_sampling(shape_x).to(self.device) 
            a = self.sde_a.prior_sampling_sym(shape_a).to(self.device) 
            mask = init_mask
            x = gnn4s.data.mask_x(x, mask)
            a = gnn4s.data.mask_adj(a, mask)
            diff_steps = self.sde_a.time_steps_num
            timesteps = torch.linspace(self.sde_a.time_end, self.eps, diff_steps, device=self.device)

            # -------- Reverse diffusion process --------
            for i in trange(0, (diff_steps), desc = '[Sampling]', position = 1, leave=False):
                t = timesteps[i]
                vec_t = torch.ones(shape_a[0], device=t.device) * t

                _x = x
                x, x_mean = self.corrector_x.update_fn(x, a, mask, vec_t)
                a, a_mean = self.corrector_a.update_fn(_x, a, mask, vec_t)

                _x = x
                x, x_mean = self.predictor_x.update_fn(x, a, mask, vec_t)
                a, a_mean = self.predictor_a.update_fn(_x, a, mask, vec_t)
            print(' ')
            return (x_mean if self.denoise else x), (a_mean if self.denoise else a), diff_steps * (self.time_steps_num + 1)

def adj2graph(adj, sample_node):
    """Covert the PyTorch tensor adjacency matrices to numpy array.

    Args:
        adj: [Batch_size, channel, Max_node, Max_node], assume channel=1
        sample_node: [Batch_size]
    """
    adj_list = []
    # discretization
    adj[adj >= 0.5] = 1.
    adj[adj < 0.5] = 0.
    for i in range(adj.shape[0]):
        adj_tmp = adj[i, 0]
        # symmetric
        adj_tmp = torch.tril(adj_tmp, -1)
        adj_tmp = adj_tmp + adj_tmp.transpose(0, 1)
        # truncate
        adj_tmp = adj_tmp.cpu().numpy()[:sample_node[0], :sample_node[0]]
        # adj_tmp = adj_tmp.cpu().numpy()[:sample_node[i], :sample_node[i]]
        adj_list.append(adj_tmp)

    return adj_list

def modify_conformer_torsion_angles(pos, edge_index, mask_rotate, torsion_updates, as_numpy=False):
    pos = copy.deepcopy(pos)
    if type(pos) != np.ndarray: pos = pos.cpu().numpy()

    for idx_edge, e in enumerate(edge_index.cpu().numpy()):
        if torsion_updates[idx_edge] == 0:
            continue
        u, v = e[0], e[1]

        # check if need to reverse the edge, v should be connected to the part that gets rotated
        assert not mask_rotate[idx_edge, u]
        assert mask_rotate[idx_edge, v]

        rot_vec = pos[u] - pos[v]  # convention: positive rotation if pointing inwards
        rot_vec = rot_vec * torsion_updates[idx_edge] / np.linalg.norm(rot_vec) # idx_edge!
        rot_mat = R.from_rotvec(rot_vec).as_matrix()

        pos[mask_rotate[idx_edge]] = (pos[mask_rotate[idx_edge]] - pos[v]) @ rot_mat.T + pos[v]

    if not as_numpy: pos = torch.from_numpy(pos.astype(np.float32))
    return pos

def randomize_position(data_list, no_torsion, no_random, tr_sigma_max):
    # in place modification of the list
    if not no_torsion:
        # randomize torsion angles
        idx = 0
        for complex_graph in data_list:
            torsion_updates = np.random.uniform(low=-np.pi, high=np.pi, size=complex_graph['ligand'].edge_mask.sum())
            idx += 1
            complex_graph['ligand'].pos = \
                modify_conformer_torsion_angles(complex_graph['ligand'].pos,
                                                complex_graph['ligand', 'ligand'].edge_index.T[
                                                complex_graph['ligand'].edge_mask],
                                                complex_graph['ligand'].mask_rotate[0], torsion_updates)
    idx = 0
    for complex_graph in data_list:
        # randomize position
        molecule_center = torch.mean(complex_graph['ligand'].pos, dim=0, keepdim=True)
        random_rotation = torch.from_numpy(R.random().as_matrix()).float()
        complex_graph['ligand'].pos = (complex_graph['ligand'].pos - molecule_center) @ random_rotation.T
        # base_rmsd = np.sqrt(np.sum((complex_graph['ligand'].pos.cpu().numpy() - orig_complex_graph['ligand'].pos.numpy()) ** 2, axis=1).mean())

        if not no_random:  # note for now the torsion angles are still randomised
            tr_update = torch.normal(mean=0, std=tr_sigma_max, size=(1, 3))
            complex_graph['ligand'].pos += tr_update
        
        idx += 1

class Sampler():
    def __init__(self, t_to_sigma, transform, score_model, confi_model, no_torsion, score_all_atom, confi_all_atom,
                 tr_sigma_min, tr_sigma_max, rot_sigma_min, rot_sigma_max, tor_sigma_min, tor_sigma_max, 
                 no_random=False, use_ode=False, no_final_step_noise=False, device='cpu'):
        self.t_to_sigma = t_to_sigma
        self.transform = transform
        self.score_model = score_model
        self.confi_model = confi_model
        self.no_torsion = no_torsion
        self.score_all_atom = score_all_atom
        self.confi_all_atom = confi_all_atom
        self.tr_sigma_min = tr_sigma_min
        self.tr_sigma_max = tr_sigma_max
        self.rot_sigma_min = rot_sigma_min
        self.rot_sigma_max = rot_sigma_max
        self.tor_sigma_min = tor_sigma_min
        self.tor_sigma_max = tor_sigma_max
        self.no_random = no_random
        self.use_ode = use_ode
        self.no_final_step_noise = no_final_step_noise
        self.device = device

    def get_t_schedule(self, inference_steps):
        return np.linspace(1, 0, inference_steps + 1)[:-1]

    def sampling(self, data_list, confidence_data_list=None, batch_size: int=32, inference_steps: int=10):
        N = len(data_list)

        tr_schedule = self.get_t_schedule(inference_steps)
        rot_schedule = self.get_t_schedule(inference_steps)
        tor_schedule = self.get_t_schedule(inference_steps)

        data_list_process = []

        for t_idx in range(inference_steps):
            t_tr, t_rot, t_tor = tr_schedule[t_idx], rot_schedule[t_idx], tor_schedule[t_idx]
            dt_tr = tr_schedule[t_idx] - tr_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tr_schedule[t_idx]
            dt_rot = rot_schedule[t_idx] - rot_schedule[t_idx + 1] if t_idx < inference_steps - 1 else rot_schedule[t_idx]
            dt_tor = tor_schedule[t_idx] - tor_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tor_schedule[t_idx]

            loader = DataLoader(data_list, batch_size=batch_size)
            new_data_list = []

            for complex_graph_batch in loader:
                b = complex_graph_batch.num_graphs
                complex_graph_batch = complex_graph_batch.to(self.device)

                tr_sigma, rot_sigma, tor_sigma = self.t_to_sigma(t_tr, t_rot, t_tor)
                self.transform.set_time(complex_graph_batch, t_tr, t_rot, t_tor, b, self.score_all_atom, self.device)
        
                with torch.no_grad():
                    tr_score, rot_score, tor_score = self.score_model(complex_graph_batch)
        
                tr_g = tr_sigma * torch.sqrt(torch.tensor(2 * np.log(self.tr_sigma_max / self.tr_sigma_min)))
                rot_g = 2 * rot_sigma * torch.sqrt(torch.tensor(np.log(self.rot_sigma_max / self.rot_sigma_min)))

                if self.use_ode:
                    tr_perturb = (0.5 * tr_g ** 2 * dt_tr * tr_score.cpu()).cpu()
                    rot_perturb = (0.5 * rot_score.cpu() * dt_rot * rot_g ** 2).cpu()
                else:
                    tr_z = torch.zeros((b, 3)) if self.no_random or (self.no_final_step_noise and t_idx == inference_steps - 1) \
                        else torch.normal(mean=0, std=1, size=(b, 3))
                    tr_perturb = (tr_g ** 2 * dt_tr * tr_score.cpu() + tr_g * np.sqrt(dt_tr) * tr_z).cpu()

                    rot_z = torch.zeros((b, 3)) if self.no_random or (self.no_final_step_noise and t_idx == inference_steps - 1) \
                        else torch.normal(mean=0, std=1, size=(b, 3))
                    rot_perturb = (rot_score.cpu() * dt_rot * rot_g ** 2 + rot_g * np.sqrt(dt_rot) * rot_z).cpu()

                if not self.no_torsion:
                    tor_g = tor_sigma * torch.sqrt(torch.tensor(2 * np.log(self.tor_sigma_max / self.tor_sigma_min)))
                    if self.use_ode:
                        tor_perturb = (0.5 * tor_g ** 2 * dt_tor * tor_score.cpu()).numpy()
                    else:
                        tor_z = torch.zeros(tor_score.shape) if self.no_random or (self.no_final_step_noise and t_idx == inference_steps - 1) \
                            else torch.normal(mean=0, std=1, size=tor_score.shape)
                        tor_perturb = (tor_g ** 2 * dt_tor * tor_score.cpu() + tor_g * np.sqrt(dt_tor) * tor_z).numpy()
                    torsions_per_molecule = tor_perturb.shape[0] // b
                else:
                    tor_perturb = None

                # Apply noise
                new_data_list.extend([self.transform.modify_conformer(complex_graph, tr_perturb[i:i+1], rot_perturb[i:i+1].squeeze(0),
                                      tor_perturb[i*torsions_per_molecule:(i+1) * torsions_per_molecule] if (not self.no_torsion) else None)
                for i, complex_graph in enumerate(complex_graph_batch.to('cpu').to_data_list())])
            data_list = new_data_list
            data_list_process.append(new_data_list)

        with torch.no_grad():
            if self.confi_model is not None:
                loader = DataLoader(data_list, batch_size=batch_size)
                confidence_loader = iter(DataLoader(confidence_data_list, batch_size=batch_size))
                confidence = []
                for complex_graph_batch in loader:
                    complex_graph_batch = complex_graph_batch.to(self.device)
                    if confidence_data_list is not None:
                        confidence_complex_graph_batch = next(confidence_loader).to(self.device)
                        confidence_complex_graph_batch['ligand'].pos = complex_graph_batch['ligand'].pos
                        self.transform.set_time(confidence_complex_graph_batch, 0, 0, 0, N,
                                                self.confi_all_atom, self.device)
                        confidence.append(self.confi_model(confidence_complex_graph_batch))
                    else:
                        confidence.append(self.confi_model(complex_graph_batch))
                confidence = torch.cat(confidence, dim=0)
            else:
                confidence = None

        return data_list, confidence, data_list_process

class GeneratorDiffDock():
    def __init__(self, test_loader, confi_test_dataset, confi_model, sampler,
                 inference_step_num, sample_per_complex, rmsd_class_cutoff, logger):
        self.test_loader = test_loader
        self.confi_test_dataset = confi_test_dataset
        self.confi_model = confi_model
        self.sampler = sampler
        self.inference_step_num = inference_step_num
        self.sample_per_complex = sample_per_complex
        self.rmsd_class_cutoff = rmsd_class_cutoff
        self.logger = logger

    def generate(self, inference_sample_size, output_dirt):
        failures, skipped, confidences_list, names_list, min_self_distances_list = 0, 0, [], [], []
        self.logger.log_info(f'Size of test dataset: {len(self.test_loader.dataset)}')

        confi_complex_dict = {d.name: d for d in self.confi_test_dataset}
        for idx, orig_complex_graph in tqdm(enumerate(self.test_loader)):
            if self.confi_model is not None and orig_complex_graph.name[0] not in confi_complex_dict.keys():
                skipped += 1
                self.logger.log_info(f'HAPPENING | The confidence dataset did not contain {orig_complex_graph.name[0]}. '+
                                'We are skipping this complex.')
                continue

            # form data list
            data_list = [copy.deepcopy(orig_complex_graph) for _ in range(self.sample_per_complex)]
            
            if self.confi_model is not None:
                confi_data_list = [copy.deepcopy(confi_complex_dict[orig_complex_graph.name[0]]) 
                                for _ in range(self.sample_per_complex)]
            else:
                confi_data_list = None
            
            # sampling
            gnn4s.diffusion.randomize_position(data_list, self.sampler.no_torsion, 
                self.sampler.no_random, self.sampler.tr_sigma_max)
            
            data_list, confidence, data_list_process = self.sampler.sampling(data_list=data_list,
                confidence_data_list=confi_data_list,
                batch_size=inference_sample_size, inference_steps=self.inference_step_num)
            
            self.logger.log_info('confidence:')
            self.logger.log_info(confidence)
            
            lig_pos = np.asarray([(complex_graph['ligand'].pos + orig_complex_graph.original_center).cpu().numpy() 
                                for complex_graph in data_list])
            lig_pos_process = [np.asarray([(complex_graph['ligand'].pos + orig_complex_graph.original_center).cpu().numpy() 
                                        for complex_graph in dlist]) for dlist in data_list_process] ##

            # sort
            if confidence is not None:
                confidence, re_order = gnn4s.postprocess.sort_confidence(confidence, self.rmsd_class_cutoff)
                confidences_list.append(confidence)
                lig_pos = lig_pos[re_order]
                for i in range(len(lig_pos_process)):
                    lig_pos_process[i] = lig_pos_process[i][re_order]
            
            min_self_distances_list.append(gnn4s.postprocess.cal_min_self_distances(lig_pos))
            names_list.append(orig_complex_graph.name[0])

            # write ligand
            lig = orig_complex_graph.mol[0]
            gnn4s.postprocess.write_mol_with_cord(lig, lig_pos, self.confi_test_dataset.remove_hs, confidence,
                output_dirt, idx, data_list[0]["name"][0], lig_pos_process)
            

        np.save(f'{output_dirt}/min_self_distances.npy', np.array(min_self_distances_list))
        np.save(f'{output_dirt}/confidences.npy', np.array(confidences_list))
        np.save(f'{output_dirt}/complex_names.npy', np.array(names_list))

        self.logger.log_info(f'Failed for {failures} complexes')
        self.logger.log_info(f'Skipped {skipped} complexes')
        self.logger.log_info(f'Results are in {output_dirt}')
