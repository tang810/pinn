import torch
import copy
import time

import gnn4s

class Evaluator():
    def __init__(self, dataloader, predictor):
        """ Evaluator for predicting and evaluating error.
        Args:
            dataloader: dataloader for load valid dataset
            predictor: predictor for predicting solution
        """
        self.dataloader = dataloader
        self.predictor = predictor
    
    def evaluate(self):
        """ Predict and evaluate error. """
        error, pred = [], []
        for data in self.dataloader:
            pre = self.predictor(data)

            err = torch.zeros(len(data))
            for i in range(len(data)):
                err[i] = ((pre[i]-data[i].y)**2).mean()
            
            error.append(err)
            pred.append(pre)

        return error, pred

class EvaluatorCardiovascular():
    def __init__(self, dataset, predictor, bc_type, statistics):
        self.dataset = dataset
        self.predictor = predictor
        self.bc_type = bc_type
        self.statistics = statistics
    
    def evaluate_error(self, graph, pred):
        pred = torch.tensor(pred).clone()

        true_graph = copy.deepcopy(graph)
        tgnf = true_graph.ndata['nfeatures'][:,0:2,:].clone()

        # we only compute errors on branch nodes
        branch_mask = torch.reshape(graph.ndata['branch_mask'],(-1,1,1))
        branch_mask = branch_mask.repeat(1,2,tgnf.shape[2])

        # compute error
        tgnf = tgnf * branch_mask
        pred = pred * branch_mask

        diff = tgnf - pred
        errs = torch.sum(torch.sum(diff**2, dim = 0), dim = 1)
        errs = errs / torch.sum(torch.sum(tgnf**2, dim = 0), dim = 1)
        errs_normalized = torch.sqrt(errs)

        tgnf[:,0,:] = gnn4s.data.invert_normalize(
            tgnf[:,0,:], 'pressure', self.statistics, 'features')
        tgnf[:,1,:] = gnn4s.data.invert_normalize(
            tgnf[:,1,:], 'flowrate', self.statistics, 'features')

        pred[:,0,:] = gnn4s.data.invert_normalize(
            pred[:,0,:], 'pressure', self.statistics, 'features')
        pred[:,1,:] = gnn4s.data.invert_normalize(
            pred[:,1,:], 'flowrate', self.statistics, 'features')

        diff = tgnf - pred
        errs = torch.sum(torch.sum(diff**2, dim=0), dim=1)
        errs = errs / torch.sum(torch.sum(tgnf**2, dim=0), dim=1)
        errs = torch.sqrt(errs)

        return errs_normalized.detach().numpy(), errs.detach().numpy()

    def compute_rollout_errors(self, dataset):
        """ Compute rollout errors

        Arguments:
            gnn_model: the GNN
            params: dictionary of parameters
            dataset: the dataset over which the errors are computed
            idxs_train: indices of graphs to use to evaluating the training
            idxs_test: indices of graphs to use to evaluate the test
        
        Returns:
            2D array containing the error for pressure and flow rate (train)
            2D array containing the error for pressure and flow rate (test)

        """
        # import random
        # np.random.seed(10)
        ngraphs = np.min((10, len(dataset.data_list)))
        idxs_train = random.sample(range(len(dataset.data_list)), ngraphs)

        train_errs = np.zeros(2)
        for idx in idxs_train:
            pred = self.predictor(dataset.data_list[idx])
            cur_train_errs, _ = self.evaluate_error(dataset.data_list[idx], pred)
            train_errs = cur_train_errs + train_errs
        
        train_errs = train_errs / len(idxs_train)

        return train_errs

    def evaluate(self):
        """
        Runs the rollout phase for all models and computes errors.

        Arguments:
            dataset: dictionary containing two keys, 'train' and 'test', with
                    the different datasets
            self.model: the GNN
            params: dictionary of parameters
        Returns:
            2D array containing average pressure and flow rate normalized errors
            2D array containing average pressure and flow rate errors, 
            Average continuity loss
            Average run time
            Average number of timesteps

        """
        pred = []
        time_steps_total = 0
        time_total = 0
        error_total = 0
        for i in range(0,len(self.dataset.data_list)):
            print('model name = {}'.format(self.dataset.split_idx[i]))
            
            start_time = time.time()
            r_features = self.predictor(self.dataset.data_list[i])
            _, error = self.evaluate_error(self.dataset.data_list[i], r_features)
            elapsed = time.time() - start_time
            
            pred.append(r_features)
            time_total += elapsed
            time_steps_total += r_features.shape[2]
            print('error:')
            print(error)
            error_total += error

        data_num = len(self.dataset.data_list)
        return pred, error_total/data_num, time_total/data_num, time_steps_total/data_num