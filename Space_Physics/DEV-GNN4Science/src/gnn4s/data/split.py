import numpy as np
import random
from functools import partial

from .dataset import DglDataset

def split(graphs_name, split_type, dataset_info, divs=5):
    """ Split a list of graphs.

    The graphs are split into multiple train/test groups. Number of groups is
    determined by the divs argument. The function takes as input the type of
    graphs to make the datasets balanced.

    Arguments:
        divs: number of train/test groups.
        types: dictionary (key: graph name, value: dataset info)

    Returns:
        List of groups
    """
    def chunk_data_list(data_list, n):
        data_list_chunk = [[] for l in range(n)]
        for i, data in enumerate(data_list):
            data_list_chunk[i%n].append(data)
        return data_list_chunk
    
    # we do this to keep sims with the same last number in the same group
    dictnames = {}
    for name in graphs_name:
        simname = name.split('.')[0] + '.' + name.split('.')[1]
        if simname not in dictnames:
            dictnames[simname] = []
        dictnames[simname].append(name)

    if len(dictnames) == 1:
        datasets = [{'train': dictnames[simname],
                     'test': dictnames[simname]}]
        return datasets

    names = list(dictnames.keys())

    # the seed MUST be set when using parallelism! Otherwise cores get different
    # splits
    random.seed(10)
    random.shuffle(names)

    sublists = {}
    for name in names:
        type = dataset_info[name]['model_type']
        if type not in sublists:
            sublists[type] = []
        sublists[type].append(name)

    subsets = {}
    for sublist_n, sublist_v in sublists.items():
        subsets[sublist_n] = list(chunk_data_list(sublist_v, divs))
        nsets = len(subsets[sublist_n])
        # we distribute the last sets among the first n-1
        if nsets != divs:
            for i, graph in enumerate(subsets[sublist_n][-1]):
                subsets[sublist_n][i % divs] += [dictnames[graph]]
            del subsets[sublist_n][-1]
        nsets = len(subsets[sublist_n])

    subsets_data_name = {}
    for subset_n, subset_v in subsets.items():
        list_all = []

        for single_list in subset_v:
            single_all = []
            for sim_name in single_list:
                single_all += dictnames[sim_name]

            list_all.append(single_all)
        subsets_data_name[subset_n] = list_all

    subsets = subsets_data_name

    datasets = []

    i = 0
    cur_set = []
    for _, subset_v in subsets.items():
        cur_set = cur_set + subset_v[i]

    newdata = {'test': cur_set}
    train_s = []
    for j in range(nsets):
        if j != i:
            cur_set = []
            for _, subset_v in subsets.items():
                cur_set = cur_set + subset_v[j]
            train_s = train_s + cur_set
    newdata['train'] = train_s
    datasets.append(newdata)
    
    if split_type=='train':
        return datasets[0]['train']
    if split_type=='test':
        return datasets[0]['test']
