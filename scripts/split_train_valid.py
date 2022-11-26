import os
import sys
import pickle

import copy
import shutil
from random import seed, sample
import numpy as np
import networkx as nx

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

# set fraction for test set
fraction_test = 0.0
normalize = True
cross_validation = True

# EITHER set fraction for train set (cross_validation = False)
fraction_train = 0.5
fraction_val = 1 - fraction_train - fraction_test

# OR options for cross-validation (cross_validation = True)
nsplits = 5

cutoff_correct_pose = 2.0
max_rmsd_interpose = 5.0

downsampling = True
max_nsamples = 50
min_rmsd_interpose = 0.1

nargs = len(sys.argv)
if nargs == 2:
    pickfile = sys.argv[1]
else:
    raise ValueError("Number of arguments should be 1 (pickle file containing all the graphs)")

with open(pickfile, "rb") as ff:
   graphs = pickle.load(ff)

pdbids = set(graphs.keys())
pdbids = sorted(list(pdbids))

npdbids = len(pdbids)

ntest = int(fraction_test*npdbids)
pdbids_test = sorted(sample(pdbids, ntest))

pdbids_train_val = []
labels_train_val = []

for pdbid in pdbids:
    G = graphs[pdbid]

    if downsampling:
        # remove nodes that are close from one another
        nearby_nodes = filter(lambda e: e[2] < min_rmsd_interpose, list(G.edges.data('rmsd')))
        redundant_nodes = []
        for node1, node2, rmsd in list(nearby_nodes):
            redundant_nodes.append(node2)

        redundant_nodes = list(set(redundant_nodes))
        G.remove_nodes_from(redundant_nodes)

    labels = []
    for node, data in G.nodes(data=True):
        # set label from cutoff_correct_pose
        if data['rmsd'] <= cutoff_correct_pose:
            label = 1
        else:
            label = 0
        data['label'] = label
        labels.append(label)

    if pdbid not in pdbids_test:
        pdbids_train_val.append(pdbid)
        labels_train_val.append(np.any(np.array(labels)==1).astype(int))

    # remove edges with rmsd greater than max_rmsd_interpose
    discarded_edges = filter(lambda e: e[2] > max_rmsd_interpose, list(G.edges.data('rmsd')))
    G.remove_edges_from(list(discarded_edges))


if cross_validation:
    skf = StratifiedKFold(n_splits=nsplits)

    datasets_list = []
    for idxs_train, idxs_val in skf.split(np.zeros_like(labels_train_val), labels_train_val):
        pdbids_train = list(pdbids_train_val[idx] for idx in idxs_train)
        pdbids_val = list(pdbids_train_val[idx] for idx in idxs_val)

        datasets_list.append({'train': sorted(pdbids_train), 'val': sorted(pdbids_val), 'test': pdbids_test})
else:
    # select only one train and validation set
    ntrain = int(fraction_train*npdbids)

    datasets = {'test': pdbids_test}
    datasets['train'] = sorted(sample(list(set(pdbids) - set(datasets['test'])), ntrain))
    datasets['val'] = sorted(list(set(pdbids) - set(datasets['train']) - set(datasets['test'])))

    datasets_list = [datasets]

for kdx, datasets in enumerate(datasets_list):
    ratio_correct_incorrect = []

    graphs_copy = copy.deepcopy(graphs) 
    for pdbid in pdbids:
        G = graphs_copy[pdbid]
    
        correct_nodes = []
        incorrect_nodes = []
        for node, data in G.nodes(data=True):
            if data['label'] == 1:
                correct_nodes.append(node)
            else:
                incorrect_nodes.append(node)

        # select maximum number of samples
        if pdbid in datasets['train']:
            if downsampling and max_nsamples is not None:
                if len(correct_nodes) > max_nsamples:
                    correct_nodes = sample(correct_nodes, max_nsamples)
     
                if len(incorrect_nodes) > max_nsamples:
                    incorrect_nodes = sample(incorrect_nodes, max_nsamples)
    
                discarded_nodes = list(set(G.nodes())-set(correct_nodes + incorrect_nodes))
                G.remove_nodes_from(discarded_nodes)
    
            #print("%s: correct: %i, incorrect: %i"%(pdbid, len(correct_nodes), len(incorrect_nodes)))
            if incorrect_nodes:
                ratio_correct_incorrect.append(len(correct_nodes)*1./len(incorrect_nodes))

    if cross_validation:
        print('split %i: training set: %i elements, validation set: %i elements'%(kdx+1, len(datasets['train']), len(datasets['val'])))
    else:
        print('training set: %i elements, validation set: %i elements'%(len(datasets['train']), len(datasets['val'])))

    print("alpha coefficient needed for balanced set: %.3f"%(1/(1+np.mean(ratio_correct_incorrect))))
    if normalize:
        feats = []
        for pdbid in datasets['train']:
            for node, data in graphs_copy[pdbid].nodes(data=True):
                feats.append(data['feature'])
    
        feats = np.vstack(feats)
        scaler = StandardScaler()
        scaler.fit(feats)
    
        for pdbid in graphs_copy:
            G = graphs_copy[pdbid]
            for node, data in G.nodes(data=True):
                normalized_feats = scaler.transform(data['feature'][np.newaxis,:])
                G.nodes[node]['feature'] = normalized_feats.squeeze()
    
    for setname, dataset_pdbids in datasets.items():
        dataset_graphs = []
        for pdbid in dataset_pdbids:
            dataset_graphs.append(graphs_copy[pdbid])

        if cross_validation:    
            filename = setname + '_%i.pickle'%(kdx+1)
        else:
            filename = setname + '.pickle'

        if dataset_pdbids:
            with open(filename, "wb") as ff:
                pickle.dump(dataset_graphs, ff)

