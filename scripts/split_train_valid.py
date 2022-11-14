import sys
import pickle

from random import seed, sample
import numpy as np
import networkx as nx

from sklearn.preprocessing import StandardScaler

# set random seed
seed(123)

# define fraction of train, validation and test sets
fraction_train = 0.5
fraction_val = 0.3

normalize = True

cutoff_correct_pose = 2.0
max_rmsd_interpose = 2.0

downsampling = True
max_nsamples = 40
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
ntrain = int(fraction_train*npdbids)
nval = int(fraction_val*npdbids)

datasets = {'train': sorted(sample(pdbids, ntrain))}
datasets['val'] = sorted(sample(list(set(pdbids) - set(datasets['train'])), nval))
datasets['test'] = sorted(list(set(pdbids) - set(datasets['train']) - set(datasets['val'])))

idx_train = -1
ratio_correct_incorrect = []

# set label from cutoff_correct_pose
for pdbid in pdbids:
    G = graphs[pdbid]

    if pdbid in datasets['train']:
        idx_train += 1

    if downsampling:
        # remove nodes that are close from one another
        nearby_nodes = filter(lambda e: e[2] < min_rmsd_interpose, list(G.edges.data('rmsd')))
        redundant_nodes = []
        for node1, node2, rmsd in list(nearby_nodes):
            redundant_nodes.append(node2)

        redundant_nodes = list(set(redundant_nodes))
        G.remove_nodes_from(redundant_nodes)

    correct_nodes = []
    incorrect_nodes = []
    for node, data in G.nodes(data=True):
        if data['rmsd'] <= cutoff_correct_pose:
            data['label'] = 1
            correct_nodes.append(node)
        else:
            data['label'] = 0
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

    # remove edges with rmsd greater than max_rmsd_interpose
    discarded_edges = filter(lambda e: e[2] > max_rmsd_interpose, list(G.edges.data('rmsd')))
    G.remove_edges_from(list(discarded_edges))

print("alpha coefficient needed for balanced set: %.3f"%(1/(1+np.mean(ratio_correct_incorrect))))
if normalize:
    feats = []
    for pdbid in datasets['train']:
        for node, data in graphs[pdbid].nodes(data=True):
            feats.append(data['feature'])

    feats = np.vstack(feats)
    scaler = StandardScaler()
    scaler.fit(feats)

    for pdbid in graphs:
        G = graphs[pdbid]
        for node, data in G.nodes(data=True):
            normalized_feats = scaler.transform(data['feature'][np.newaxis,:])
            G.nodes[node]['feature'] = normalized_feats.squeeze()

for setname, pdbids in datasets.items():
    dataset_graphs = []
    for pdbid in pdbids:
        dataset_graphs.append(graphs[pdbid])

    with open(setname+'_%iA.pickle'%max_rmsd_interpose, "wb") as ff:
        pickle.dump(dataset_graphs, ff) 

