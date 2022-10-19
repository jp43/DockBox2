import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import argparse
import pickle
import numpy as np
import tensorflow as tf

import loss
import utils

nrof_neigh_per_batch = 25
depth = 2

def __sample(graph_id):
    graph_size = train['Graph_size'][graph_id]
    adj = train['Graph_adj_list'][graph_id]
    diff = train['max_nrof_nodes'] - graph_size

    if diff > 0:
        vertices = np.concatenate([train['Graph_feats'][graph_id], np.zeros((diff, nfeats), dtype=np.float32)], axis=0)
        labels = np.concatenate([train['Graph_labels'][graph_id], np.zeros((diff, nlabels), dtype=np.int32)], axis=0)
    else:
        vertices = train['Graph_feats'][graph_id]
        labels = train['Graph_labels'][graph_id]

    indices_all, adj_mask_all, nneigh_all = utils.generate_neighbours(adj, graph_size, depth, nrof_neigh_per_batch)
    sys.exit()

def sample(graph_id):
    return tf.py_function(__sample, [graph_id], tf.int64)

with open('datasets/data_ppi/train_ppi.pickle', "rb") as ff:
   train = pickle.load(ff)

train['Graph_size'] = []
for i_g, graph in enumerate(train['Graph_nodes']):
    train['Graph_size'].append(len(graph))

nfeats = train['Graph_feats'][0].shape[1]
nlabels = train['Graph_labels'][0].shape[1]

data_slices = np.random.permutation(len(train['Graph_nodes']))
data_loader = tf.data.Dataset.from_tensor_slices((data_slices))

data_loader = data_loader.map(**{'num_parallel_calls': 4, 'map_func': sample})

for tensor in data_loader:
    pass

