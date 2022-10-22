import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import argparse
import pickle
import numpy as np

import tensorflow as tf
import tensorflow_addons as tfa

import utils

from aggregators import *

depth = 2
activation = 'sigmoid'
nrof_neigh_per_batch = 25

# aggregator options
aggregator_options = {'aggregator_shape': 50,
'use_concat': True,
'aggregator_type': 'PoolAggregator',
'aggregator_type_options': {'activation': 'leaky_relu', 'pool_op': 'sigmoid'}}

class GraphDataset(object):

    def __init__(self, filename):

        with open(filename, "rb") as ff:
           data = pickle.load(ff)
 
        self.graph_feats = data['Graph_feats']
        self.graph_adj_list = data['Graph_adj_list']
        self.graph_labels = data['Graph_labels']
        self.max_nrof_nodes = data['max_nrof_nodes']

        self.nfeats = data['Graph_feats'][0].shape[1]

        self.nlabels = data['Graph_labels'][0].shape[1]
        self.ngraphs = len(data['Graph_nodes'])

        self.graph_size = []
        for i_g, graph in enumerate(data['Graph_nodes']):
            self.graph_size.append(len(graph))

    def __sample(self, graph_id, depth, nrof_neigh_per_batch):

        graph_size = self.graph_size[graph_id]
        adj = self.graph_adj_list[graph_id]
        diff = self.max_nrof_nodes - graph_size

        depth = tf.get_static_value(depth)
        nrof_neigh_per_batch = tf.get_static_value(nrof_neigh_per_batch)  

        if diff > 0:
            feats = np.concatenate([self.graph_feats[graph_id], np.zeros((diff, self.nfeats), dtype=np.float32)], axis=0)
            labels = np.concatenate([self.graph_labels[graph_id], np.zeros((diff, self.nlabels), dtype=np.int32)], axis=0)
        else:
            feats = self.graph_feats[graph_id]
            labels = self.graph_labels[graph_id]
    
        neigh_indices, adj_mask, nneigh = utils.sample_neighbors(adj, graph_size, depth, nrof_neigh_per_batch)
    
        for idx in range(depth):
            if diff > 0:
                # zero padding for each layer until getting max_nrof_nodes nodes per graph
                neigh_indices[idx] = np.concatenate([neigh_indices[idx], np.zeros((diff, nrof_neigh_per_batch), dtype=np.int32)], axis=0)
                adj_mask[idx] = np.concatenate([adj_mask[idx], np.zeros((diff, nrof_neigh_per_batch), dtype=np.int32)], axis=0)
                nneigh[idx] = np.concatenate([nneigh[idx], np.zeros((diff), dtype=np.int32)], axis=0)
    
        return np.asarray(feats, np.float32), graph_size, \
               np.asarray(neigh_indices, dtype=np.int32), np.asarray(adj_mask, dtype=np.float32), \
               np.asarray(nneigh, dtype=np.int32), labels
    
    def sample(self, graph_id, depth, nrof_neigh_per_batch):

        return tf.py_function(self.__sample, [graph_id, depth, nrof_neigh_per_batch], [tf.float32, tf.int32,
                                   tf.int32, tf.float32, tf.int32, tf.int32])

class GraphSAGE(tf.keras.models.Model):

    def __init__(self, in_shape, out_shape, activation, aggregator_options):
        super(GraphSAGE, self).__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape

        self.activation = activation
        if self.activation is not None:
            self.activation = getattr(tf.nn, activation)

        self.aggregator_options = aggregator_options
        self.aggregator_options['aggregator_type_options']['use_concat'] = self.aggregator_options['use_concat']

        self.loss_function = tf.keras.losses.BinaryCrossentropy()
        self.accuracy = tfa.metrics.F1Score(num_classes=out_shape, average='micro', threshold=0.5)

    def build(self):
        sys._getframe(1).f_locals.update(self.aggregator_options)

        self.output_layer = tf.keras.layers.Dense(self.out_shape, input_shape=((use_concat+1)*aggregator_shape,), name='output_layer')
        self.output_layer.build(((use_concat+1)*aggregator_shape,))

        self.aggregator_layers = []
        for idx in range(depth):
            # TODO: possibility to create different classes of aggregators
            aggregator = PoolAggregator(**aggregator_type_options)
        
            if idx == 0:
                in_shape = self.in_shape
            else:
                in_shape = (use_concat+1)*aggregator_shape

            aggregator.build(in_shape, aggregator_shape)
            self.aggregator_layers.append(aggregator)

        super(GraphSAGE, self).build(())

train = GraphDataset('datasets/data_ppi/train_ppi.pickle')

data_slices = np.random.permutation(train.ngraphs)
data_loader = tf.data.Dataset.from_tensor_slices((data_slices))

data_loader = data_loader.map(**{'num_parallel_calls': 4, 'map_func': lambda x: train.sample(x, depth, nrof_neigh_per_batch)})

model = GraphSAGE(train.nfeats, train.nlabels, activation, aggregator_options)
model.build()

model.summary()

