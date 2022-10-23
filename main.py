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

total_epochs = 1

depth = 2
activation = 'sigmoid'
nrof_neigh_per_batch = 25
batch = 2

# aggregator options
aggregator_options = {'aggregator_shape': 50,
'use_concat': True,
'aggregator_type': 'PoolAggregator',
'aggregator_type_options': {'activation': 'leaky_relu', 'pool_op': 'reduce_max'}}

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
        for kdx, graph in enumerate(data['Graph_nodes']):
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

    def __init__(self, in_shape, out_shape, activation, depth, nrof_neigh_per_batch, aggregator_options):
        super(GraphSAGE, self).__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape

        self.activation = activation
        if self.activation is not None:
            self.activation = getattr(tf.nn, activation)

        self.depth = depth
        self.nrof_neigh_per_batch = nrof_neigh_per_batch

        self.aggregator_options = aggregator_options
        self.aggregator_options['aggregator_type_options']['use_concat'] = self.aggregator_options['use_concat']

        self.loss_function = tf.keras.losses.BinaryCrossentropy()
        self.accuracy = tfa.metrics.F1Score(num_classes=out_shape, average='micro', threshold=0.5)

    def build(self):
        sys._getframe(1).f_locals.update(self.aggregator_options)

        self.output_layer = tf.keras.layers.Dense(self.out_shape, input_shape=((use_concat+1)*aggregator_shape,), name='output_layer')
        self.output_layer.build(((use_concat+1)*aggregator_shape,))

        self.aggregator_layers = []
        for idx in range(self.depth):
            # TODO: possibility to create different classes of aggregators
            aggregator = PoolAggregator(**aggregator_type_options)
        
            if idx == 0:
                in_shape = self.in_shape
            else:
                in_shape = (use_concat+1)*aggregator_shape

            aggregator.build(in_shape, aggregator_shape)
            self.aggregator_layers.append(aggregator)

        super(GraphSAGE, self).build(())

    def call(self, feats, graph_size, neigh_indices, adj_mask, nneigh, labels, training=True):

        nrof_graphs = len(graph_size) # number of graphs in the batch
        graph_cumsize = np.insert(np.cumsum(graph_size), 0, 0)

        # initialize self_feats
        self_feats = feats[0][:graph_size[0]]
        for kdx in range(1, nrof_graphs):
             self_feats = tf.concat([self_feats, feats[kdx][:graph_size[kdx]]], axis=0)

        for idx in range(self.depth):
            # construct neigh_feats form self_feats
            for kdx in range(nrof_graphs):
                graph_self_feats = tf.gather(self_feats, tf.range(graph_cumsize[kdx], graph_cumsize[kdx+1]))
                graph_neigh_feats = tf.gather(graph_self_feats, neigh_indices[kdx][idx][:graph_size[kdx],:])

                # needs to exclude zero-padded nodes
                graph_adj_mask = adj_mask[kdx][idx][:graph_size[kdx],:]
                graph_neigh_feats = tf.multiply(tf.expand_dims(graph_adj_mask, axis=2), graph_neigh_feats)

                if kdx == 0:
                    neigh_feats = graph_neigh_feats
                else:
                    neigh_feats = tf.concat([neigh_feats, graph_neigh_feats], axis=0)

            self_feats = self.aggregator_layers[idx](self_feats, neigh_feats, training=training)
        self_feats = tf.math.l2_normalize(self_feats, axis=1)

        embedded_feats = self.output_layer(self_feats)

        predictions = self.activation(self_feats)
        output = predictions

        if len(labels.shape) > 1:
            batch_labels = labels[0][:graph_size[0],:]
            for kdx in range(1, nrof_graphs):
                batch_labels = tf.concat([batch_labels, labels[kdx][:graph_size[kdx],:]], axis=0)
        else:
            batch_labels = np.expand_dims(labels, 1)

        #TODO: needs to add loss function accuracy

train = GraphDataset('datasets/data_ppi/train_ppi.pickle')

data_slices = np.random.permutation(train.ngraphs)
data_loader = tf.data.Dataset.from_tensor_slices((data_slices))

data_loader = data_loader.map(**{'num_parallel_calls': 1, 'map_func': lambda x: train.sample(x, depth, nrof_neigh_per_batch)})
data_loader = data_loader.batch(batch)

model = GraphSAGE(train.nfeats, train.nlabels, activation, depth, nrof_neigh_per_batch, aggregator_options)
model.build()

model.summary()

for epoch in range(total_epochs):
    for idx, data_batch in enumerate(data_loader):

        with tf.GradientTape() as tape:
            model(*data_batch, training=True)

