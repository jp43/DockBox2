import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import time
import argparse
import pickle
import numpy as np

import tensorflow as tf
import tensorflow_addons as tfa

import utils
from aggregators import *

total_epochs = 10

# minibatch options
num_parallel_calls = 4
batch_size = 2

# model options
depth = 2
activation = 'sigmoid'
nrof_neigh_per_batch = 25

# aggregator options
#aggregator_options = {'shape': 50, 'use_concat': True, 'type': 'mean', 'activation': 'leaky_relu'}
#aggregator_options = {'shape': 50, 'use_concat': True, 'type': 'pooling', 'activation': 'leaky_relu'}
aggregator_options = {'shape': 50, 'use_concat': True, 'type': 'attention', 'attention_shape': 50, 'activation': 'leaky_relu', 'attention_activation': 'sigmoid'}

if 'shape' not in aggregator_options:
    sys.exit("Aggregation shape is mandatory in aggregator_options!")

for option in default_aggregator_options:
    if option not in aggregator_options:
        aggregator_options[option] = default_aggregator_options[option]

# optimizer options
optimizer_type = 'Adam'
lr_schedule = {'initial_learning_rate': 1e-2, 'decay_steps': 1000, 'decay_rate': 0.99, 'staircase':True}

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
               np.asarray(nneigh, dtype=np.float32), labels
    
    def sample(self, graph_id, depth, nrof_neigh_per_batch):

        return tf.py_function(self.__sample, [graph_id, depth, nrof_neigh_per_batch], [tf.float32, tf.int32,
                                   tf.int32, tf.float32, tf.float32, tf.int32])

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

        self.loss_function = tf.keras.losses.BinaryCrossentropy()

        self.accuracy = tf.keras.metrics.Accuracy()

        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()
        self.f1_score = tfa.metrics.F1Score(num_classes=out_shape, average='micro', threshold=0.5)

    def build(self):

        aggregator_type = self.aggregator_options['type']
        aggregator_shape = self.aggregator_options['shape']

        aggregator_activation = self.aggregator_options['activation']
        use_concat = self.aggregator_options['use_concat']

        attention_shape = self.aggregator_options['attention_shape']
        attention_activation = self.aggregator_options['attention_activation']

        self.aggregator_layers = []
        for idx in range(self.depth):
            aggregator_layer = Aggregator(aggregator_type, aggregator_activation, use_concat, attention_activation=attention_activation)

            if idx == 0:
                in_shape = self.in_shape
            else:
                in_shape = (use_concat+1)*aggregator_shape

            if aggregator_type == 'attention' and attention_shape is None:
                attention_shape = in_shape

            aggregator_layer.build(in_shape, aggregator_shape, attention_shape=attention_shape)
            self.aggregator_layers.append(aggregator_layer)

        self.output_layer = tf.keras.layers.Dense(self.out_shape, input_shape=((use_concat+1)*aggregator_shape,), name='output_layer')
        self.output_layer.build(((use_concat+1)*aggregator_shape,))

        super(GraphSAGE, self).build(())

    def call(self, feats, graph_size, neigh_indices, adj_mask, nneigh, labels, training=True):

        nrof_graphs = len(graph_size) # number of graphs in the batch
        graph_cumsize = np.insert(np.cumsum(graph_size), 0, 0)

        # initialize self_feats
        self_feats = feats[0][:graph_size[0]]
        for kdx in range(1, nrof_graphs):
             self_feats = tf.concat([self_feats, feats[kdx][:graph_size[kdx]]], axis=0)

        for idx in range(self.depth):
            # construct neigh_feats from self_feats
            for kdx in range(nrof_graphs):
                graph_self_feats = tf.gather(self_feats, tf.range(graph_cumsize[kdx], graph_cumsize[kdx+1]))
                graph_neigh_feats = tf.gather(graph_self_feats, neigh_indices[kdx][idx][:graph_size[kdx],:])

                # needed to exclude zero-padded nodes
                graph_adj_mask = adj_mask[kdx][idx][:graph_size[kdx],:]
                graph_neigh_feats = tf.multiply(tf.expand_dims(graph_adj_mask, axis=2), graph_neigh_feats)

                if kdx == 0:
                    neigh_feats = graph_neigh_feats
                else:
                    neigh_feats = tf.concat([neigh_feats, graph_neigh_feats], axis=0)

            nneigh_per_graph = []
            for jdx, nn in enumerate(nneigh[:, idx, :]):
                nneigh_per_graph.extend(nn[:graph_size[jdx]])

            self_feats = self.aggregator_layers[idx](self_feats, neigh_feats, nneigh_per_graph, training=training)
        self_feats = tf.math.l2_normalize(self_feats, axis=1)

        embedded_feats = self.output_layer(self_feats)
        predicted_labels = self.activation(embedded_feats)

        if len(labels.shape) > 1:
            batch_labels = labels[0][:graph_size[0],:]
            for kdx in range(1, nrof_graphs):
                batch_labels = tf.concat([batch_labels, labels[kdx][:graph_size[kdx],:]], axis=0)
        else:
            batch_labels = np.expand_dims(labels, 1)

        results = self.call_accuracy(predicted_labels, batch_labels)
        results.update(self.call_loss(predicted_labels, batch_labels, regularization=True))
        return results

    def call_accuracy(self, predicted_labels, labels):
        accuracy = self.accuracy(labels, predicted_labels)

        precision = self.precision(labels, predicted_labels)
        recall = self.recall(labels, predicted_labels)
        f1_score = self.f1_score(labels, predicted_labels)

        return {'precision': precision, 'recall': recall, 'f1_score': f1_score}

    def call_loss(self, predicted_labels, labels, regularization=True):
        loss = self.loss_function(labels, predicted_labels)

        if regularization:
            reg_loss = 0.0005 * tf.add_n([tf.nn.l2_loss(w) for w in self.trainable_variables])
            total_loss = loss + reg_loss
            return {'total_loss': loss + reg_loss, 'loss': loss, 'reg_loss': reg_loss}
        else:
            return {'total_loss': loss, 'loss': loss}


def generate_data_loader(dataset, num_parallel_calls=1, batch_size=1):
    """Create data loader to be fed to model"""

    data_slices = np.random.permutation(dataset.ngraphs)
    data_loader = tf.data.Dataset.from_tensor_slices((data_slices))

    data_loader = data_loader.map(**{'num_parallel_calls': num_parallel_calls, 'map_func': lambda x: dataset.sample(x, depth, nrof_neigh_per_batch)})
    data_loader = data_loader.batch(batch_size=batch_size)

    return data_loader


train = GraphDataset('datasets/data_ppi/train_ppi.pickle')
data_loader_train = generate_data_loader(train, num_parallel_calls=num_parallel_calls, batch_size=batch_size)

valid = GraphDataset('datasets/data_ppi/val_ppi.pickle')
data_loader_valid = generate_data_loader(valid, num_parallel_calls=num_parallel_calls, batch_size=batch_size)

model = GraphSAGE(train.nfeats, train.nlabels, activation, depth, nrof_neigh_per_batch, aggregator_options)
model.build()

model.summary()

# set optimizer
optimizer_class = getattr(tf.keras.optimizers, optimizer_type)
optimizer = optimizer_class(learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(**lr_schedule))

for epoch in range(total_epochs):
    print("Epoch %i"%epoch)

    for idx, data_batch in enumerate(data_loader_train):

        with tf.GradientTape() as tape:
            results = model(*data_batch, training=True)

        print(''.join(['%s: %f\t' % (key, value) for key, value in results.items()]))

        grads = tape.gradient(results['total_loss'], model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

    for idx, data_batch in enumerate(data_loader_valid):

        with tf.GradientTape() as tape:
            results = model(*data_batch, training=False)

        print(''.join(['%s_val: %f\t' % (key, value) for key, value in results.items()]))
    time.sleep(1)
