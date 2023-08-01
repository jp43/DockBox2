import numpy as np

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp

from dockbox2.layers import *
from dockbox2.utils import *

from dockbox2 import loss as db2loss

class GraphSAGE(tf.keras.models.Model):

    def __init__(self, in_shape, out_shape, depth, nrof_neigh, loss_options, aggregator_options, classifier_options, readout_options, \
        edge_options, attention_options=None, task_level=False):

        super(GraphSAGE, self).__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape

        self.depth = depth
        self.nrof_neigh = nrof_neigh

        self.aggregator_options = aggregator_options
        self.attention_options = attention_options

        self.edge_options = edge_options

        if task_level == 'graph':
            self.classifier_options = None
            self.readout_options = readout_options
            loss = 'loss_g'

        elif task_level == 'node':
            self.classifier_options = classifier_options
            self.readout_options = None
            loss = 'loss_n'
        else:
            raise ValueError("Task level %s not recognized! Should be node or graph")

        self.loss_function = self.build_classification_loss(loss_options[loss])
        self.loss_reg_w = loss_options['loss_reg']['weight']

        self.task_level = task_level

    def build_classification_loss(self, options):

        loss_type = options.pop('type')
        loss_function = getattr(db2loss, loss_type)

        return loss_function(**options)

    def build(self):

        aggregator_options = self.aggregator_options

        aggregator_type = aggregator_options.pop('type')
        aggregator_shape = aggregator_options.pop('shape')
        
        aggregator_activation = aggregator_options.pop('activation')
        use_concat = aggregator_options['use_concat']

        edge_options = self.edge_options

        edge_type = edge_options.pop('type')
        edge_activation =  edge_options.pop('activation')

        self.edge_layers = []
        self.aggregator_layers = []

        for idx in range(self.depth):
            if edge_type is not None:
                edge_layer = Edger(idx, edge_type, edge_activation, **edge_options)
            aggregator_layer = Aggregator(idx, aggregator_type, aggregator_activation, use_concat, gat_options=self.attention_options)

            if idx == 0:
                in_shape = self.in_shape
            else:
                in_shape = (use_concat+1)*aggregator_shape[idx-1]

            if edge_type is not None:
                edge_layer.build(in_shape)
                self.edge_layers.append(edge_layer)

            if aggregator_type == 'gat':
                if self.attention_options['shape'] is None:
                    gat_shape = in_shape
                else:
                    gat_shape = self.attention_options['shape'][idx]
            else:
                gat_shape = None

            aggregator_layer.build(in_shape, aggregator_shape[idx], gat_shape=gat_shape)
            self.aggregator_layers.append(aggregator_layer)

        if self.task_level == 'graph':
            readout_options = self.readout_options

            readout_type = readout_options.pop('type')
            readout_shape = readout_options.pop('shape')

            readout_activation = readout_options.pop('activation')
            readout_activation_h = readout_options.pop('activation_h')

            self.readout = GraphPooler('Readout', readout_type, readout_shape, readout_activation, readout_activation_h, **readout_options)
            self.readout.build(2*(use_concat+1)*aggregator_shape[-1])

        elif self.task_level == 'node':
            self.build_classifier((use_concat+1)*aggregator_shape[-1])
        else:
            raise ValueError("Task level %s not recognized! Should be node or graph")

        super(GraphSAGE, self).build(())

    def build_classifier(self, input_shape):

        classifier_options = self.classifier_options
        self.classifier = tf.keras.Sequential(name='Classifier')
        depth = len(classifier_options['shape'])

        for idx in range(depth):
            if idx + 1 == depth:
                activation = classifier_options['activation']
            else:
                activation = classifier_options['activation_h']

            if idx == 0:
                in_shape = input_shape
            else:
                in_shape = classifier_options['shape'][idx-1]

            self.classifier.add(tf.keras.layers.Dense(classifier_options['shape'][idx], input_shape=(in_shape,), use_bias=True, activation=activation))
        self.classifier.build((input_shape,))

    def call(self, feats, graph_size, neigh_indices, neigh_adj_values, neigh_rmsd, nneigh, node_labels, graph_labels, training=True):

        nrof_graphs = len(graph_size) # number of graphs in the minibatch
        graph_cumsize = np.insert(np.cumsum(graph_size), 0, 0)

        # initialize self_feats
        self_feats = feats[0][:graph_size[0]]

        for kdx in range(1, nrof_graphs):
             self_feats = tf.concat([self_feats, feats[kdx][:graph_size[kdx]]], axis=0)

        for idx in range(self.depth):
            # construct neigh_feats from self_feat
            for kdx in range(nrof_graphs):
                graph_self_feats = tf.gather(self_feats, tf.range(graph_cumsize[kdx], graph_cumsize[kdx+1]))
                graph_neigh_feats = tf.gather(graph_self_feats, neigh_indices[kdx][idx][:graph_size[kdx],:])

                graph_neigh_adj_values = neigh_adj_values[kdx][idx][:graph_size[kdx],:]
                graph_neigh_feats = tf.multiply(tf.expand_dims(graph_neigh_adj_values, axis=2), graph_neigh_feats)

                graph_neigh_rmsd = neigh_rmsd[kdx][idx][:graph_size[kdx],:]

                if kdx == 0:
                    neigh_feats = graph_neigh_feats
                    layer_neigh_rmsd = graph_neigh_rmsd
                else:
                    neigh_feats = tf.concat([neigh_feats, graph_neigh_feats], axis=0)
                    layer_neigh_rmsd = tf.concat([layer_neigh_rmsd, graph_neigh_rmsd], axis=0)

            nneigh_per_graph = []
            for jdx, nn in enumerate(nneigh[:, idx, :]):
                nneigh_per_graph.extend(nn[:graph_size[jdx]])

            if self.edge_layers:
                neigh_feats = self.edge_layers[idx](self_feats, neigh_feats, layer_neigh_rmsd, training=training)

            # update node features
            self_feats = self.aggregator_layers[idx](self_feats, neigh_feats, nneigh_per_graph, training=training)
            embedded_feats = tf.math.l2_normalize(self_feats, axis=1)

        if self.task_level == 'graph':
            batch_labels = tf.expand_dims(graph_labels, 1)

            # use classifier and readout
            batch_pred_labels = self.readout(embedded_feats, graph_size, training=training)
            (batch_best_node_labels, batch_pred_best_node_labels, batch_is_correct_labels) = (None, None, None)

        elif self.task_level == 'node':
            # pass values to classifier
            batch_pred_labels = self.classifier(embedded_feats, training=training)

            # extract batch labels
            if len(node_labels.shape) > 1:
                batch_labels = node_labels[0][:graph_size[0],:]
                for kdx in range(1, nrof_graphs):
                    batch_labels = tf.concat([batch_labels, node_labels[kdx][:graph_size[kdx],:]], axis=0)
            else:
                batch_labels = np.expand_dims(node_labels, 1)

            # extract predicted and ground-truth label of best nodes
            batch_pred_best_node_labels = np.zeros(nrof_graphs, dtype=np.float32)

            batch_best_node_labels = np.zeros(nrof_graphs, dtype=np.int32)
            batch_is_correct_labels = np.zeros(nrof_graphs, dtype=np.int32)

            for kdx in range(nrof_graphs):
                graph_batch_pred_labels = tf.gather(batch_pred_labels, tf.range(graph_cumsize[kdx], graph_cumsize[kdx+1]))
                graph_batch_labels = node_labels[kdx][:graph_size[kdx],:]

                best_node_idx = tf.math.argmax(graph_batch_pred_labels)
                batch_best_node_labels[kdx] = tf.gather(graph_batch_labels, best_node_idx)
                batch_pred_best_node_labels[kdx] = tf.gather(graph_batch_pred_labels, best_node_idx)

                batch_is_correct_labels[kdx] = tf.reduce_any(tf.equal(graph_batch_labels, 1))
 
            batch_pred_best_node_labels = tf.convert_to_tensor(batch_pred_best_node_labels[:,np.newaxis]) 
            batch_best_node_labels = tf.convert_to_tensor(batch_best_node_labels[:,np.newaxis])

            batch_is_correct_labels = tf.convert_to_tensor(batch_is_correct_labels[:,np.newaxis])

        return batch_labels, batch_pred_labels, batch_best_node_labels, batch_pred_best_node_labels, batch_is_correct_labels, \
            graph_size

    def call_loss(self, labels, pred_labels, regularization=True):

        loss = self.loss_function(labels, pred_labels)
        values = {'total_loss': loss, 'loss': loss}

        if regularization:
            reg_loss = self.loss_reg_w * 0.0005 * tf.add_n([tf.nn.l2_loss(w) for w in self.trainable_variables])

            values['reg_loss'] = reg_loss
            values['total_loss'] += reg_loss
        return values

    def success_rate(self, best_node_labels, pred_best_node_labels, is_correct_labels, threshold=0.5):

        pred_best_node_labels_i = tf.cast(tf.greater_equal(pred_best_node_labels[:, 0], threshold), tf.int32)
        pred_best_node_labels_i = tf.expand_dims(pred_best_node_labels_i, axis=1)

        correct_preds = tf.logical_and(tf.equal(is_correct_labels, pred_best_node_labels_i), \
                       tf.logical_or(tf.equal(is_correct_labels, 0), tf.equal(best_node_labels, 1)))

        nrof_correct_preds = tf.get_static_value(tf.math.count_nonzero(correct_preds))
        return nrof_correct_preds*100./len(is_correct_labels)

    def pearson(self, labels, pred_labels):
        return tfp.stats.correlation(labels, pred_labels, sample_axis=0, event_axis=1).numpy()[0][0]

    def save_weights_h5(self, filename):

        weight = self.get_weights()
        with h5py.File(filename, 'w') as h5f:

            for idx in range(len(weight)):
                h5f.create_dataset('weight'+str(idx), data=weight[idx])

    def load_weights_h5(self, filename):
        weight = []
        with h5py.File(filename, 'r') as h5f:

            for idx in range(len(h5f.keys())):
                weight.append(h5f['weight'+str(idx)].value)
        self.set_weights(weight)
