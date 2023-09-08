import numpy as np

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp

from dockbox2.layers import *
from dockbox2.utils import *

from dockbox2 import loss as db2loss

class GraphSAGE(tf.keras.models.Model):

    def __init__(self, in_shape, out_shape, depth, nrof_neigh, use_edger, loss_options, aggregator_options, classifier_options, readout_options, \
        node_options, attention_options=None, edger_options=None, task_level=False):

        super(GraphSAGE, self).__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape

        self.depth = depth
        self.nrof_neigh = nrof_neigh

        self.aggregator_options = aggregator_options
        self.attention_options = attention_options

        self.node_options = node_options
        self.node_features = node_options['features']

        self.use_edger = use_edger
        self.edger_options = edger_options

        if task_level == ['node']:
            self.classifier_options = classifier_options
            self.readout_options = None

            self.loss_n = self.build_loss(loss_options['loss_n'])
            self.loss_g = None

        elif task_level == ['graph']:
            self.classifier_options = None
            self.readout_options = readout_options

            self.loss_n = None
            self.loss_g = self.build_loss(loss_options['loss_g'])

        elif task_level == ['node', 'graph']:
            self.classifier_options = classifier_options
            self.readout_options = readout_options

            self.loss_n = self.build_loss(loss_options['loss_n'])
            self.loss_g = self.build_loss(loss_options['loss_g'])
        else:
            raise ValueError("Task level %s not recognized! Should be node and/or graph")

        self.loss_reg_w = loss_options['loss_reg']['weight']
        self.task_level = task_level

    def build_loss(self, options):

        loss_type = options.pop('type')
        loss_function = getattr(db2loss, loss_type)

        return loss_function(**options)

    def build(self):

        aggregator_options = self.aggregator_options

        aggregator_type = aggregator_options.pop('type')
        aggregator_shape = aggregator_options.pop('shape')
        
        aggregator_activation = aggregator_options.pop('activation')
        use_concat = aggregator_options['use_concat']
        use_bias = aggregator_options['use_bias']

        self.edger_layers = []
        if self.use_edger:
            edger_options = self.edger_options
            edger_activation = edger_options.pop('activation')
            edger_depth = edger_options.pop('depth')

        self.aggregator_layers = []

        for idx in range(self.depth):
            aggregator_layer = Aggregator(idx, aggregator_type, aggregator_activation, use_concat, self.attention_options, self.use_edger, use_bias=use_bias)

            if idx == 0:
                in_shape = self.in_shape
            else:
                in_shape = (use_concat+1)*aggregator_shape[idx-1]

            if self.use_edger:
                edger_layer = Edger(idx, edger_activation, edger_depth, **edger_options)

                edger_layer.build(in_shape)
                self.edger_layers.append(edger_layer)

            if aggregator_type == 'gat':
                if self.attention_options['shape'] is None:
                    gat_shape = in_shape
                else:
                    gat_shape = self.attention_options['shape'][idx]
            else:
                gat_shape = None

            aggregator_layer.build(in_shape, aggregator_shape[idx], gat_shape=gat_shape)
            self.aggregator_layers.append(aggregator_layer)

        if 'graph' in self.task_level:
            readout_options = self.readout_options

            readout_type = readout_options.pop('type')
            readout_shape = readout_options.pop('shape')

            readout_activation = readout_options.pop('activation')
            readout_activation_h = readout_options.pop('activation_h')

            self.readout = GraphPooler('Readout', readout_type, readout_shape, readout_activation, readout_activation_h, **readout_options)
            self.readout.build((use_concat+1)*aggregator_shape[-1])

        if 'node' in self.task_level:
            self.build_classifier((use_concat+1)*aggregator_shape[-1])

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

        (batch_pred_node_labels, batch_best_node_labels, batch_pred_best_node_labels, batch_is_correct_labels, batch_pred_graph_labels) = \
   (None, None, None, None, None)

        nrof_graphs = len(graph_size) # number of graphs in the minibatch
        graph_cumsize = np.insert(np.cumsum(graph_size), 0, 0)

        # initialize self_feats
        self_feats = feats[0][:graph_size[0]]

        for kdx in range(1, nrof_graphs):
             self_feats = tf.concat([self_feats, feats[kdx][:graph_size[kdx]]], axis=0)

        for idx in range(self.depth):
            self_nneigh = nneigh[0, idx, :graph_size[0]]
            for kdx in range(1, nrof_graphs):
                self_nneigh = tf.concat([self_nneigh, nneigh[kdx, idx, :graph_size[kdx]]], axis=0)

            for kdx in range(nrof_graphs):
                graph_self_feats = tf.gather(self_feats, tf.range(graph_cumsize[kdx], graph_cumsize[kdx+1]))
                graph_neigh_feats = tf.gather(graph_self_feats, neigh_indices[kdx][idx][:graph_size[kdx],:])

                graph_neigh_adj_values = neigh_adj_values[kdx][idx][:graph_size[kdx],:]
                graph_neigh_feats = tf.multiply(tf.expand_dims(graph_neigh_adj_values, axis=2), graph_neigh_feats)

                graph_neigh_rmsd = neigh_rmsd[kdx][idx][:graph_size[kdx],:]

                graph_self_nneigh = tf.gather(self_nneigh, tf.range(graph_cumsize[kdx], graph_cumsize[kdx+1]))
                graph_neigh_nneigh = tf.gather(graph_self_nneigh, neigh_indices[kdx][idx][:graph_size[kdx],:])
                graph_neigh_nneigh = tf.multiply(graph_neigh_adj_values, graph_neigh_nneigh)

                if kdx == 0:
                    neigh_feats = graph_neigh_feats

                    edger_neigh_rmsd = graph_neigh_rmsd
                    neigh_nneigh = graph_neigh_nneigh
                else:
                    neigh_feats = tf.concat([neigh_feats, graph_neigh_feats], axis=0)

                    edger_neigh_rmsd = tf.concat([edger_neigh_rmsd, graph_neigh_rmsd], axis=0)
                    neigh_nneigh = tf.concat([neigh_nneigh, graph_neigh_nneigh], axis=0)

            if self.use_edger:
                neigh_feats = self.edger_layers[idx](self_feats, neigh_feats, edger_neigh_rmsd, training=training)
            else: # concat with edge feature at each layer
                neigh_feats = tf.concat([neigh_feats, tf.expand_dims(edger_neigh_rmsd, axis=2)], axis=2)

            # update node features
            self_feats = self.aggregator_layers[idx](self_feats, neigh_feats, self_nneigh, neigh_nneigh, training=training)
            embedded_feats = tf.math.l2_normalize(self_feats, axis=1)

        if 'node' in self.task_level:
            # pass values to classifier
            batch_pred_node_labels = self.classifier(embedded_feats, training=training)

            # extract batch labels
            if len(node_labels.shape) > 1:
                batch_node_labels = node_labels[0][:graph_size[0],:]
                for kdx in range(1, nrof_graphs):
                    batch_node_labels = tf.concat([batch_node_labels, node_labels[kdx][:graph_size[kdx],:]], axis=0)
            else:
                batch_node_labels = np.expand_dims(node_labels, 1)

            # extract predicted and ground-truth label of best nodes
            batch_pred_best_node_labels = np.zeros(nrof_graphs, dtype=np.float32)

            batch_best_node_labels = np.zeros(nrof_graphs, dtype=np.int32)
            batch_is_correct_labels = np.zeros(nrof_graphs, dtype=np.int32)

            for kdx in range(nrof_graphs):
                batch_pred_node_labels_cg = tf.gather(batch_pred_node_labels, tf.range(graph_cumsize[kdx], graph_cumsize[kdx+1]))
                batch_node_labels_cg = node_labels[kdx][:graph_size[kdx],:] # node labels of current graph (_cg)

                best_node_idx = tf.math.argmax(batch_pred_node_labels_cg)
                batch_best_node_labels[kdx] = tf.gather(batch_node_labels_cg, best_node_idx)
                batch_pred_best_node_labels[kdx] = tf.gather(batch_pred_node_labels_cg, best_node_idx)

                batch_is_correct_labels[kdx] = tf.reduce_any(tf.equal(batch_node_labels_cg, 1))
 
            batch_pred_best_node_labels = tf.convert_to_tensor(batch_pred_best_node_labels[:,np.newaxis]) 
            batch_best_node_labels = tf.convert_to_tensor(batch_best_node_labels[:,np.newaxis])

            batch_is_correct_labels = tf.convert_to_tensor(batch_is_correct_labels[:,np.newaxis])

        if 'graph' in self.task_level:
            batch_graph_labels = tf.expand_dims(graph_labels, 1)

            # use classifier and readout
            batch_pred_graph_labels = self.readout(embedded_feats, graph_size, training=training)

        return batch_node_labels, batch_pred_node_labels, batch_best_node_labels, batch_pred_best_node_labels, batch_is_correct_labels, \
            batch_graph_labels, batch_pred_graph_labels, graph_size

    def call_loss(self, node_labels, pred_node_labels, graph_labels, pred_graph_labels, regularization=True, use_gradnorm=False):

        values = {'total_loss': 0}
        if 'node' in self.task_level:
            loss_n = self.loss_n(node_labels, pred_node_labels)

            values['total_loss'] += loss_n
            values['loss_n'] = loss_n

        if 'graph' in self.task_level:
            loss_g = self.loss_g(graph_labels, pred_graph_labels)

            values['total_loss'] += loss_g
            values['loss_g'] = loss_g

        if regularization:
            reg_loss = self.loss_reg_w * 0.0005 * tf.add_n([tf.nn.l2_loss(w) for w in self.trainable_variables])
            values['reg_loss'] = reg_loss

        values['total_loss'] += reg_loss
        return values

    def get_shared_weights(self, trainable=True):
        """In GradNorm paper, it is said that shared weights should be the weights of the last shared layer... To be checked."""
        shared_weights = []
        for aggregator_layer in self.aggregator_layers:
            if trainable:
                shared_weights.extend(aggregator_layer.trainable_weights)
            else:
                shared_weights.extend(aggregator_layer.weights)

        for edger_layer in self.edger_layers:
            if trainable:
                shared_weights.extend(edger_layer.trainable_weights)
            else:   
                shared_weights.extend(edger_layer.weights)
        return shared_weights

    def gradient_normalization(self, tape, epoch, loss, trainable=True):

        if epoch == 0:
            self.ln_0 = loss['loss_n']
            self.lg_0 = loss['loss_g']

        shared_weights = self.get_shared_weights(trainable=trainable)

        grads_n = tape.gradient(loss['loss_n'], shared_weights)
        norm_grads_n = tf.norm(tf.concat([tf.reshape(g, [-1]) for g in grads_n], axis=0), ord=2)

        grads_g = tape.gradient(loss['loss_g'], shared_weights)
        norm_grads_g = tf.norm(tf.concat([tf.reshape(g, [-1]) for g in grads_g], axis=0), ord=2)

        G_avg = tf.div(tf.add(norm_grads_n, norm_grads_g), 2)

        l_tilda_n = tf.div(loss['loss_n'], self.ln_0)
        l_tilda_g = tf.div(loss['loss_g'], self.lg_0)
        l_tilda_avg = tf.div(tf.add(l_tilda_n, l_tilda_g), 2)

        inv_rate_n = tf.div(l_tilda_n, l_tilda_avg)
        inv_rate_g = tf.div(l_tilda_g, l_tilda_avg)

        cn = tf.multiply(G_avg, tf.pow(inv_rate_n, a))
        cg = tf.multiply(G_avg, tf.pow(inv_rate_g, a))
 
        loss_gradnorm = tf.add(tf.reduce_sum(tf.abs(tf.subtract(norm_grads_n, cn))), \
            tf.reduce_sum(tf.abs(tf.subtract(norm_grads_g, cg))))

        return loss_gradnorm

    def success_rate(self, best_node_labels, pred_best_node_labels, is_correct_labels, threshold=0.5):

        pred_best_node_labels_i = tf.cast(tf.greater_equal(pred_best_node_labels[:, 0], threshold), tf.int32)
        pred_best_node_labels_i = tf.expand_dims(pred_best_node_labels_i, axis=1)

        correct_preds = tf.logical_and(tf.equal(is_correct_labels, pred_best_node_labels_i), \
                       tf.logical_or(tf.equal(is_correct_labels, 0), tf.equal(best_node_labels, 1)))

        nrof_correct_preds = tf.get_static_value(tf.math.count_nonzero(correct_preds))
        return nrof_correct_preds*100./len(is_correct_labels)

    def pearson(self, labels, pred_labels):
        return tfp.stats.correlation(labels, pred_labels, sample_axis=0, event_axis=1).numpy()[0][0]

    def r_squared_value(self, labels, pred_labels):
        y_mean = tf.reduce_mean(labels)

        ss_total = tf.reduce_sum(tf.square(labels - y_mean))
        ss_res = tf.reduce_sum(tf.square(labels - pred_labels))

        r_squared = 1.0 - (ss_res / ss_total)
        return r_squared

    def save_weights_h5(self, filename):
        weights = self.get_weights()

        with h5py.File(filename, 'w') as h5f:
            for idx, weight in enumerate(weights):
                h5f.create_dataset('weight'+str(idx), data=weight)

    def load_weights_h5(self, filename):
        weights = []
        with h5py.File(filename, 'r') as h5f:

            for idx in range(len(h5f.keys())):
                weights.append(h5f['weight'+str(idx)][()])
        self.set_weights(weights)
