import numpy as np

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp

from sklearn.metrics import roc_curve, auc

from dockbox2.layers import *
from dockbox2.utils import *

from dockbox2 import loss as db2loss

class GraphSAGE(tf.keras.models.Model):

    def __init__(self, in_shape, out_shape, depth, nrof_neigh, use_edger, loss_options, aggregator_options, classifier_options, readout_options, \
        node_options, attention_options=None, edger_options=None, task_level=False, weighting=None):

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
        self.loss_options = loss_options

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

            if weighting != 'uw':
                self.loss_n = self.build_loss(loss_options['loss_n'])
                self.loss_g = self.build_loss(loss_options['loss_g'])

        else:
            raise ValueError("Task level %s not recognized! Should be node and/or graph")

        self.loss_reg_w = loss_options['loss_reg']['weight']
        self.task_level = task_level
        self.weighting = weighting

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

        if 'node' in self.task_level and 'graph' in self.task_level and self.weighting == 'uw':
            loss_n_options = self.loss_options['loss_n']
            alpha = loss_n_options['alpha']
            gamma = loss_n_options['gamma']

            self.loss_uw = MultiLossLayer(alpha=alpha, gamma=gamma)
            self.loss_uw.build()
        else:
            self.loss_uw = None

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

        # extract batch labels
        if len(node_labels.shape) > 1:
            batch_node_labels = node_labels[0][:graph_size[0],:]
            for kdx in range(1, nrof_graphs):
                batch_node_labels = tf.concat([batch_node_labels, node_labels[kdx][:graph_size[kdx],:]], axis=0)
        else:
            batch_node_labels = np.expand_dims(node_labels, 1)

        batch_graph_labels = tf.expand_dims(graph_labels, 1)

        if 'node' in self.task_level:
            # pass values to classifier
            batch_pred_node_labels = self.classifier(embedded_feats, training=training)

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
            # use classifier and readout
            batch_pred_graph_labels = self.readout(embedded_feats, graph_size, training=training)

        return batch_node_labels, batch_pred_node_labels, batch_best_node_labels, batch_pred_best_node_labels, batch_is_correct_labels, \
            batch_graph_labels, batch_pred_graph_labels, graph_size


    def call_loss(self, node_labels, pred_node_labels, graph_labels, pred_graph_labels, regularization=True):

        values = {}
        if self.task_level == ['node', 'graph'] and self.weighting == 'uw':

            loss_n, loss_g = self.loss_uw(node_labels, pred_node_labels, graph_labels, pred_graph_labels)
            values = {'total_loss': loss_n + loss_g, 'loss_n': loss_n, 'loss_g': loss_g}

        elif self.task_level == ['node', 'graph'] and self.weighting == 'rlw':

            weights_rlw = tf.nn.softmax(tf.random.normal((2, )))

            loss_n = weights_rlw[0] * self.loss_n(node_labels, pred_node_labels)
            loss_g = weights_rlw[1] * self.loss_g(graph_labels, pred_graph_labels)

            values['total_loss'] = loss_n + loss_g
            values['loss_n'] = loss_n
            values['loss_g'] = loss_g

        else:
            values['total_loss'] = 0
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


    def confusion_matrix(self, best_node_labels, pred_best_node_labels, is_correct_labels, threshold=0.5):

        pred_best_node_labels_i = tf.cast(tf.greater_equal(pred_best_node_labels[:, 0], threshold), tf.int32)
        pred_best_node_labels_i = tf.expand_dims(pred_best_node_labels_i, axis=1)

        # the model correctly predicted the native pose
        tp = tf.logical_and(tf.equal(pred_best_node_labels_i, 1), tf.equal(best_node_labels, 1))
        tp = tf.get_static_value(tf.math.count_nonzero(tp))

        # the model predicted a pose as native but the real one does not exist OR is different from the one predicted
        fp = tf.logical_and(tf.equal(pred_best_node_labels_i, 1), tf.equal(best_node_labels, 0))
        fp = tf.get_static_value(tf.math.count_nonzero(fp))

        # the model correctly predicted no native poses
        tn = tf.logical_and(tf.equal(pred_best_node_labels_i, 0), tf.equal(is_correct_labels, 0))
        tn = tf.get_static_value(tf.math.count_nonzero(tn))

        # the model predicted no native poses but the real one exists among the docked poses
        fn = tf.logical_and(tf.equal(pred_best_node_labels_i, 0), tf.equal(is_correct_labels, 1))

        fn = tf.get_static_value(tf.math.count_nonzero(fn))
        return tp, fp, tn, fn

    def success_rate(self, best_node_labels, pred_best_node_labels, is_correct_labels):

        tp, fp, tn, fn = self.confusion_matrix(best_node_labels, pred_best_node_labels, is_correct_labels, threshold=0.0)
        return tp * 100./(tp + fp)

    def roc_metrics(self, best_node_labels, pred_best_node_labels, is_correct_labels, num=1000):

        threshold = np.linspace(0, 1, num)
        tpr, fpr, accuracy = ([], [], [])

        for kdx, th in enumerate(threshold):
            tp, fp, tn, fn = self.confusion_matrix(best_node_labels, pred_best_node_labels, is_correct_labels, threshold=th)
            accuracy.append((tp + tn) * 100./(tp + tn + fp + fn))

            tpr.append(tp * 1./(tp + fn))
            fpr.append(fp * 1./(fp + tn))

        #best_threshold_idx = np.argmax(np.array(tpr)-np.array(fpr))
        best_threshold_idx = np.argmax(accuracy)

        best_tpr = tpr[best_threshold_idx]
        best_fpr = fpr[best_threshold_idx]
        best_accuracy = accuracy[best_threshold_idx]

        auc_value = auc(fpr, tpr)
        best_threshold = threshold[best_threshold_idx]

        return best_tpr, 1 - best_fpr, best_accuracy, auc_value, best_threshold

    def roc_metrics_graph(self, labels, pred_labels):

        fpr, tpr, threshold = roc_curve(labels[:, 0], pred_labels[:, 0])
        auc_value = auc(fpr, tpr)

        best_threshold_idx = np.argmax(tpr - fpr)
        best_threshold = threshold[best_threshold_idx]
        
        pred_labels_i = tf.cast(tf.greater_equal(pred_labels[:, 0], best_threshold), tf.int32)
        labels_i = tf.cast(labels[:,0], tf.int32)

        tp = tf.math.multiply(pred_labels_i, labels_i)
        tp = tf.get_static_value(tf.math.count_nonzero(tp))

        fp = tf.math.multiply(pred_labels_i, 1 - labels_i)
        fp = tf.get_static_value(tf.math.count_nonzero(fp))

        tn = tf.math.multiply(1 - pred_labels_i, 1 - labels_i)
        tn = tf.get_static_value(tf.math.count_nonzero(tn))

        fn = tf.math.multiply(1 - pred_labels_i, labels_i)
        fn = tf.get_static_value(tf.math.count_nonzero(fn))

        best_tpr = tp * 1./ (tp + fn)
        best_fpr = fp * 1./ (fp + tn)

        return best_tpr, 1 - best_fpr, auc_value, best_threshold

    def pearson(self, labels, pred_labels):
        return tfp.stats.correlation(labels, pred_labels, sample_axis=0, event_axis=1).numpy()[0][0]

    def r_squared_value(self, labels, pred_labels):
        y_mean = tf.reduce_mean(labels)

        ss_total = tf.reduce_sum(tf.square(labels - y_mean))
        ss_res = tf.reduce_sum(tf.square(labels - pred_labels))

        r_squared = 1.0 - (ss_res / ss_total)
        return r_squared

    def rmse(self, labels, pred_labels):
        return tf.sqrt(tf.reduce_mean((labels - pred_labels)**2, axis=0))

    def std(self, labels, pred_labels):
        return tf.math.reduce_std(labels - pred_labels)

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
