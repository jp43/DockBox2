import numpy as np

import tensorflow as tf
import tensorflow_addons as tfa

from dockbox2.aggregators import *

from dockbox2 import loss as db2loss
from dockbox2 import metrics as mt

class GraphSAGE(tf.keras.models.Model):

    def __init__(self, in_shape, out_shape, activation, depth, nrof_neigh, loss_options, aggregator_options, is_edge_feature=False, gat_options=None):
        super(GraphSAGE, self).__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape

        self.activation = activation
        if self.activation is not None:
            self.activation = getattr(tf.nn, activation)

        self.depth = depth
        self.nrof_neigh = nrof_neigh

        self.aggregator_options = aggregator_options

        self.is_edge_feature = is_edge_feature
        self.gat_options = gat_options

        # node-level loss functions
        self.loss_n_function = self.build_loss(loss_options['loss_n'])
        self.loss_reg_w = loss_options['loss_reg']['weight']

        # graph-level loss function
        self.loss_g_function = self.build_loss(loss_options['loss_g'])

        # performance metrics
        self.metrics_fn = {}
        for level in ['node', 'graph']:
            self.metrics_fn[level] = {}

            if self.out_shape == 1:
                # precision and recall for class 0
                self.metrics_fn[level]['precision_0'] = mt.ClassificationMetric(0, metric='precision', level=level)
                self.metrics_fn[level]['recall_0'] = mt.ClassificationMetric(0, metric='recall', level=level)

                # precision and recall for class 1
                self.metrics_fn[level]['precision_1'] = mt.ClassificationMetric(1, metric='precision', level=level)
                self.metrics_fn[level]['recall_1'] = mt.ClassificationMetric(1, metric='recall', level=level)

                # f1 score
                self.metrics_fn[level]['f1'] = tfa.metrics.F1Score(num_classes=out_shape, average='micro', threshold=0.5)
            else:
                # precision and recall
                self.metrics_fn[level]['precision'] = tf.keras.metrics.Precision()
                self.metrics_fn[level]['recall'] = tf.keras.metrics.Recall()

                self.metrics_fn[level]['f1'] = tfa.metrics.F1Score(num_classes=out_shape, average='micro', threshold=0.5)

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

        self.aggregator_layers = []
        for idx in range(self.depth):
            aggregator_layer = Aggregator(aggregator_type, aggregator_activation, use_concat, is_edge_feature=self.is_edge_feature, \
                                  gat_options=self.gat_options)

            if idx == 0:
                in_shape = self.in_shape
            else:
                in_shape = (use_concat+1)*aggregator_shape

            if aggregator_type.lower() == 'gat':
                if 'shape' not in self.gat_options:
                    gat_shape = in_shape
                else:
                    gat_shape = self.gat_options['shape']
            else:
                gat_shape = None

            aggregator_layer.build(in_shape, aggregator_shape, gat_shape=gat_shape)
            self.aggregator_layers.append(aggregator_layer)

        self.output_layer = tf.keras.layers.Dense(self.out_shape, input_shape=((use_concat+1)*aggregator_shape,), name='output_layer')
        self.output_layer.build(((use_concat+1)*aggregator_shape,))

        super(GraphSAGE, self).build(())


    def call(self, feats, graph_size, neigh_indices, adj_mask, edge_feats_mask, nneigh, labels, training=True):

        nrof_graphs = len(graph_size) # number of graphs in the minibatch
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
                graph_edge_feats = edge_feats_mask[kdx][idx][:graph_size[kdx],:]

                graph_neigh_feats = tf.multiply(tf.expand_dims(graph_adj_mask, axis=2), graph_neigh_feats)
                
                if kdx == 0:
                    neigh_feats = graph_neigh_feats
                    neigh_edge_feats = graph_edge_feats
                else:
                    neigh_feats = tf.concat([neigh_feats, graph_neigh_feats], axis=0)
                    neigh_edge_feats = tf.concat([neigh_edge_feats, graph_edge_feats], axis=0)

            nneigh_per_graph = []
            for jdx, nn in enumerate(nneigh[:, idx, :]):
                nneigh_per_graph.extend(nn[:graph_size[jdx]])

            self_feats = self.aggregator_layers[idx](self_feats, neigh_feats, neigh_edge_feats, nneigh_per_graph, training=training)
        self_feats = tf.math.l2_normalize(self_feats, axis=1)

        embedded_feats = self.output_layer(self_feats)
        batch_predicted_labels = self.activation(embedded_feats)

        if len(labels.shape) > 1:
            batch_labels = labels[0][:graph_size[0],:]
            for kdx in range(1, nrof_graphs):
                batch_labels = tf.concat([batch_labels, labels[kdx][:graph_size[kdx],:]], axis=0)
        else:
            batch_labels = np.expand_dims(labels, 1)

        batch_predicted_graph_labels = np.zeros(nrof_graphs, dtype=np.float32)

        batch_graph_labels = np.zeros(nrof_graphs, dtype=np.int32)
        batch_best_node_labels = np.zeros(nrof_graphs, dtype=np.int32) # labels of best nodes

        for kdx in range(nrof_graphs):
            graph_batch_predicted_labels = tf.gather(batch_predicted_labels, tf.range(graph_cumsize[kdx], graph_cumsize[kdx+1]))

            best_node_idx = tf.math.argmax(graph_batch_predicted_labels)
            batch_predicted_graph_labels[kdx] = tf.gather(graph_batch_predicted_labels, best_node_idx)
            graph_batch_labels = labels[kdx][:graph_size[kdx],:]

            batch_best_node_labels[kdx] = tf.gather(graph_batch_labels, best_node_idx)
            batch_graph_labels[kdx] = tf.reduce_any(tf.equal(graph_batch_labels, 1))

        batch_predicted_graph_labels = tf.convert_to_tensor(batch_predicted_graph_labels[:,np.newaxis])
        batch_graph_labels = tf.convert_to_tensor(batch_graph_labels[:,np.newaxis])

        batch_best_node_labels = tf.convert_to_tensor(batch_best_node_labels[:,np.newaxis])

        loss = self.call_loss(batch_labels, batch_predicted_labels, batch_graph_labels, batch_predicted_graph_labels, regularization=True)
        return batch_labels, batch_predicted_labels, batch_graph_labels, batch_predicted_graph_labels, batch_best_node_labels, loss


    def call_loss(self, labels, predicted_labels, graph_labels, predicted_graph_labels, regularization=True):

        loss_n = self.loss_n_function(labels, predicted_labels)
        loss_g = self.loss_g_function(graph_labels, predicted_graph_labels)

        if regularization:
            reg_loss = self.loss_reg_w * 0.0005 * tf.add_n([tf.nn.l2_loss(w) for w in self.trainable_variables])
            return {'total_loss': loss_n + loss_g + reg_loss, 'loss_n': loss_n, 'loss_g': loss_g, 'reg_loss': reg_loss}
        else:
            return {'total_loss': loss_n + loss_g, 'loss_n': loss_n, 'loss_g': loss_g}

    def call_metrics(self, labels, predicted_labels, level='node'):

        metrics_fn = self.metrics_fn[level]
        graph_suffix = '_g' if level == 'graph' else ''

        if self.out_shape == 1:
            precision_0 = metrics_fn['precision_0'](labels, predicted_labels)
            recall_0 = metrics_fn['recall_0'](labels, predicted_labels)

            precision_1 = metrics_fn['precision_1'](labels, predicted_labels)
            recall_1 = metrics_fn['recall_1'](labels, predicted_labels)

            f1_score = metrics_fn['f1'](labels, predicted_labels)

            return {'precision_0'+graph_suffix: precision_0, 'recall_0'+graph_suffix: recall_0, 'precision_1'+graph_suffix: precision_1,
               'recall_1'+graph_suffix: recall_1, 'f1'+graph_suffix: f1_score}

        else:
            precision = metrics_fn['precision'](labels, predicted_labels)
            recall = metrics_fn['recall'](labels, predicted_labels)

            f1_score = metrics_fn['f1'](labels, predicted_labels)

            return {'precision'+graph_suffix: precision, 'recall'+graph_suffix: recall, 'f1'+graph_suffix: f1_score}


    def success_rate(self, graph_labels, pred_graph_labels, best_node_labels, threshold=0.5):

        pred_graph_labels_i = tf.cast(tf.greater_equal(pred_graph_labels[:, 0], threshold), tf.int32)
        pred_graph_labels_i = tf.expand_dims(pred_graph_labels_i, axis=1)

        correct_preds = tf.logical_and(tf.equal(graph_labels, pred_graph_labels_i), \
                       tf.logical_or(tf.equal(graph_labels, 0), tf.logical_and(tf.equal(graph_labels, 1), tf.equal(pred_graph_labels_i, best_node_labels))))

        nrof_correct_preds = tf.get_static_value(tf.math.count_nonzero(correct_preds))
        return nrof_correct_preds*100./len(graph_labels)
