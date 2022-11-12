import numpy as np

import tensorflow as tf
import tensorflow_addons as tfa

from dockbox2.aggregators import *

from dockbox2 import loss as db2loss
from dockbox2 import metrics as db2mt

class GraphSAGE(tf.keras.models.Model):

    def __init__(self, in_shape, out_shape, activation, depth, nrof_neigh, loss_options, aggregator_options, is_edge_feature=False, attention_options=None):
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
        self.attention_options = attention_options

        # set up loss functions
        self.loss_n_function = self.build_loss(loss_options['loss_n'])
        self.loss_b_function = self.build_loss(loss_options['loss_b'])
        self.loss_reg_w = loss_options['loss_reg']['weight']

        # performance metrics
        if self.out_shape == 1:
            self.precis_0 = db2mt.ClassificationMetric(0, metric='precision')
            self.precis_1 = db2mt.ClassificationMetric(1, metric='precision')

            self.recall_0 = db2mt.ClassificationMetric(0, metric='recall')
            self.recall_1 = db2mt.ClassificationMetric(1, metric='recall')
        else:
            self.precision = tf.keras.metrics.Precision()
            self.recall = tf.keras.metrics.Recall()

        self.f1_score = tfa.metrics.F1Score(num_classes=out_shape, average='micro', threshold=0.5)

    def build_loss(self, options):

        loss_type = options.pop('type')
        if hasattr(db2loss, loss_type):
            loss_module = db2loss
        else:
            loss_module = tf.keras.losses
        loss_function = getattr(loss_module, loss_type)

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
attention_options=self.attention_options)

            if idx == 0:
                in_shape = self.in_shape
            else:
                in_shape = (use_concat+1)*aggregator_shape

            if aggregator_type == 'attention':
                if 'attention_shape' not in self.attention_options:
                    attention_shape = in_shape
                else:
                    attention_shape = self.attention_options['shape']
            else:
                attention_shape = None

            aggregator_layer.build(in_shape, aggregator_shape, attention_shape=attention_shape)
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

        # extract best node of each graph
        best_node_predicted_labels = np.zeros(nrof_graphs, dtype=np.float32)
        best_node_labels = np.zeros(nrof_graphs, dtype=np.int32)

        for kdx in range(nrof_graphs):
            graph_predicted_labels = tf.gather(batch_predicted_labels, tf.range(graph_cumsize[kdx], graph_cumsize[kdx+1]))
            best_node_idx = tf.math.argmax(graph_predicted_labels)

            best_node_predicted_labels[kdx] = tf.gather(graph_predicted_labels, best_node_idx)
            best_node_labels[kdx] = tf.gather(labels[kdx][:graph_size[kdx],:], best_node_idx)

        best_node_predicted_labels = tf.convert_to_tensor(best_node_predicted_labels[:,np.newaxis])
        best_node_labels = tf.convert_to_tensor(best_node_labels[:,np.newaxis])

        loss = self.call_loss(batch_labels, batch_predicted_labels, best_node_labels, best_node_predicted_labels, regularization=True)
        self.call_metrics(best_node_labels, best_node_predicted_labels)

        return batch_labels, batch_predicted_labels, best_node_labels, best_node_predicted_labels, loss

    def call_loss(self, labels, predicted_labels, best_node_labels, best_node_predicted_labels, regularization=True):

        loss_n = self.loss_n_function(labels, predicted_labels)
        loss_b = self.loss_b_function(best_node_labels, best_node_predicted_labels)

        if regularization:
            reg_loss = 0.0005 * self.loss_reg_w * tf.add_n([tf.nn.l2_loss(w) for w in self.trainable_variables])
            return {'total_loss': loss_n + loss_b + reg_loss, 'loss_n': loss_n, 'loss_b': loss_b, 'reg_loss': reg_loss}
        else:
            return {'total_loss': loss_n + loss_b, 'loss_n': loss_n, 'loss_b': loss_b}

    def call_metrics(self, labels, predicted_labels):

        if self.out_shape == 1:
            precis_0 = self.precis_0(labels, predicted_labels)
            precis_1 = self.precis_1(labels, predicted_labels)

            recall_0 = self.recall_0(labels, predicted_labels)
            recall_1 = self.recall_1(labels, predicted_labels)

            f1_score = self.f1_score(labels, predicted_labels)
            return {'precis_0': precis_0, 'precis_1': precis_1, 'recall_0': recall_0, 'recall_1': recall_1, 'f1': f1_score}
        else:
            precision = self.precision(labels, predicted_labels)
            recall = self.recall(labels, predicted_labels)

            f1_score = self.f1_score(labels, predicted_labels)
            return {'precision': precision, 'recall': recall, 'f1': f1_score}

