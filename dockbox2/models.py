import numpy as np

import tensorflow as tf
import tensorflow_addons as tfa

from dockbox2.layers import *
from dockbox2.utils import *

from dockbox2 import loss as db2loss
from dockbox2 import metrics as mt

class GraphSAGE(tf.keras.models.Model):

    def __init__(self, in_shape, out_shape, depth, nrof_neigh, loss_options, aggregator_options, classifier_options, edge_options):

        super(GraphSAGE, self).__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape

        self.depth = depth
        self.nrof_neigh = nrof_neigh

        self.aggregator_options = aggregator_options

        self.classifier_options = classifier_options
        self.edge_options = edge_options

        # node-level loss functions
        self.loss_n_function = self.build_loss(loss_options['loss_n'])
        self.loss_reg_w = loss_options['loss_reg']['weight']

        if loss_options['loss_g']['weight'] > .0:
            self.loss_g_function = self.build_loss(loss_options['loss_g'])
        else:
            self.loss_g_function = None

        self.metrics_fn = {}
        for level in ['node', 'graph']:
            self.metrics_fn[level] = {}

            graph_suffix = '_graph' if level == 'graph' else ''
            if self.out_shape == 1:
                self.metrics_fn[level]['pr0'] = mt.ClsMetric(0, metric='precision', level=level)
                self.metrics_fn[level]['rc0'] = mt.ClsMetric(0, metric='recall', level=level)

                self.metrics_fn[level]['pr1'] = mt.ClsMetric(1, metric='precision', level=level)
                self.metrics_fn[level]['rc1'] = mt.ClsMetric(1, metric='recall', level=level)

                self.metrics_fn[level]['f1'] = tfa.metrics.F1Score(num_classes=out_shape, average='micro', threshold=0.5, \
                                                                   name=abbreviation['f1_score']+graph_suffix)
            else:
                self.metrics_fn[level]['pr'] = tf.keras.metrics.Precision()
                self.metrics_fn[level]['rc'] = tf.keras.metrics.Recall()

                self.metrics_fn[level]['f1'] = tfa.metrics.F1Score(num_classes=out_shape, average='micro', threshold=0.5, \
                                               name=abbreviation['f1_score']+graph_suffix)

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

        edge_options = self.edge_options
        edge_type = edge_options.pop('type')
        edge_activation =  edge_options.pop('activation')
        
        self.edge_layers = []
        self.aggregator_layers = []

        for idx in range(self.depth):
            if edge_type is not None:
                edge_layer = Edger(idx, edge_type, edge_activation, **edge_options)
            aggregator_layer = Aggregator(idx, aggregator_type, aggregator_activation, use_concat)

            if idx == 0:
                in_shape = self.in_shape
            else:
                in_shape = (use_concat+1)*aggregator_shape[idx-1]

            if edge_type is not None:
                edge_layer.build(in_shape)
                self.edge_layers.append(edge_layer)

            aggregator_layer.build(in_shape, aggregator_shape[idx])
            self.aggregator_layers.append(aggregator_layer)

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


    def call(self, feats, cogs, graph_size, neigh_indices, neigh_adj_values, neigh_rmsd, nneigh, labels, bm_xyz, training=True):

        nrof_graphs = len(graph_size) # number of graphs in the minibatch
        graph_cumsize = np.insert(np.cumsum(graph_size), 0, 0)

        # initialize self_feats
        self_feats = feats[0][:graph_size[0]]
        self_cogs = cogs[0][:graph_size[0]]

        for kdx in range(1, nrof_graphs):
             self_feats = tf.concat([self_feats, feats[kdx][:graph_size[kdx]]], axis=0)
             self_cogs = tf.concat([self_cogs, cogs[kdx][:graph_size[kdx]]], axis=0)

        for idx in range(self.depth):
            # construct neigh_feats from self_feat
            for kdx in range(nrof_graphs):

                graph_self_feats = tf.gather(self_feats, tf.range(graph_cumsize[kdx], graph_cumsize[kdx+1]))
                graph_self_cogs = tf.gather(self_cogs, tf.range(graph_cumsize[kdx], graph_cumsize[kdx+1]))

                graph_neigh_feats = tf.gather(graph_self_feats, neigh_indices[kdx][idx][:graph_size[kdx],:])
                graph_neigh_cogs = tf.gather(graph_self_cogs, neigh_indices[kdx][idx][:graph_size[kdx],:])

                graph_neigh_adj_values = neigh_adj_values[kdx][idx][:graph_size[kdx],:]

                graph_neigh_rmsd = neigh_rmsd[kdx][idx][:graph_size[kdx],:]
                graph_neigh_rmsd = tf.multiply(graph_neigh_adj_values, graph_neigh_rmsd)

                graph_neigh_feats = tf.multiply(tf.expand_dims(graph_neigh_adj_values, axis=2), graph_neigh_feats)

                # compute relative COGs (Xj - Xi)
                graph_neigh_cogs = tf.subtract(graph_neigh_cogs, tf.stack([graph_self_cogs]*tf.shape(graph_neigh_cogs)[1].numpy(), axis=1))
                graph_neigh_cogs = tf.multiply(tf.expand_dims(graph_neigh_adj_values, axis=2), graph_neigh_cogs)

                if kdx == 0:
                    neigh_feats = graph_neigh_feats
                    neigh_cogs = graph_neigh_cogs
                    layer_neigh_rmsd = graph_neigh_rmsd
                else:
                    neigh_feats = tf.concat([neigh_feats, graph_neigh_feats], axis=0)
                    neigh_cogs = tf.concat([neigh_cogs, graph_neigh_cogs], axis=0)
                    layer_neigh_rmsd = tf.concat([layer_neigh_rmsd, graph_neigh_rmsd], axis=0)

            nneigh_per_graph = []
            for jdx, nn in enumerate(nneigh[:, idx, :]):
                nneigh_per_graph.extend(nn[:graph_size[jdx]])

            if self.edge_layers:
                neigh_feats = self.edge_layers[idx](self_feats, neigh_feats, neigh_cogs, layer_neigh_rmsd, training=training)

            # update node features
            self_feats = self.aggregator_layers[idx](self_feats, neigh_feats, nneigh_per_graph, training=training)

        embedded_feats = tf.math.l2_normalize(self_feats, axis=1)
        batch_predicted_labels = self.classifier(embedded_feats, training=training)

        # extract batch labels
        if len(labels.shape) > 1:
            batch_labels = labels[0][:graph_size[0],:]
            for kdx in range(1, nrof_graphs):
                batch_labels = tf.concat([batch_labels, labels[kdx][:graph_size[kdx],:]], axis=0)
        else:
            batch_labels = np.expand_dims(labels, 1)

        # extract predicted and ground-truth label of best nodes
        batch_predicted_graph_labels = np.zeros(nrof_graphs, dtype=np.float32)

        batch_graph_labels = np.zeros(nrof_graphs, dtype=np.int32)
        batch_best_node_labels = np.zeros(nrof_graphs, dtype=np.int32)

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

        return batch_labels, batch_predicted_labels, batch_graph_labels, batch_predicted_graph_labels, batch_best_node_labels, graph_size


    def call_loss(self, labels, predicted_labels, graph_labels, predicted_graph_labels, regularization=True):

        loss_n = self.loss_n_function(labels, predicted_labels)
        values = {'total_loss': loss_n, 'loss_n': loss_n}

        if self.loss_g_function is not None:
            loss_g = self.loss_g_function(graph_labels, predicted_graph_labels)

            values['loss_g'] = loss_g 
            values['total_loss'] += loss_g

        if regularization:
            reg_loss = self.loss_reg_w * 0.0005 * tf.add_n([tf.nn.l2_loss(w) for w in self.trainable_variables])

            values['reg_loss'] = reg_loss
            values['total_loss'] += reg_loss

        return values


    def call_metrics(self, labels, predicted_labels, level='node'):

        metrics_fn = self.metrics_fn[level]
        graph_suffix = '_graph' if level == 'graph' else ''

        if self.out_shape == 1:
            precision_0 = metrics_fn['pr0'](labels, predicted_labels)
            recall_0 = metrics_fn['rc0'](labels, predicted_labels)

            precision_1 = metrics_fn['pr1'](labels, predicted_labels)
            recall_1 = metrics_fn['rc1'](labels, predicted_labels)

            f1_score = metrics_fn['f1'](labels, predicted_labels)

            return {'pr0'+graph_suffix: precision_0, 'rc0'+graph_suffix: recall_0, 'pr1'+graph_suffix: precision_1, \
                'rc1'+graph_suffix: recall_1, 'f1'+graph_suffix: f1_score}

        else:
            precision = metrics_fn['precision'](labels, predicted_labels)
            recall = metrics_fn['recall'](labels, predicted_labels)

            f1_score = metrics_fn['f1'](labels, predicted_labels)

            return {'pr'+graph_suffix: precision, 'rc'+graph_suffix: recall, 'f1'+graph_suffix: f1_score}


    def success_rate(self, graph_labels, pred_graph_labels, best_node_labels, threshold=0.5):

        pred_graph_labels_i = tf.cast(tf.greater_equal(pred_graph_labels[:, 0], threshold), tf.int32)
        pred_graph_labels_i = tf.expand_dims(pred_graph_labels_i, axis=1)

        correct_preds = tf.logical_and(tf.equal(graph_labels, pred_graph_labels_i), \
                       tf.logical_or(tf.equal(graph_labels, 0), tf.logical_and(tf.equal(graph_labels, 1), tf.equal(pred_graph_labels_i, best_node_labels))))

        nrof_correct_preds = tf.get_static_value(tf.math.count_nonzero(correct_preds))

        return nrof_correct_preds*100./len(graph_labels)
