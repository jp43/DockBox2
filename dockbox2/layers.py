import sys

import numpy as np
import tensorflow as tf

class Edger(tf.keras.layers.Layer):

    def __init__(self, depth_idx, activation, depth, use_bias=False):

        name = 'Rmsd_extractor_' + str(depth_idx)
        super(Edger, self).__init__(name=name)

        self.depth = depth
        self.use_bias = use_bias

        self.activation = getattr(tf.nn, activation)

    def build(self, input_shape):

        self.dense_layers = tf.keras.Sequential()

        for idx in range(self.depth):
            if idx == 0:
                in_shape = input_shape + 1
            else:
                in_shape = input_shape
            self.dense_layers.add(tf.keras.layers.Dense(input_shape, input_shape=(in_shape,), use_bias=self.use_bias, activation=self.activation))

        self.dense_layers.build((input_shape+1, ))
        super(Edger, self).build(())

    def call(self, self_feats, neigh_feats, neigh_rmsd, training=True):

        input_feats = [neigh_feats]
        input_feats.append(tf.expand_dims(neigh_rmsd, axis=2))

        concat = tf.concat(input_feats, axis=2)

        # F was experiencing some problems if reshaping was not done
        neigh_feats = tf.reshape(self.dense_layers(tf.reshape(concat, [-1, int(concat.shape[-1])])), list(neigh_feats.shape))
        return neigh_feats


class Embedder(tf.keras.layers.Layer):

    def __init__(self, name, input_dim, node_features, activation='relu'):

        super(Embedder, self).__init__(name=name)

        self.input_dim = input_dim

        self.node_features = node_features
        self.activation = activation

    def build(self):

        self.embedding = tf.keras.layers.Embedding(self.input_dim, 1)
        self.embedding.build((1, ))

        super(Embedder, self).build(())

    def call(self, self_feats):

        instance_idx = self.node_features.index('instance')
        instances = self.embedding(self_feats[:, instance_idx])

        self_feats = tf.concat([self_feats[:,:instance_idx], instances, self_feats[:,instance_idx+1:]], axis=1)
        return self_feats


class Aggregator(tf.keras.layers.Layer):

    def __init__(self, depth_idx, type, activation, use_concat, gat_options, use_bias=True):

        name = type.capitalize() + '_agg_' + str(depth_idx) 
        super(Aggregator, self).__init__(name=name)

        self.type = type
        self.use_concat = use_concat

        self.activation = activation
        self.use_bias = use_bias

        if self.type == 'gat':
            self.gat_layer = GATLayer(gat_options['activation'])


    def build(self, input_shape, output_shape, gat_shape=None):

        if self.type == 'gat':
            self.gat_layer.build(input_shape, gat_shape)

        self.self_layer = tf.keras.layers.Dense(output_shape, input_shape=(input_shape,), use_bias=self.use_bias)
        self.self_layer.build((input_shape, ))

        if self.type in ['maxmean', 'meanmax']:
            in_shape = 2*input_shape 
        else:
            in_shape = input_shape

        self.neigh_layer = tf.keras.layers.Dense(output_shape, input_shape=(in_shape,), use_bias=self.use_bias)
        self.neigh_layer.build((in_shape, ))

        self.bn = tf.keras.layers.BatchNormalization()
        self.bn.build((None, (self.use_concat+1)*output_shape))

        self.activation_fn = getattr(tf.nn, self.activation)

        super(Aggregator, self).build(())


    def call(self, self_feats, neigh_feats, nneigh, training=True):

        if self.type == 'maxpool':
            aggregated_feats = tf.reduce_max(neigh_feats, axis=1)

        elif self.type == 'mean':
            nneigh = tf.where(tf.equal(nneigh, 0), tf.ones_like(nneigh), nneigh)
            aggregated_feats = tf.divide(tf.reduce_sum(neigh_feats, axis=1), tf.expand_dims(nneigh, 1))

        elif self.type == 'maxmean':
            max_feats = tf.reduce_max(neigh_feats, axis=1)

            nneigh = tf.where(tf.equal(nneigh, 0), tf.ones_like(nneigh), nneigh)
            mean_feats = tf.divide(tf.reduce_sum(neigh_feats, axis=1), tf.expand_dims(nneigh, 1))
            aggregated_feats = tf.concat([max_feats, mean_feats], axis=1)

        elif self.type == 'meanmax':
            nneigh = tf.where(tf.equal(nneigh, 0), tf.ones_like(nneigh), nneigh)
            mean_feats = tf.divide(tf.reduce_sum(neigh_feats, axis=1), tf.expand_dims(nneigh, 1))

            max_feats = tf.reduce_max(neigh_feats, axis=1)
            aggregated_feats = tf.concat([mean_feats, max_feats], axis=1)

        elif self.type == 'gat':
            attention_weights = self.gat_layer(self_feats, neigh_feats, training=training)
            aggregated_feats = tf.reduce_sum(tf.multiply(attention_weights, neigh_feats), axis=1)
        else:
            sys.exit("Unrecognized aggregator type %s"%self.type)

        self_feats = self.self_layer(self_feats)
        aggregated_feats = self.neigh_layer(aggregated_feats)

        if self.use_concat:
            self_feats = tf.concat([self_feats, aggregated_feats], axis=1)
        else:
            self_feats = tf.add_n([self_feats, aggregated_feats])

        self_feats = self.bn(self_feats, training=training)
        self_feats = self.activation_fn(self_feats)

        return self_feats


class GraphPooler(tf.keras.layers.Layer):

    def __init__(self, name, type, shape, activation, activation_h, use_bias=True):

        super(GraphPooler, self).__init__(name=name)

        self.type = type
        self.shape = shape

        self.activation = activation
        self.activation_h = activation_h

        self.use_bias = use_bias


    def build(self, input_shape):

        self.dense_layers = tf.keras.Sequential()
        depth = len(self.shape)

        for idx in range(depth):
            if idx + 1 == depth:
                activation = self.activation
            else:
                activation = self.activation_h

            if idx == 0:
                if self.type == 'meanmax':
                    in_shape = 2*input_shape
                else:
                    in_shape = input_shape
            else:
                in_shape = self.shape[idx-1]

            self.dense_layers.add(tf.keras.layers.Dense(self.shape[idx], input_shape=(in_shape,), use_bias=self.use_bias, activation=activation))

        self.dense_layers.build((input_shape,))
        super(GraphPooler, self).build(())


    def call(self, self_feats, graph_size, training=True):

        graph_cumsize = np.insert(np.cumsum(graph_size), 0, 0)
        nrof_graphs = len(graph_size) # number of graphs in the minibatch

        for kdx in range(nrof_graphs): 
            graph_self_feats = tf.gather(self_feats, tf.range(graph_cumsize[kdx], graph_cumsize[kdx+1]))

            if self.type == 'meanmax':
                mean_feats = tf.expand_dims(tf.reduce_mean(graph_self_feats, axis=0), 0)
                max_feats = tf.expand_dims(tf.reduce_max(graph_self_feats, axis=0), 0)

                if kdx == 0:
                    pooled_feats = tf.concat([mean_feats, max_feats], axis=1)
                else:
                    pooled_feats = tf.concat([pooled_feats, tf.concat([mean_feats, max_feats], axis=1)], axis=0)

            elif self.type == 'maxpool':
                max_feats = tf.expand_dims(tf.reduce_max(graph_self_feats, axis=0), 0)
                if kdx == 0:
                    pooled_feats = max_feats
                else:
                    pooled_feats = tf.concat([pooled_feats, max_feats], axis=0)

        return self.dense_layers(pooled_feats, training=training)


class GATLayer(tf.keras.layers.Layer):

    def __init__(self, activation):

        super(GATLayer, self).__init__()

        self.activation = getattr(tf.nn, activation)


    def build(self, input_shape, attention_shape):

        self.shared_layer = tf.keras.layers.Dense(attention_shape, input_shape=(input_shape,), use_bias=False)

        self.shared_layer.build((input_shape,))

        self.bn = tf.keras.layers.BatchNormalization()
        self.bn.build((None, attention_shape))

        self.output_gat_layer = tf.keras.layers.Dense(1, input_shape=(2*attention_shape,))

        self.output_gat_layer.build((2*attention_shape,))

        super(GATLayer, self).build(())


    def call(self, self_feats, neigh_feats, training=True):

        self_feats_shared = self.shared_layer(self_feats)
        self_feats_shared = self.bn(self_feats_shared, training=training)

        neigh_feats_shared = self.shared_layer(neigh_feats)

        neigh_feats_upd = self.bn(tf.reshape(neigh_feats_shared, [-1, int(neigh_feats_shared.shape[-1])]), training=training)

        neigh_feats_shared = tf.reshape(neigh_feats_upd, list(neigh_feats_shared.shape))
        concat = tf.concat([tf.stack([self_feats_shared]*tf.shape(neigh_feats_shared)[1].numpy(), axis=1), neigh_feats_shared], axis=2)

        weights = self.output_gat_layer(concat)
        weights = self.activation(weights)

        weights = tf.nn.softmax(tf.squeeze(weights, axis=2))

        weights = tf.expand_dims(weights, axis=2)

        return weights

