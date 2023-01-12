import sys

import math
import tensorflow as tf

class Edger(tf.keras.layers.Layer):

    def __init__(self, depth_idx, type, activation, depth):

        name = '_'.join(type).capitalize() + '_extractor_' + str(depth_idx)
        super(Edger, self).__init__(name=name)

        self.type = type
        self.depth = depth

        self.activation = activation

    def build(self, input_shape):

        nfeats = 0
        self.layer = tf.keras.Sequential()

        for idx in range(self.depth):
            if idx == 0:
                if 'cog' in self.type:
                    nfeats += 3
                if 'rmsd' in self.type:
                    nfeats += 1
                in_shape = input_shape + nfeats
            else:
                in_shape = input_shape

            self.layer.add(tf.keras.layers.Dense(input_shape, input_shape=(in_shape,), use_bias=False, activation=self.activation))

        self.layer.build((input_shape+1, ))
        super(Edger, self).build(())

    def call(self, self_feats, neigh_feats, neigh_cogs, neigh_rmsd, training=True):

        input_feats = [neigh_feats]

        if 'cog' in self.type:
            input_feats.append(neigh_cogs)

        if 'rmsd' in self.type:
            input_feats.append(tf.expand_dims(neigh_rmsd, axis=2))

        concat = tf.concat(input_feats, axis=2)
        # F was experiencing some problems if reshaping was not done
        neigh_feats = tf.reshape(self.layer(tf.reshape(concat, [-1, int(concat.shape[-1])])), list(neigh_feats.shape))

        return neigh_feats


class Aggregator(tf.keras.layers.Layer):

    def __init__(self, depth_idx, type, activation, use_concat, gat_options):

        name = type.capitalize() + '_agg_' + str(depth_idx) 
        super(Aggregator, self).__init__(name=name)

        self.type = type

        self.use_concat = use_concat
        self.activation = activation

        if self.type == 'gat':
            self.gat_layer = GATLayer(gat_options['activation'])

    def build(self, input_shape, output_shape, gat_shape=None):

        if self.type == 'gat':
            self.gat_layer.build(input_shape, gat_shape)

        self.self_layer = tf.keras.layers.Dense(output_shape, input_shape=(input_shape,), use_bias=True)
        self.self_layer.build((input_shape, ))

        self.neigh_layer = tf.keras.layers.Dense(output_shape, input_shape=(input_shape,), use_bias=True)
        self.neigh_layer.build((input_shape, ))

        self.bn = tf.keras.layers.BatchNormalization()
        self.bn.build((None, (self.use_concat+1)*output_shape))

        self.activation_fn = getattr(tf.nn, self.activation)

        super(Aggregator, self).build(())


    def call(self, self_feats, neigh_feats, nneigh, training=True):

        if self.type == 'maxpool':
            aggregated_feats = tf.reduce_max(neigh_feats, axis=1)

        elif self.type == 'mean':
            aggregated_feats = tf.divide(tf.reduce_sum(neigh_feats, axis=1), tf.expand_dims(nneigh, 1))

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

