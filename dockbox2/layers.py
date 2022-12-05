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
        neigh_feats = self.layer(concat)

        return neigh_feats

class Aggregator(tf.keras.layers.Layer):

    def __init__(self, depth_idx, type, activation, use_concat):

        name = type.capitalize() + '_agg_' + str(depth_idx) 
        super(Aggregator, self).__init__(name=name)

        self.type = type

        self.use_concat = use_concat
        self.activation = activation

    def build(self, input_shape, output_shape):

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

