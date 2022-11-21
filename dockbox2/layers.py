import sys
import tensorflow as tf

class Edger(tf.keras.layers.Layer):

    def __init__(self, depth_idx, type, activation, depth, offset=False):

        name = type.capitalize() + '_extractor_' + str(depth_idx)
        super(Edger, self).__init__(name=name)

        self.type = type
        self.depth = depth

        self.activation = activation

    def build(self, input_shape):

        self.edge_layer = tf.keras.Sequential(name='edge_seq')

        for idx in range(self.depth):
            if idx == 0:
                layer_input_shape = input_shape + 1
            else:
                layer_input_shape = input_shape
            self.edge_layer.add(tf.keras.layers.Dense(input_shape, input_shape=(layer_input_shape,), activation=self.activation))

        self.edge_layer.build((input_shape+1, ))

        super(Edger, self).build(())

    def call(self, neigh_feats, neigh_edge_feats):

        concat = tf.concat([neigh_feats, tf.expand_dims(neigh_edge_feats, axis=2)], axis=2)
        neigh_feats = self.edge_layer(concat)

        return neigh_feats

class Aggregator(tf.keras.layers.Layer):

    def __init__(self, depth_idx, type, activation, use_concat, attention_options=None):

        name = type.capitalize() + '_agg_' + str(depth_idx) 
        super(Aggregator, self).__init__(name=name)

        self.type = type
        self.use_concat = use_concat

        self.activation = getattr(tf.nn, activation)

        if self.type == 'attention':
            self.attention_layer = GATLayer(attention_options['activation'])

    def build(self, input_shape, output_shape, attention_shape=None):

        if self.type == 'attention':
            self.attention_layer.build(input_shape, attention_shape)

        self.self_agg_layer = tf.keras.layers.Dense(output_shape, input_shape=(input_shape,), name='self_agg')
        self.self_agg_layer.build((input_shape, ))

        self.neigh_agg_layer = tf.keras.layers.Dense(output_shape, input_shape=(input_shape,), name='neigh_agg')
        self.neigh_agg_layer.build((input_shape, ))

        self.bn = tf.keras.layers.BatchNormalization()
        self.bn.build((None, (self.use_concat+1)*output_shape))

        super(Aggregator, self).build(())


    def call(self, self_feats, neigh_feats, nneigh, training=True):

        if self.type == 'maxpool':
            aggregated_feats = tf.reduce_max(neigh_feats, axis=1)

        elif self.type == 'mean':
            aggregated_feats = tf.divide(tf.reduce_sum(neigh_feats, axis=1), tf.expand_dims(nneigh, 1))

        elif self.type == 'attention':
            attention_weights = self.attention_layer(self_feats, neigh_feats, training=training)
            aggregated_feats = tf.reduce_sum(tf.multiply(attention_weights, neigh_feats), axis=1)
        else:
            sys.exit("Unrecognized aggregator type %s"%self.type)

        self_feats = self.self_agg_layer(self_feats)
        aggregated_feats = self.neigh_agg_layer(aggregated_feats)
        if self.use_concat:
            self_feats = tf.concat([self_feats, aggregated_feats], axis=1)
        else:
            self_feats = tf.add_n([self_feats, aggregated_feats])

        self_feats = self.bn(self_feats, training=training)
        self_feats = self.activation(self_feats)

        return self_feats


class GATLayer(tf.keras.layers.Layer):

    def __init__(self, activation):

        super(GATLayer, self).__init__()

        self.activation = getattr(tf.nn, activation)


    def build(self, input_shape, attention_shape):

        self.shared_gat_layer = tf.keras.layers.Dense(attention_shape, input_shape=(input_shape,), name='shared_gat')
        self.shared_gat_layer.build((input_shape,))

        self.bn = tf.keras.layers.BatchNormalization()
        self.bn.build((None, attention_shape))

        self.output_gat_layer = tf.keras.layers.Dense(1, input_shape=(2*attention_shape,), name='output_gat')

        self.output_gat_layer.build((2*attention_shape,))

        super(GATLayer, self).build(())

    def call(self, self_feats, neigh_feats, training=True):

        self_feats_shared = self.shared_gat_layer(self_feats)
        self_feats_shared = self.bn(self_feats_shared, training=training)

        neigh_feats_shared = self.shared_gat_layer(neigh_feats)
        neigh_feats_upd = self.bn(tf.reshape(neigh_feats_shared, [-1, int(neigh_feats_shared.shape[-1])]), training=training)

        neigh_feats_shared = tf.reshape(neigh_feats_upd, list(neigh_feats_shared.shape))

        concat = tf.concat([tf.stack([self_feats_shared]*tf.shape(neigh_feats_shared)[1].numpy(), axis=1), neigh_feats_shared], axis=2)

        weights = self.output_gat_layer(concat)
        weights = self.activation(weights)

        weights = tf.nn.softmax(tf.squeeze(weights, axis=2))
        weights = tf.expand_dims(weights, axis=2)

        return weights
