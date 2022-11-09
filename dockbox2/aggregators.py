import sys
import tensorflow as tf

supported_types = ['pooling', 'mean', 'attention']

default_aggregator_options = {'use_concat': True, 'activation': 'leaky_relu'}
default_attention_options = {'activation': 'leaky_relu'}

class Aggregator(tf.keras.layers.Layer):

    def __init__(self, type, activation, use_concat, is_edge_feature=False, attention_options=None):

        super(Aggregator, self).__init__()

        self.type = type
        self.use_concat = use_concat

        self.activation = getattr(tf.nn, activation)

        self.is_edge_feature = is_edge_feature
        if self.is_edge_feature:
            self.edge_activation = getattr(tf.nn, activation)

        if self.type == 'attention':
            attention_activation = attention_options['activation']
            self.attention_layer = AttentionLayer(attention_activation)

    def build(self, input_shape, output_shape, attention_shape=None):

        if self.is_edge_feature:
            self.edge_layer = tf.keras.layers.Dense(input_shape, input_shape=(input_shape+1,), name='edge_layer')

        if self.type == 'attention':
            self.attention_layer.build(input_shape, attention_shape)

        self.self_layer = tf.keras.layers.Dense(output_shape, input_shape=(input_shape,), name='self_layer')
        self.self_layer.build((input_shape, ))

        self.neigh_layer = tf.keras.layers.Dense(output_shape, input_shape=(input_shape,), name='neigh_layer')
        self.neigh_layer.build((input_shape, ))

        self.bn = tf.keras.layers.BatchNormalization()
        self.bn.build((None, (self.use_concat+1)*output_shape))

        super(Aggregator, self).build(())

    def call(self, self_feats, neigh_feats, neigh_edge_feats, nneigh, training=True):

        if self.is_edge_feature:
            concat = tf.concat([neigh_feats, tf.expand_dims(neigh_edge_feats, axis=2)], axis=2)
            extracted_neigh_feats = self.edge_layer(concat)
            extracted_neigh_feats = self.edge_activation(extracted_neigh_feats)
        else:
            extracted_neigh_feats = neigh_feats

        if self.type == 'pooling':
            aggregated_feats = tf.reduce_max(extracted_neigh_feats, axis=1)

        elif self.type == 'mean':
            aggregated_feats = tf.divide(tf.reduce_sum(extracted_neigh_feats, axis=1), tf.expand_dims(nneigh, 1))

        elif self.type == 'attention':
            attention_weights = self.attention_layer(self_feats, extracted_neigh_feats, training=training)
            aggregated_feats = tf.reduce_sum(tf.multiply(attention_weights, extracted_neigh_feats), axis=1)
        else:
            sys.exit("Unrecognized type aggregator %s"%self.type)

        self_feats = self.self_layer(self_feats)
        aggregated_feats = self.neigh_layer(aggregated_feats)

        if self.use_concat:
            self_feats = tf.concat([self_feats, aggregated_feats], axis=1)
        else:
            self_feats = tf.add_n([self_feats, aggregated_feats])

        self_feats = self.bn(self_feats, training=training)
        self_feats = self.activation(self_feats)

        return self_feats

class AttentionLayer(tf.keras.layers.Layer):

    def __init__(self, activation):

        super(AttentionLayer, self).__init__()
        self.activation = getattr(tf.nn, activation)

    def build(self, input_shape, attention_shape):

        self.shared_layer = tf.keras.layers.Dense(attention_shape, input_shape=(input_shape,), name='shared_layer')
        self.shared_layer.build((input_shape,))

        self.bn = tf.keras.layers.BatchNormalization()
        self.bn.build((None, attention_shape))

        self.output_layer = tf.keras.layers.Dense(1, input_shape=(2*attention_shape,), name='output_layer')
        self.output_layer.build((2*attention_shape,))

        super(AttentionLayer, self).build(())

    def call(self, self_feats, neigh_feats, training=True):

        self_feats_shared = self.shared_layer(self_feats)
        self_feats_shared = self.bn(self_feats_shared, training=training)

        neigh_feats_shared = self.shared_layer(neigh_feats)
        neigh_feats_upd = self.bn(tf.reshape(neigh_feats_shared, [-1, int(neigh_feats_shared.shape[-1])]), training=training)
        neigh_feats_shared = tf.reshape(neigh_feats_upd, list(neigh_feats_shared.shape))

        concat = tf.concat([tf.stack([self_feats_shared]*tf.shape(neigh_feats_shared)[1].numpy(), axis=1), neigh_feats_shared], axis=2)
        weights = self.output_layer(concat)
        weights = self.activation(weights)

        weights = tf.nn.softmax(tf.squeeze(weights, axis=2))
        weights = tf.expand_dims(weights, axis=2)

        return weights
