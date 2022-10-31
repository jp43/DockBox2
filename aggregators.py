import sys
import tensorflow as tf

supported_types = ['pooling', 'mean', 'attention']

default_aggregator_options = {'use_concat': True, 'attention_shape': None, 'activation': 'leaky_relu', 'attention_activation': 'sigmoid'}


class Aggregator(tf.keras.layers.Layer):

    def __init__(self, type, activation, use_concat, attention_activation='sigmoid'):

        super(Aggregator, self).__init__()

        self.activation = getattr(tf.nn, activation)

        self.use_concat = use_concat
        self.atype = type

        if self.atype == 'attention':
            self.attention_activation = getattr(tf.nn, attention_activation)


    def build(self, input_shape, output_shape, attention_shape=None):

        if self.atype == 'attention':

            self.attention_layer1 = tf.keras.layers.Dense(attention_shape, input_shape=(input_shape,), name='attention_layer1')
            self.attention_layer1.build((input_shape,))

            self.attention_bn = tf.keras.layers.BatchNormalization()
            self.attention_bn.build((None, attention_shape))

            self.attention_layer2 = tf.keras.layers.Dense(1, input_shape=(2*attention_shape,), name='attention_layer2')
            self.attention_layer2.build((2*attention_shape,))

        self.self_layer = tf.keras.layers.Dense(output_shape, input_shape=(input_shape,), name='self_layer')
        self.self_layer.build((input_shape, ))

        self.neigh_layer = tf.keras.layers.Dense(output_shape, input_shape=(input_shape,), name='neigh_layer')
        self.neigh_layer.build((input_shape, ))

        self.bn = tf.keras.layers.BatchNormalization()
        self.bn.build((None, (self.use_concat+1)*output_shape))

        super(Aggregator, self).build(())


    def call(self, self_feats, neigh_feats, nneigh, training=True):

        if self.atype == 'pooling':

            aggregated_feats = tf.reduce_max(neigh_feats, axis=1)

        elif self.atype == 'mean':

            aggregated_feats = tf.divide(tf.reduce_sum(neigh_feats, axis=1), tf.expand_dims(nneigh, 1))

        elif self.atype == 'attention':

            self_feats_attention = self.attention_layer1(self_feats)
            self_feats_attention = self.attention_bn(self_feats_attention, training=training)

            neigh_feats_attention = self.attention_layer1(neigh_feats)

            neigh_feats_upd = self.attention_bn(tf.reshape(neigh_feats_attention, [-1, int(neigh_feats_attention.shape[-1])]), training=training)
            neigh_feats_attention = tf.reshape(neigh_feats_upd, list(neigh_feats_attention.shape))
 
            concat = tf.concat([tf.stack([self_feats_attention]*tf.shape(neigh_feats_attention)[1].numpy(), axis=1), neigh_feats_attention], axis=2)
            coefficients = self.attention_layer2(concat)

            coefficients = tf.nn.softmax(tf.squeeze(coefficients, axis=2))
            coefficients = tf.expand_dims(coefficients, axis=2)

            aggregated_feats = self.attention_activation(tf.reduce_sum(tf.multiply(coefficients, neigh_feats), axis=1))

        self_feats = self.self_layer(self_feats)
        neigh_feats = self.neigh_layer(aggregated_feats)

        if self.use_concat:
            self_feats = tf.concat([self_feats, neigh_feats], axis=1)
        else:
            self_feats = tf.add_n([self_feats, neigh_feats])


        self_feats = self.bn(self_feats, training=training)
        self_feats = self.activation(self_feats)

        return self_feats
