import tensorflow as tf

class PoolAggregator(tf.keras.layers.Layer):

    def __init__(self, activation, pool_op, use_concat):

        super(PoolAggregator, self).__init__()

        self.activation = getattr(tf.nn, activation)
        self.pool_op = pool_op
        self.use_concat = use_concat

    def build(self, input_shape, output_shape):

        self.transform_node_weight = tf.keras.layers.Dense(output_shape, input_shape=(input_shape,), name='transform_node_weight')
        self.transform_node_weight.build((input_shape, ))

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn1.build((None, output_shape))

        self.self_layer = tf.keras.layers.Dense(output_shape, input_shape=(input_shape,), name='self_layer', use_bias=False)
        self.self_layer.build((input_shape, ))

        self.neigh_layer = tf.keras.layers.Dense(output_shape, input_shape=(output_shape,), name='neigh_layer', use_bias=False)
        self.neigh_layer.build((output_shape, ))

        self.bn2 = tf.keras.layers.BatchNormalization()
        self.bn2.build((None, (self.use_concat+1)*output_shape))

        super(PoolAggregator, self).build(())

    def call(self, self_feats, neigh_feats, training=True):

        neigh = self.transform_node_weight(neigh_feats)
        neigh_feats_upd = self.bn1(tf.reshape(neigh, [-1, int(neigh.shape[-1])]), training=training)
        neigh = tf.reshape(neigh_feats_upd, list(neigh.shape))
        neigh = getattr(tf, self.pool_op)(neigh, axis=1)

        self_feats = self.self_layer(self_feats)
        neigh_feats = self.neigh_layer(neigh)

        if self.use_concat:
            self_feats = tf.concat([self_feats, neigh], axis=1)
        else:
            self_feats = tf.add_n([self_feats, neigh])

        self_feats = self.bn2(self_feats, training=training)
        self_feats = self.activation(self_feats)

        return self_feats
