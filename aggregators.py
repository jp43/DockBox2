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
        if self.use_concat:
            self.bn2.build((None, 2*output_shape))
        else:
            self.bn2.build((None, output_shape))

        super(PoolAggregator, self).build(())

    def call(self, self_nodes, neigh_nodes, len_adj_nodes, training=True):

        neigh = self.transform_node_weight(neigh_nodes)
        neigh_nodes_upd = self.bn1(tf.reshape(neigh, [-1, int(neigh.shape[-1])]), training=training)
        neigh = tf.reshape(neigh_nodes_upd, list(neigh.shape))
        neigh = getattr(tf, self.pool_op)(neigh, axis=1)

        neigh = self.neigh_layer(neigh)
        self_nodes = self.self_layer(self_nodes)

        if self.use_concat:
            output = tf.concat([self_nodes, neigh], axis=1)
        else:
            output = tf.add_n([self_nodes, neigh])

        output = self.bn2(output, training=training)
        output = self.activation(output)

        return output
