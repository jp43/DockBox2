import sys
import tensorflow as tf

_EPSILON = tf.keras.backend.epsilon()

class BinaryFocalCrossentropy(tf.keras.losses.Loss):

    def __init__(self, alpha=0.5, gamma=2.0, weight=1.0):

        self.alpha = alpha

        self.gamma = gamma
        self.weight = weight

        super(BinaryFocalCrossentropy, self).__init__()

    def call(self, labels, preds):

        labels_f = tf.dtypes.cast(labels, tf.float32)

        #if self.weight_g:
        #    nrof_graphs = len(graph_size)
        #    graph_cumsize = np.insert(np.cumsum(graph_size), 0, 0)

        #    for kdx in range(nrof_graphs):
        #        graph_labels_f = tf.gather(labels_f, tf.range(graph_cumsize[kdx], graph_cumsize[kdx+1]))

        #        ncorrect = tf.math.maximum(tf.cast(tf.reduce_sum(tf.equal(graph_labels_f, 1)), tf.int32), 1)
        #        nincorrect = tf.math.maximum(tf.cast(tf.reduce_sum(tf.equal(graph_labels_f, 0)), tf.int32), 1)

        #        graph_wg = tf.where(tf.equal(graph_labels_f, 1), 1./ncorrect, 1./nincorrect)

        #        is_first = True if kdx == 0 else False
        #        append_batch_results(wg, graph_wg, first=is_first)

        preds_clipped = tf.clip_by_value(preds, clip_value_min=_EPSILON, clip_value_max=1-_EPSILON)
        p_t = tf.where(tf.equal(labels_f, 1), preds_clipped, 1 - preds_clipped)

        alpha_factor = tf.ones_like(labels_f) * self.alpha 
        alpha_t = tf.where(tf.equal(labels_f, 1), alpha_factor, 1 - alpha_factor)

        loss = -alpha_t * tf.math.pow(1 - p_t, self.gamma) * tf.math.log(p_t)

        return self.weight*tf.reduce_mean(loss)


class BinaryCrossEntropyLoss(tf.keras.losses.Loss):

    def __init__(self, from_logits=False, weight=1.0):

        super(BinaryCrossEntropyLoss, self).__init__()
        self.weight = weight

        self.cls_criterion = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits)

    def call(self, labels, preds):

        return self.weight * self.cls_criterion(labels, preds)
