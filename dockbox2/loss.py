import sys
import tensorflow as tf

_EPSILON = tf.keras.backend.epsilon()

class BinaryFocalCrossEntropy(tf.keras.losses.Loss):

    def __init__(self, alpha=0.5, gamma=2.0, weight=1.0):

        self.alpha = alpha

        self.gamma = gamma
        self.weight = weight

        super(BinaryFocalCrossEntropy, self).__init__()

    def call(self, labels, preds):

        labels_f = tf.dtypes.cast(labels, tf.float32)

        preds_clipped = tf.clip_by_value(preds, clip_value_min=_EPSILON, clip_value_max=1-_EPSILON)
        p_t = tf.where(tf.equal(labels_f, 1), preds_clipped, 1 - preds_clipped)

        alpha_factor = tf.ones_like(labels_f) * self.alpha 
        alpha_t = tf.where(tf.equal(labels_f, 1), alpha_factor, 1 - alpha_factor)

        loss = -alpha_t * tf.math.pow(1 - p_t, self.gamma) * tf.math.log(p_t)

        return self.weight*tf.reduce_mean(loss)


class BinaryCrossEntropyLoss(tf.keras.losses.Loss):

    def __init__(self, from_logits=False, weight=1.0):

        self.weight = weight
        self.cls_criterion = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits)

        super(BinaryCrossEntropyLoss, self).__init__()

    def call(self, labels, preds):
        return self.weight * self.cls_criterion(labels, preds)


class RootMeanSquaredError(tf.keras.losses.Loss):

    def __init__(self, weight=1.0):

        self.weight = weight
        super(RootMeanSquaredError, self).__init__()

    def call(self, labels, preds):
        return self.weight * tf.sqrt(tf.reduce_mean((labels - preds)**2, axis=0))

