import sys
import tensorflow as tf

_EPSILON = tf.keras.backend.epsilon()

class BinaryFocalCrossentropy(tf.keras.losses.Loss):

    def __init__(self, apply_class_balancing=False, alpha=0.25, gamma=2.0, from_logits=False, label_smoothing=0.0, weight=1.0):

        self.apply_class_balancing = apply_class_balancing
        self.alpha = alpha

        self.gamma = gamma
        self.weight = weight
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing

        super(BinaryFocalCrossentropy, self).__init__()

    def call(self, labels, preds):

        labels_f = tf.dtypes.cast(labels, tf.float32)

        preds_clipped = tf.clip_by_value(preds, clip_value_min=_EPSILON, clip_value_max=1-_EPSILON)
        p_t = tf.where(tf.equal(labels_f, 1), preds_clipped, 1 - preds_clipped)

        if self.apply_class_balancing:
            alpha_factor = tf.ones_like(labels_f) * self.alpha 
            alpha_t = tf.where(tf.equal(labels_f, 1), alpha_factor, 1 - alpha_factor)
        else:
            alpha_t = tf.ones_like(labels_f)

        loss = -alpha_t * tf.math.pow(1 - p_t, self.gamma) * tf.math.log(p_t)
        return  self.weight*tf.reduce_mean(loss)
