import sys
import tensorflow as tf

_EPSILON = tf.keras.backend.epsilon()

class BinaryFocalCrossEntropy(tf.keras.losses.Loss):

    def __init__(self, alpha=0.5, gamma=2.0, weight=1.0):

        self.alpha = alpha

        self.gamma = gamma
        self.weight = weight

        super(BinaryFocalCrossentropy, self).__init__()

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

        super(BinaryCrossEntropyLoss, self).__init__()

        self.weight = weight
        self.cls_criterion = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits)

    def call(self, labels, preds):

        return self.weight * self.cls_criterion(labels, preds)


class RootMeanSquaredError(tf.keras.losses.Loss):

    def __init__(self):
        super(RootMeanSquaredError, self).__init__()

    def call(self, labels, preds):

        return tf.sqrt(tf.reduce_mean((labels - preds)**2, axis=0))


class GradNormLoss(tf.keras.losses.Loss):

    def __init__(self, loss_fn1, loss_fn2, alpha=0.12, l1_scale=1.0, l2_scale=100.0, **kwargs):

        super(GradNormLoss, self).__init__(**kwargs)

        self.loss_fn1 = loss_fn1
        self.loss_fn2 = loss_fn2

        self.alpha = alpha

        self.l1_scale = l1_scale
        self.l2_scale = l2_scale


    def call(self, labels_1, labels_2, preds_1, preds_2):

        loss_1 = self.loss_fn1(labels_1, preds_1)
        scaled_loss_1 = self.l1_scale * loss_1

        grads_1 = tf.gradients(loss_1, self.trainable_variables)
        norm_grads_1 = tf.norm(tf.concat([tf.reshape(g, [-1]) for g in grads_1], axis=0), ord=2)
        l_hat_1 = loss_1 / scaled_loss_1

        loss_2 = self.loss_fn2(labels_2, preds_2) 
        scaled_loss_2 = self.l2_scale * loss_2

        grads_2 = tf.gradients(loss_2, self.trainable_variables)
        norm_grads_2 = tf.norm(tf.concat([tf.reshape(g, [-1]) for g in grads_2], axis=0), ord=2)
        l_hat_2 = loss_2 / scaled_loss_2

        l_hat_avg = (l_hat_1 + l_hat_2) / 2.0
        inv_rate_1 = l_hat_1 / l_hat_avg
        inv_rate_2 = l_hat_2 / l_hat_avg

        G_avg = (norm_grads_1 + norm_grads_2) / 2.0
        C1 = G_avg * tf.pow(inv_rate_1, self.alpha)
        C2 = G_avg * tf.pow(inv_rate_2, self.alpha)

        return tf.reduce_sum(tf.abs(norm_grads_1 - C1)) + tf.reduce_sum(tf.abs(norm_grads_2 - C2))

