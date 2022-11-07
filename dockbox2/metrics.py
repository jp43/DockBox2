import tensorflow as tf

class LabeledPrecision(tf.keras.metrics.Metric):

    def __init__(self, label=0, threshold=0.5, name=None, **kwargs):

        if name is None:
            super(LabeledPrecision, self).__init__(name='precision_%i'%label, **kwargs)
        else:
            super(LabeledPrecision, self).__init__(name=name, **kwargs)

        self.threshold = threshold

        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')

        if label not in [0, 1]:
            sys.exit("Label should be 0 or 1 not %s"%label)
        else:
            self.label = label
        
    def update_state(self, labels, preds, sample_weight=None):

        labels_i = tf.cast(tf.equal(labels[:, 0], self.label), tf.int64)

        if self.label == 0:
            preds_i = tf.cast(tf.less_equal(preds[:, 0], self.threshold), tf.int64)
        else:
            preds_i = tf.cast(tf.greater_equal(preds[:, 0], self.threshold), tf.int64)
            
        tp = tf.math.count_nonzero(labels_i * preds_i)
        fp = tf.math.count_nonzero((1-labels_i) * preds_i)

        self.tp.assign_add(tf.cast(tp, tf.float32))
        self.fp.assign_add(tf.cast(fp, tf.float32))

    def result(self):
        return self.tp / (self.tp + self.fp)

    def reset_states(self):
        self.tp.assign(0)
        self.fp.assign(0)


class LabeledRecall(tf.keras.metrics.Metric):

    def __init__(self, label=0, threshold=0.5, name=None, **kwargs):

        if name is None:
            super(LabeledRecall, self).__init__(name='recall_%i'%label, **kwargs)
        else:
            super(LabeledRecall, self).__init__(name=name, **kwargs)

        self.threshold = threshold

        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')

        if label not in [0, 1]:
            sys.exit("Label should be 0 or 1 not %s"%label)
        else:
            self.label = label

    def update_state(self, labels, preds, sample_weight=None):

        labels_i = tf.cast(tf.equal(labels[:, 0], self.label), tf.int64)

        if self.label == 0:
            preds_i = tf.cast(tf.less_equal(preds[:, 0], self.threshold), tf.int64)
        else:
            preds_i = tf.cast(tf.greater_equal(preds[:, 0], self.threshold), tf.int64)

        tp = tf.math.count_nonzero(labels_i * preds_i)
        fn = tf.math.count_nonzero(labels_i * (1-preds_i))

        self.tp.assign_add(tf.cast(tp, tf.float32))
        self.fn.assign_add(tf.cast(fn, tf.float32))

    def result(self):
        return self.tp / (self.tp + self.fn)

    def reset_states(self):
        self.tp.assign(0)
        self.fn.assign(0)

