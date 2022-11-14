import tensorflow as tf

class ClassificationMetric(tf.keras.metrics.Metric):

    def __init__(self, label=0, threshold=0.5, metric='precision', **kwargs):

        super(ClassificationMetric, self).__init__(name=metric+'_%i'%label, **kwargs)

        self.threshold = threshold
        self.metric = metric

        self.label = label
        if self.label not in [0, 1]:
            sys.exit("Label should be 0 or 1 not %s"%label)

        self.tp = self.add_weight(name='tp', initializer='zeros')

        if self.metric == 'precision':
            self.fp = self.add_weight(name='fp', initializer='zeros')

        elif self.metric == 'recall':
            self.fn = self.add_weight(name='fn', initializer='zeros')
        else:
            sys.exit("Type %s for classification metric not recognized"%self.metric)

    def update_state(self, labels, preds, sample_weight=None):

        labels_i = tf.cast(tf.equal(labels[:, 0], self.label), tf.int32)

        if self.label == 0:
            preds_i = tf.cast(tf.less_equal(preds[:, 0], self.threshold), tf.int32)
        else:
            preds_i = tf.cast(tf.greater_equal(preds[:, 0], self.threshold), tf.int32)
            
        tp = tf.math.count_nonzero(labels_i * preds_i)
        self.tp.assign_add(tf.cast(tp, tf.float32))

        if self.metric == 'precision':
            fp = tf.math.count_nonzero((1-labels_i) * preds_i)
            self.fp.assign_add(tf.cast(fp, tf.float32))

        elif self.metric == 'recall':
            fn = tf.math.count_nonzero(labels_i * (1-preds_i))
            self.fn.assign_add(tf.cast(fn, tf.float32))

    def result(self):
        if self.metric == 'precision':
            return self.tp / (self.tp + self.fp)

        elif self.metric == 'recall':
            return self.tp / (self.tp + self.fn)

    def reset_states(self):
        self.tp.assign(0)

        if self.metric == 'precision':
            self.fp.assign(0)

        elif self.metric == 'recall':
            self.fn.assign(0)
