import tensorflow as tf

def append_batch_results(values, batch_values, first=True):

    if first:
        return batch_values
    else:
        return tf.concat([values, batch_values], axis=0)