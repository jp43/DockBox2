import tensorflow as tf

def append_batch_results(index, values, batch_values):

    if index == 0:
        return batch_values
    else:
        return tf.concat([values, batch_values], axis=0)
