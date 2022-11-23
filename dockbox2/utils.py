import random
import tensorflow as tf
import numpy as np

abbreviation = {"precision": "Pr", "recall": "Rc", "f1_score": "F1",}

def append_batch_results(values, batch_values, first=True):

    if first:
        return batch_values
    else:
        return tf.concat([values, batch_values], axis=0)

def set_seed(seed):

   tf.random.set_seed(seed)
   np.random.seed(seed)
   random.seed(seed)

