import random
import pickle
import tensorflow as tf
import numpy as np
import h5py

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

def save_predicted_labels(filename, labels, pred_labels, graph_size, data_slices):

    graph_cumsize = np.insert(np.cumsum(graph_size), 0, 0)

    results = {}
    for kdx, idx in enumerate(np.argsort(data_slices)):

        graph_labels = labels[graph_cumsize[idx]:graph_cumsize[idx+1]]
        graph_pred_labels = pred_labels[graph_cumsize[idx]:graph_cumsize[idx+1]]

        results[kdx] = {'label': list(tf.squeeze(graph_labels).numpy()),
                        'pred': list(tf.squeeze(graph_pred_labels).numpy())}

    with open(filename, "wb") as ff:
        pickle.dump(results, ff)

