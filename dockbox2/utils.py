import random
import pickle
import tensorflow as tf

import numpy as np
import h5py

known_scoring_functions = ['autodock', 'dock', 'dsx', 'gnina', 'moe', 'vina']
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

def save_predicted_node_labels(filename, node_labels, pred_node_labels, graph_size, data_slices):

    graph_cumsize = np.insert(np.cumsum(graph_size), 0, 0)

    results = {}
    for kdx, idx in enumerate(np.argsort(data_slices)):

        node_labels_cg = node_labels[graph_cumsize[idx]:graph_cumsize[idx+1]]
        pred_node_labels_cg = pred_node_labels[graph_cumsize[idx]:graph_cumsize[idx+1]]

        results[kdx] = {'label': list(tf.squeeze(node_labels_cg).numpy()),
                        'pred': list(tf.squeeze(pred_node_labels_cg).numpy())}

    with open(filename, "wb") as ff:
        pickle.dump(results, ff)

