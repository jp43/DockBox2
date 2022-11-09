import os
import sys
import configparser

import pickle
import networkx as nx
import numpy as np

import tensorflow as tf

class GraphDataset(object):

    def __init__(self, filename, edge_feature=None):

        with open(filename, "rb") as ff:
           graphs = pickle.load(ff)

        self.graph_feats = []
        self.graph_labels = []
        self.graph_adj_list = []

        self.edge_feature = edge_feature
        if edge_feature is None:
            self.graph_edge_feats = None
        else:
            self.graph_edge_feats = []

        for graph in graphs:
            feats = []
            labels = []

            for node, data in graph.nodes(data=True):
                feats.append(data['feature'])
                labels.append(data['label'])

            adj = nx.adjacency_matrix(graph).toarray()
            self.graph_adj_list.append(adj)
            self.graph_feats.append(np.array(feats))

            if edge_feature is not None:
                self.graph_edge_feats.append(nx.to_numpy_array(graph, weight=edge_feature))

            if isinstance(labels[0], int):
                labels = np.array(labels)
                labels = labels[:,np.newaxis]
            else:
                labels = np.array(labels)
            self.graph_labels.append(labels)

        self.max_nrof_nodes = max([len(graph) for graph in graphs])
        self.nfeats = self.graph_feats[0].shape[1]

        self.nlabels = self.graph_labels[0].shape[1]
        self.ngraphs = len(graphs)

        self.graph_size = []
        for kdx, graph in enumerate(graphs):
            self.graph_size.append(len(graph))

    def __sample(self, graph_id, depth, nrof_neigh):

        graph_size = self.graph_size[graph_id]
        adj = self.graph_adj_list[graph_id]

        edge_feats = None 
        if self.edge_feature is not None:
            edge_feats = self.graph_edge_feats[graph_id]

        diff = self.max_nrof_nodes - graph_size

        depth = tf.get_static_value(depth)
        nrof_neigh = tf.get_static_value(nrof_neigh)

        if diff > 0:
            feats = np.concatenate([self.graph_feats[graph_id], np.zeros((diff, self.nfeats), dtype=np.float32)], axis=0)
            labels = np.concatenate([self.graph_labels[graph_id], np.zeros((diff, self.nlabels), dtype=np.int32)], axis=0)
        else:
            feats = self.graph_feats[graph_id]
            labels = self.graph_labels[graph_id]

        neigh_indices, adj_mask, edge_feats_mask, nneigh = sample_neighbors(adj, edge_feats, graph_size, depth, nrof_neigh)

        for idx in range(depth):
            if diff > 0:
                # zero padding for each layer until getting max_nrof_nodes nodes per graph
                neigh_indices[idx] = np.concatenate([neigh_indices[idx], np.zeros((diff, nrof_neigh), dtype=np.int32)], axis=0)
                adj_mask[idx] = np.concatenate([adj_mask[idx], np.zeros((diff, nrof_neigh), dtype=np.int32)], axis=0)

                edge_feats_mask[idx] = np.concatenate([edge_feats_mask[idx], np.zeros((diff, nrof_neigh), dtype=np.float32)], axis=0)
                nneigh[idx] = np.concatenate([nneigh[idx], np.zeros((diff), dtype=np.int32)], axis=0)

        return np.asarray(feats, np.float32), graph_size, np.asarray(neigh_indices, dtype=np.int32), \
               np.asarray(adj_mask, dtype=np.float32), np.asarray(edge_feats_mask, dtype=np.float32), \
               np.asarray(nneigh, dtype=np.float32), labels

    def sample(self, graph_id, depth, nrof_neigh):

        return tf.py_function(self.__sample, [graph_id, depth, nrof_neigh], [tf.float32, tf.int32,
                                   tf.int32, tf.float32, tf.float32, tf.float32, tf.int32])


def generate_data_loader(dataset, depth, nrof_neigh, num_parallel_calls=1, batch_size=1):
    """Create data loader to be fed to model"""

    data_slices = np.random.permutation(dataset.ngraphs)
    data_loader = tf.data.Dataset.from_tensor_slices((data_slices))

    data_loader = data_loader.map(**{'num_parallel_calls': num_parallel_calls, 'map_func': lambda x: dataset.sample(x, depth, nrof_neigh)})
    data_loader = data_loader.batch(batch_size=batch_size)

    return data_loader


def sample_neighbors(adj, edge_feats, graph_size, depth, nrof_neigh):
    neigh_indices = []
    adj_mask = []
    edge_feats_mask = []
    nneigh = []

    for idx in range(depth):
        graph_neigh_indices = []

        graph_adj_mask = []
        graph_edge_feats_mask = []
        graph_nneigh = []

        for jdx in range(graph_size):
            # get indices of neighbors of node jdx
            node_neigh_indices = np.where(adj[jdx] != 0)[0]

            # get indices of non neighbors
            node_non_neigh_indices = np.setdiff1d(np.arange(graph_size), node_neigh_indices)

            # if number of neighbors is too high, select randomly until having "nrof_neigh" neighbors
            if len(node_neigh_indices) > nrof_neigh:
                node_neigh_indices = np.random.choice(node_neigh_indices, nrof_neigh, replace=False)

            node_nneigh = len(node_neigh_indices)

            if node_nneigh < nrof_neigh: # zero padding until getting "nrof_neigh" neighbors
                node_neigh_indices = np.concatenate([node_neigh_indices, [node_non_neigh_indices[0]]*(nrof_neigh - node_nneigh)])

            graph_neigh_indices.append(node_neigh_indices)
            graph_adj_mask.append(adj[jdx][node_neigh_indices])

            if edge_feats is None:
                graph_edge_feats_mask.append(np.zeros_like(node_neigh_indices))
            else:
                graph_edge_feats_mask.append(edge_feats[jdx][node_neigh_indices])

            graph_nneigh.append(node_nneigh)

        neigh_indices.append(graph_neigh_indices)

        adj_mask.append(graph_adj_mask)
        edge_feats_mask.append(graph_edge_feats_mask) 
        nneigh.append(graph_nneigh)

    return neigh_indices, adj_mask, edge_feats_mask, nneigh

