import os
import sys
import configparser

import pickle
import networkx as nx
import numpy as np

import tensorflow as tf

class GraphDataset(object):

    def __init__(self, filename, edge_options):

        with open(filename, "rb") as ff:
            graphs, bm_cogs = pickle.load(ff)

        self.feats = []
        self.labels = []

        self.cogs = []
        # relative position to binding mode
        self.bm_xyz = []

        self.adj = []
        self.rmsd = []

        for kdx, graph in enumerate(graphs):
            graph_feats = []
            graph_labels = []
            graph_cogs = []

            # load features and labels
            for node, data in graph.nodes(data=True):
                graph_feats.append(data['feature'])
                graph_labels.append(data['label'])
                graph_cogs.append(data['cog'])

            self.feats.append(np.array(graph_feats))

            graph_cogs = np.array(graph_cogs)
            self.cogs.append(graph_cogs)

            self.bm_xyz.append(bm_cogs[kdx] - graph_cogs)

            if isinstance(graph_labels[0], int):
                graph_labels = np.array(graph_labels)
                graph_labels = graph_labels[:,np.newaxis]
            else:
                graph_labels = np.array(graph_labels)
            self.labels.append(graph_labels)

            # load adjacency matrix
            adj_matrix = nx.adjacency_matrix(graph).toarray()
            self.adj.append(adj_matrix)

            if edge_options['type'] is None or 'rmsd' not in edge_options['type']:
                rmsd_matrix = np.zeros_like(adj_matrix, dtype=float)

            elif 'rmsd' in edge_options['type']:
                # load rmsd values as edge feats
                rmsd_matrix = nx.to_numpy_array(graph, weight='rmsd')

            self.rmsd.append(rmsd_matrix)

        self.max_nrof_nodes = max([len(graph) for graph in graphs])
        self.nfeats = self.feats[0].shape[1]

        self.nlabels = self.labels[0].shape[1]
        self.ngraphs = len(graphs)

        self.graph_size = []
        for kdx, graph in enumerate(graphs):
            self.graph_size.append(len(graph))

    def __sample(self, graph_id, depth, nrof_neigh):

        graph_size = self.graph_size[graph_id]
        adj_matrix = self.adj[graph_id]
        rmsd_matrix = self.rmsd[graph_id]

        diff = self.max_nrof_nodes - graph_size

        depth = tf.get_static_value(depth)
        nrof_neigh = tf.get_static_value(nrof_neigh)

        if diff > 0:
            graph_feats = np.concatenate([self.feats[graph_id], np.zeros((diff, self.nfeats), dtype=np.float32)], axis=0)
            graph_labels = np.concatenate([self.labels[graph_id], np.zeros((diff, self.nlabels), dtype=np.int32)], axis=0)

            graph_cogs = np.concatenate([self.cogs[graph_id], np.zeros((diff, 3), dtype=np.float32)], axis=0)
            graph_bm_xyz = np.concatenate([self.bm_xyz[graph_id], np.zeros((diff, 3), dtype=np.float32)], axis=0)

        else:
            graph_feats = self.feats[graph_id]
            graph_labels = self.labels[graph_id]

            graph_cogs = self.cogs[graph_id]
            graph_bm_xyz = self.bm_xyz[graph_id]

        neigh_indices, neigh_adj_values, neigh_rmsd, nneigh = sample_neighbors(adj_matrix, rmsd_matrix, graph_size, depth, nrof_neigh)

        for idx in range(depth):
            if diff > 0:
                # zero padding for each layer until getting max_nrof_nodes nodes per graph
                neigh_indices[idx] = np.concatenate([neigh_indices[idx], np.zeros((diff, nrof_neigh), dtype=np.int32)], axis=0)
                neigh_adj_values[idx] = np.concatenate([neigh_adj_values[idx], np.zeros((diff, nrof_neigh), dtype=np.int32)], axis=0)

                neigh_rmsd[idx] = np.concatenate([neigh_rmsd[idx], np.zeros((diff, nrof_neigh), dtype=np.float32)], axis=0)
                nneigh[idx] = np.concatenate([nneigh[idx], np.zeros((diff), dtype=np.int32)], axis=0)

        return np.asarray(graph_feats, np.float32), np.asarray(graph_cogs, np.float32), graph_size, \
               np.asarray(neigh_indices, dtype=np.int32), np.asarray(neigh_adj_values, dtype=np.float32), \
               np.asarray(neigh_rmsd, dtype=np.float32), np.asarray(nneigh, dtype=np.float32), graph_labels, graph_bm_xyz


    def sample(self, graph_id, depth, nrof_neigh):

        return tf.py_function(self.__sample, [graph_id, depth, nrof_neigh], [tf.float32, tf.float32, tf.int32,
                                   tf.int32, tf.float32, tf.float32, tf.float32, tf.int32, tf.float32])


def generate_data_loader(dataset, depth, nrof_neigh, num_parallel_calls=1, batch_size=1):
    """Create data loader to be fed to model"""

    data_slices = np.random.permutation(dataset.ngraphs)
    data_loader = tf.data.Dataset.from_tensor_slices((data_slices))

    data_loader = data_loader.map(**{'num_parallel_calls': num_parallel_calls, 'map_func': lambda x: dataset.sample(x, depth, nrof_neigh)})
    data_loader = data_loader.batch(batch_size=batch_size)

    return data_loader, data_slices

def sample_neighbors(adj_matrix, rmsd_matrix, graph_size, depth, nrof_neigh):
    neigh_indices = []
    neigh_adj_values = []
    neigh_rmsd = []
    nneigh = []

    for idx in range(depth):
        layer_neigh_indices = []

        layer_neigh_adj_values = []
        layer_neigh_rmsd = []
        layer_nneigh = []

        for jdx in range(graph_size):
            # get indices of neighbors of node jdx
            node_neigh_indices = np.where(adj_matrix[jdx] != 0)[0]

            # get indices of non neighbors
            node_non_neigh_indices = np.setdiff1d(np.arange(graph_size), node_neigh_indices)

            # if number of neighbors is too high, select randomly until having "nrof_neigh" neighbors
            if len(node_neigh_indices) > nrof_neigh:
                node_neigh_indices = np.random.choice(node_neigh_indices, nrof_neigh, replace=False)

            node_nneigh = len(node_neigh_indices)

            if node_nneigh < nrof_neigh: # zero padding until getting "nrof_neigh" neighbors
                node_neigh_indices = np.concatenate([node_neigh_indices, [node_non_neigh_indices[0]]*(nrof_neigh - node_nneigh)])

            layer_neigh_indices.append(node_neigh_indices)
            layer_neigh_adj_values.append(adj_matrix[jdx][node_neigh_indices])

            layer_neigh_rmsd.append(rmsd_matrix[jdx][node_neigh_indices])
            layer_nneigh.append(node_nneigh)

        neigh_indices.append(layer_neigh_indices)

        neigh_adj_values.append(layer_neigh_adj_values)
        neigh_rmsd.append(layer_neigh_rmsd)
        nneigh.append(layer_nneigh)

    return neigh_indices, neigh_adj_values, neigh_rmsd, nneigh

