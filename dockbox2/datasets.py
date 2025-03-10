import os
import sys
import configparser

import pickle
import networkx as nx
import numpy as np

import tensorflow as tf

class GraphDataset(object):

    def __init__(self, filename, node_options, edge_options, task_level, pkd_model, training=True):

        self.training = training
        self.task_level = task_level

        feat_names = node_options['features']
        rmsd_cutoff = node_options['rmsd_cutoff']

        with open(filename, "rb") as ff:
            graphs = pickle.load(ff)

        # graphs is a single graph
        if isinstance(graphs, nx.Graph):
            self.graph_labels = None
            graphs = [graphs]
            if 'graph' in task_level and training:
                raise ValueError("Graph labels not found. Required for graph-level prediction!")

        # if graphs is not a graph, it should be a list
        elif not isinstance(graphs, list):
            raise ValueError("Input format not recognized!")

        # graphs is a list and each element is a graph
        elif all([isinstance(graph, nx.Graph) for graph in graphs]):
            self.graph_labels = None
            if 'graph' in task_level and training:
                raise ValueError("Graph labels not found. Required for graph-level prediction!")

        # graphs is a list with 2 elements and the first one is a graph 
        # note that the second element is not a graph as it would have passed the previous condition
        elif len(graphs) == 2 and isinstance(graphs[0], nx.Graph):

            if 'graph' in self.task_level:
                self.check_graph_labels([graphs[1]], pkd_model)
                self.graph_labels = np.array([graphs[1]])
                graphs = [graphs[0]]
            else:
                self.graph_labels = None

        # graphs is a list and all the elements are a list with 2 elements
        elif all([isinstance(graph, list) and len(graph) == 2 for graph in graphs]):

            if 'graph' in self.task_level:
                graph_labels = [pkd for graph, pkd in graphs]
                self.check_graph_labels(graph_labels, pkd_model)

                self.graph_labels = np.array(graph_labels)
                graphs = [graph for graph, pkd in graphs]
            else:
                self.graph_labels = None
        else:
            raise ValueError("Input format not recognized!")

        # remove edges with rmsd greater than cutoff
        if rmsd_cutoff is not None:
            new_graphs = []

            for graph in graphs:
                discarded_edges = filter(lambda e: e[2] > rmsd_cutoff, list(graph.edges.data('rmsd')))
                graph.remove_edges_from(list(discarded_edges))

                new_graphs.append(graph)
            graphs = list(new_graphs)
 
        self.feats, self.adj, self.rmsd, self.node_labels = ([], [], [], [])
        is_node_label = True

        for kdx, graph in enumerate(graphs):
            graph_feats = []
            graph_node_labels = []

            # load features and labels
            for node, data in graph.nodes(data=True):
                feats = []
                for ft in feat_names:
                    feats.append(data[ft])
                graph_feats.append(feats)

                if 'label' in data:
                    if is_node_label:
                        graph_node_labels.append(data['label'])
                else:
                    if 'node' in task_level and training:
                        raise ValueError("node labels (pose correctness) not found. Required for node-level prediction!")
                    is_node_label = False
                    graph_node_labels = None

            self.feats.append(np.array(graph_feats))
            if is_node_label:
                if isinstance(graph_node_labels[0], int):
                    graph_node_labels = np.array(graph_node_labels)
                    graph_node_labels = graph_node_labels[:,np.newaxis]
                else:
                    graph_node_labels = np.array(graph_node_labels)
                self.node_labels.append(graph_node_labels)

            # load adjacency matrix
            adj_matrix = nx.adjacency_matrix(graph).toarray()
            self.adj.append(adj_matrix)

            # load rmsd values as edge feats
            rmsd_matrix = nx.to_numpy_array(graph, weight='rmsd')
            self.rmsd.append(rmsd_matrix)

        if not is_node_label:
            self.node_labels = None

        self.max_nrof_nodes = max([len(graph) for graph in graphs])
        self.nfeats = self.feats[0].shape[1]

        self.nlabels = 1
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
            if self.node_labels is not None:
                node_labels = np.concatenate([self.node_labels[graph_id], np.zeros((diff, self.nlabels), dtype=np.int32)], axis=0)
        else:
            graph_feats = self.feats[graph_id]
            if self.node_labels is not None:
                node_labels = self.node_labels[graph_id]

        if self.node_labels is None:
            node_labels = np.zeros((self.max_nrof_nodes, self.nlabels), dtype=np.int32)

        if self.graph_labels is not None: 
            graph_label = self.graph_labels[graph_id]
        else:
            graph_label = 0.0

        neigh_indices, neigh_adj_values, neigh_rmsd, nneigh = sample_neighbors(adj_matrix, rmsd_matrix, graph_size, depth, nrof_neigh)

        for idx in range(depth):
            if diff > 0:
                # zero padding for each layer until getting max_nrof_nodes nodes per graph
                neigh_indices[idx] = np.concatenate([neigh_indices[idx], np.zeros((diff, nrof_neigh), dtype=np.int32)], axis=0)
                neigh_adj_values[idx] = np.concatenate([neigh_adj_values[idx], np.zeros((diff, nrof_neigh), dtype=np.int32)], axis=0)

                neigh_rmsd[idx] = np.concatenate([neigh_rmsd[idx], np.zeros((diff, nrof_neigh), dtype=np.float32)], axis=0)
                nneigh[idx] = np.concatenate([nneigh[idx], np.zeros((diff), dtype=np.int32)], axis=0)

        return np.asarray(graph_feats, np.float32), graph_size, \
               np.asarray(neigh_indices, dtype=np.int32), np.asarray(neigh_adj_values, dtype=np.float32), \
               np.asarray(neigh_rmsd, dtype=np.float32), np.asarray(nneigh, dtype=np.float32), node_labels, graph_label

    def sample(self, graph_id, depth, nrof_neigh):

        return tf.py_function(self.__sample, [graph_id, depth, nrof_neigh], [tf.float32, tf.int32,
                                   tf.int32, tf.float32, tf.float32, tf.float32, tf.int32, tf.float32])

    def check_graph_labels(self, labels, pkd_model):

        if not isinstance(labels, list):
            labels = [labels]

        if all(isinstance(label, float) for label in labels):
            if pkd_model == 'classification':
                raise ValueError("pkd classification was requested but float values were found for pkd")

        elif all(isinstance(label, int) and (label == 1 or label == 0) for label in labels):
            if pkd_model == 'regression':
                raise ValueError("pKd regression was requested but integers values were found for pKd")
        else:
            raise ValueError("Graph labels should be float numbers (pKds) or integers (active/inactive)")


def generate_data_loader(dataset, depth, nrof_neigh, num_parallel_calls=1, batch_size=1, randomize=True):
    """Create data loader to be fed to model"""

    if randomize:
        data_slices = np.random.permutation(dataset.ngraphs)
    else:
        data_slices = np.arange(dataset.ngraphs)

    data_loader = tf.data.Dataset.from_tensor_slices((data_slices))
    data_loader = data_loader.map(lambda x: dataset.sample(x, depth, nrof_neigh), num_parallel_calls=num_parallel_calls, deterministic=True)

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
