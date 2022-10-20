import numpy as np

def sample_neighbors(adj, graph_size, depth, nrof_neigh_per_batch):
    neigh_indices = []
    adj_mask = []
    nneigh = []

    for idx in range(depth):
        graph_neigh_indices = []
        graph_adj_mask = []
        graph_nneigh = []

        for jdx in range(graph_size):
            # get indices of neighbors of node jdx 
            node_neigh_indices = np.where(adj[jdx] != 0)[0]

            # get indices of non neighbors
            node_non_neigh_indices = np.setdiff1d(np.arange(graph_size), node_neigh_indices)

            # if number of neighbors is too high, select randomly until having "nrof_neigh_per_batch" neighbors
            if len(node_neigh_indices) > nrof_neigh_per_batch:
                node_neigh_indices = np.random.choice(node_neigh_indices, nrof_neigh_per_batch, replace=False)

            node_nneigh = len(node_neigh_indices)
            if node_nneigh < nrof_neigh_per_batch: # zero padding until getting "nrof_neigh_per_batch" neighbors
                node_neigh_indices = np.concatenate([node_neigh_indices, [node_non_neigh_indices[0]]*(nrof_neigh_per_batch - node_nneigh)])

            graph_neigh_indices.append(node_neigh_indices)
            graph_adj_mask.append(adj[jdx][node_neigh_indices])
            graph_nneigh.append(node_nneigh)

        neigh_indices.append(graph_neigh_indices)
        adj_mask.append(graph_adj_mask)
        nneigh.append(graph_nneigh)

    return neigh_indices, adj_mask, nneigh

