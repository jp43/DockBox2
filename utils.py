import numpy as np

def generate_neighbours(adj, graph_size, depth, nrof_neigh_per_batch):
    indices_all = []
    adj_mask_all = []

    nneigh_all = []

    for idx in range(depth):
        indices = []
        nneigh = []
        adj_mask = []

        for jdx in range(graph_size):
            # get indices of neighbors of node jdx 
            neigh_indices = np.where(adj[jdx] != 0)[0]

            # get indices of non neighbors
            non_neigh_indices = np.setdiff1d(np.arange(graph_size), neigh_indices)

            if len(neigh_indices) > nrof_neigh_per_batch:
                neigh_indices = np.random.choice(neigh_indices, nrof_neigh_per_batch, replace=False)

            _nneigh = len(neigh_indices)
            if _nneigh < nrof_neigh_per_batch:
                neigh_indices = np.concatenate([neigh_indices, [-1]*(nrof_neigh_per_batch - _nneigh)])

            neigh_adj_nodes = adj[jdx][neigh_indices]
            indices.append(neigh_indices)

            nneigh.append(_nneigh)
            adj_mask.append(neigh_adj_nodes)

        indices_all.append(indices)
        adj_mask_all.append(adj_mask)
        nneigh_all.append(nneigh)

    return indices_all, adj_mask_all, nneigh_all

