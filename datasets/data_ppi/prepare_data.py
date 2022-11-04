import os
import sys
import json
import numpy as np
import pickle

import networkx as nx
from networkx.readwrite import json_graph
from sklearn.preprocessing import StandardScaler

def compute_adj_from_subgraph(graph, subgraph):
    nnodes = len(subgraph)

    adj = np.zeros((nnodes,nnodes))
    for idx, idx_node in enumerate(subgraph):
        for jdx, jdx_node in enumerate(subgraph):
            if idx > jdx:
                is_edge = graph.has_edge(idx_node, jdx_node)
                adj[idx, jdx] = int(is_edge)
                adj[jdx, idx] = int(is_edge) # graph is not directional
    return adj

prefix = 'ppi'
normalize = True

G_data = json.load(open(prefix + "-G.json"))
G = json_graph.node_link_graph(G_data)

# loads features
feats = np.load(prefix + "-feats.npy")

# load id maps
id_map = {node: node for node in G.nodes()}

# load labels
class_map = {node: value['label'] for node, value in G.nodes().items()}

#if normalize:
#    train_ids = np.array([id_map[idx] for idx in G.nodes() if not G.nodes[idx]['val'] and not G.nodes[idx]['test']])
#    scaler = StandardScaler()
#    scaler.fit(feats[train_ids])
#    feats = scaler.transform(feats)

subgraphs = list(nx.connected_components(G))

sets = {'train': {}, 'val': {}, 'test': {}}
for key in sets:
    for att in ['nodes', 'feats', 'adj_list', 'labels',  'degree']:
        sets[key][att] = []

for kdx, sg in enumerate(subgraphs):
    is_val_test = {'val': [], 'test': []}

    for idx in sg:
        node = G.nodes[idx]
        for setname in ['val', 'test']:
            if node[setname]:
                is_in_set = True
            else:
                is_in_set = False
            is_val_test[setname].append(is_in_set)

    graph_size = len(sg)
    is_val = any(is_val_test['val'])
    is_test = any(is_val_test['test'])

    print("Subgraph %i (%i elements): "%(kdx+1,graph_size), ' Val: '+str(is_val) + ' ', 'Test: '+str(is_test))

    if graph_size > 2:
        if not is_val and not is_test:
            setname = 'train'
        elif is_val and not is_test:
            setname = 'val'
        elif not is_val and is_test:
            setname = 'test'
        else:
            continue
        sets[setname]['nodes'].append(sg)
        sets[setname]['feats'].append(np.array([G.nodes[idx]['feature'] for idx in sg]))

        adj = compute_adj_from_subgraph(G, sg)
        assert((adj == adj.transpose()).all())
        sets[setname]['adj_list'].append(adj)
        sets[setname]['labels'].append(np.array([G.nodes[idx]['label'] for idx in sg]))
        sets[setname]['degree'].append([len(list(G.neighbors(idx))) for idx in sg])
    else:
        continue

for setname in sets:
    sets[setname]['max_nrof_nodes'] = max([len(sg) for sg in sets[setname]['nodes']])
 
    with open(setname+'_ppi.pickle', "wb") as ff:
        pickle.dump(sets[setname], ff)
