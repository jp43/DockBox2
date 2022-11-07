import os
import sys
import json
import numpy as np
import pickle

import networkx as nx
from networkx.readwrite import json_graph
from sklearn.preprocessing import StandardScaler

normalize = True

G_data = json.load(open("ppi-G.json"))
G = json_graph.node_link_graph(G_data)

#if normalize:
#    train_ids = np.array([id_map[idx] for idx in G.nodes() if not G.nodes[idx]['val'] and not G.nodes[idx]['test']])
#    scaler = StandardScaler()
#    scaler.fit(feats[train_ids])
#    feats = scaler.transform(feats)

subgraphs_idxs = list(nx.connected_components(G))
sets = {'train': [], 'val': [], 'test': []}

for kdx, sg_idxs in enumerate(subgraphs_idxs):
    is_val_test = {'val': [], 'test': []}

    for idx in sg_idxs:
        node = G.nodes[idx]
        for setname in ['val', 'test']:
            if node[setname]:
                is_in_set = True
            else:
                is_in_set = False
            is_val_test[setname].append(is_in_set)

    graph_size = len(sg_idxs)
    is_val = any(is_val_test['val'])
    is_test = any(is_val_test['test'])

    print("Subgraph %i (%i elements):"%(kdx+1,graph_size), 'Val: '+str(is_val), 'Test: '+str(is_test))

    if graph_size > 2:
        if not is_val and not is_test:
            setname = 'train'
        elif is_val and not is_test:
            setname = 'val'
        elif not is_val and is_test:
            setname = 'test'
        else:
            continue
        sets[setname].append(G.subgraph(sg_idxs))
    else:
        continue

for setname in sets:
    with open(setname+'_ppi.pickle', "wb") as ff:
        pickle.dump(sets[setname], ff)
