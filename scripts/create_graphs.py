import sys
import pickle
from glob import glob

from random import seed, sample
import numpy as np
import pandas as pd

import networkx as nx

features = ['autodock', 'vina', 'dock', 'dsx']
df_results = pd.read_csv("docking_results.csv")
df_results = df_results.dropna().reset_index()

# load cog coordinates
df_cog = pd.read_csv("cog_crystal.csv")

pdbids = set(df_results['pdbid'].values)
pdbids = sorted(list(pdbids))

npdbids = len(pdbids)

print("Creating graphs with nodes...")
graphs = {}
for pdbid in pdbids:
    rows_pdbid = df_results[df_results['pdbid']==pdbid]

    G = nx.Graph()
    for idx, row in rows_pdbid.iterrows():
        cog = np.array(row[['cog_x', 'cog_y', 'cog_z']])
        G.add_node(row['pose_idx'], feature=row[features].values, rmsd=row['rmsd'], cog=cog)
    graphs[pdbid] = G

print("Adding edges...")
graphs_with_edges = {}

cogs_crst = {}
rmsdfiles = sorted(glob("rmsd/rmsd*.csv"))
for rmsdfile in rmsdfiles:
    print(rmsdfile)
    df_rmsd = pd.read_csv(rmsdfile)

    for pdbid, rows_pdbid in df_rmsd.groupby('pdbid'):
        if pdbid in graphs:
            G = graphs[pdbid]
            for idx, row in rows_pdbid.iterrows():
                if row['pose_idx'] in G.nodes() and row['pose_jdx'] in G.nodes():
                    G.add_edge(row['pose_idx'], row['pose_jdx'], rmsd=row['value'])

            nrof_nodes = len(G.nodes())
            adj = nx.adjacency_matrix(G).toarray()

            # all the nodes should be connected with each other (excluding self connections)
            assert ((adj + np.identity(nrof_nodes, dtype=int))==1).all()
            graphs_with_edges[pdbid] = G

            row_cog = df_cog[df_cog["pdbid"] == pdbid]
            cogs_crst[pdbid] = row_cog[['cog_x', 'cog_y', 'cog_z']].values

missing_pdbids = set(graphs) - set(graphs_with_edges)
if missing_pdbids:
    print("Warning: could not find edge information (RMSD) for PDBIDs: " + ', '.join(sorted(missing_pdbids)))
else:
    print("Edge information (RMSD) was found for all PDBIDs")

with open('graphs.pickle', "wb") as ff:
    pickle.dump([graphs_with_edges, cogs_crst], ff)
