import pickle

sets = ['train', 'val', 'test']

for setname in sets:
    with open(setname+'_ppi.pickle', "rb") as ff:
        data = pickle.load(ff)

        print(setname.capitalize() + ' set: '+ ', '.join(list(data.keys())))
        print(len(data['Graph_nodes']), len(data['Graph_feats']), len(data['Graph_labels']), len(data['Graph_adj_list']))
 
        for sg in data['Graph_nodes']:
            print(len(sg))

        for feats in data['Graph_feats']:
            print(len(feats))

        for lb in data['Graph_labels']:
            print(len(lb))

        for adj in data['Graph_adj_list']:
            print(adj.shape)
