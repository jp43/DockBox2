import pickle

sets = ['train', 'val', 'test']

for setname in sets:
    with open(setname+'_ppi.pickle', "rb") as ff:
        data = pickle.load(ff)

    print(setname.capitalize()+' set')

    for idx, graph in enumerate(data):
        print("Subgraph %i: %i elements"%(idx+1,len(graph)))

