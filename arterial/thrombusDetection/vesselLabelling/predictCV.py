import networkx as nx
import os
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import joblib

edge_types_dict ={0: "RICA", 
                1: "LICA", 
                2: "BA", 
                3: "LM1", 
                4: "RM1", 
                5: "LM2", 
                6: "RM2", 
                7: "LA1"}

def manageSubdict(G):
    dictio = {}
    coordKeys = ['direction', 'departure angle', 'proximal bifurcation position', 'distal bifurcation position', 'distal bifurcation position', 'pos']
    n = 0
    for n0, n1 in sorted(G.edges):
        subdict = {}
        for key in G[n0][n1]:
            if key != 'features':
                if key not in coordKeys:
                    subdict[key] = G[n0][n1][key]
                else:
                    subdict[key + '_x'] = np.round(G[n0][n1][key][0], 2)
                    subdict[key + '_y'] = np.round(G[n0][n1][key][1], 2)
                    subdict[key + '_z'] = np.round(G[n0][n1][key][2], 2)
        dictio[str(n0)+ '_' + str(n1)] = subdict
        
    return pd.DataFrame(dictio).T
def manageSubdict(G):
    dictiofin = {}
    for n0, n1 in G.edges:
        subdict = {}
        
        for key in G[n0][n1]['features']:
            subdict[key] = G[n0][n1]['features'][key]
        # subdict['vessel type'] = G[n0][n1]['vessel type']
        # subdict['vessel type name'] = G[n0][n1]['vessel type name']
        subdict['cell_id'] = G[n0][n1]['cell_id']
        
        dictiofin[str(n0)+ '_' + str(n1)] = subdict

    return pd.DataFrame(dictiofin).T

def singleVesselLabelling(G):
    X = manageSubdict(G)
    
    scaler = joblib.load(os.path.join(os.environ["arterial_dir"], 'thrombusDetection/vesselLabelling/model/cerebralScaler.gz')) 
    columns0 = X.columns
    X = pd.DataFrame(scaler.transform(X), columns = columns0)
    
    model = xgb.Booster({'nthread': 4})  
    model.load_model(os.path.join(os.environ["arterial_dir"], 'thrombusDetection/vesselLabelling/model/cerebrallabelling.model'))
    print(X.columns)
    result = model.predict(xgb.DMatrix(X)).astype(int)
    print(result)
    for i, (n0, n1) in enumerate(sorted(G.edges)):
        G[n0][n1]['Vessel type'] = result[i]
        G[n0][n1]['Vessel type name'] = edge_types_dict[result[i]]
        
    return G

def predictVesselTypes(case_dir):
    # Work with cerebral arteries directory
    case_dir = os.path.join(case_dir, 'cerebralArteries')

    # Read graph
    G = nx.gpickle.read_gpickle(os.path.join(case_dir, 'graph_simple.pickle'))
    
    # Edge abelling
    G_label = singleVesselLabelling(G)

    # Save graph    
    nx.gpickle.write_gpickle(G_label, os.path.join(case_dir, 'graph_predXGB.pickle'))

    # Make graph
    _ = plt.figure(figsize = [5, 10])
    ax = plt.gca()
    a, b = 0, 2
    node_pos_dict_P = {}
    for n in G.nodes():
        node_pos_dict_P[n] = [G.nodes(data=True)[n]['pos'][a], G.nodes(data=True)[n]['pos'][b]]
    # Set the edge labels for visualization in the png file
    edge_labels = nx.get_edge_attributes(G_label, 'Vessel type name')
    # We draw the png file with the predicted vessel types
    nx.draw(G_label, node_pos_dict_P, node_size=20)
    nx.draw_networkx_edge_labels(G_label, node_pos_dict_P, edge_labels = edge_labels, ax = ax)
    plt.savefig(os.path.join(case_dir, 'graph_predXGB.png'))
