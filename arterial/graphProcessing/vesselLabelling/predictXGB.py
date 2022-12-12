import networkx as nx
import os
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

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

def singleVesselLabelling(G):
    
    # Define dictionary
    edgeTypesDict ={0: "RICA", 
                1: "LICA", 
                2: "BA", 
                3: "LM1", 
                4: "RM1", 
                5: "LM2", 
                6: "RM2", 
                7: "LA1"}
    # Get DataFrame with relevant features required by the trained XGB model
    X = manageSubdict(G)

    # Load XGB classification model
    model = xgb.Booster({'nthread': 4})  
    model.load_model(os.path.join(os.environ["arterialDir"], 'thrombusDetection/vesselLabelling/model/cerebrallabelling.model'))

    # Predict edges
    result = model.predict(xgb.DMatrix(X)).astype(int)

    # Introduce results in the Graph edges
    for i, (n0, n1) in enumerate(sorted(G.edges)):
        G[n0][n1]['Vessel type'] = result[i]
        G[n0][n1]['Vessel type name'] = edgeTypesDict[result[i]]
        
    return G

def predictVesselTypes(case_dir):
    # Read graph
    G = nx.gpickle.read_gpickle(os.path.join(case_dir, 'graph.pickle'))

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
