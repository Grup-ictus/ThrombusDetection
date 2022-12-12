from fileinput import filename
from tkinter import Y
import networkx as nx
import matplotlib.pyplot as plt
import os
import imageio.v2 as io
import numpy as np
import pickle


def makeGraphPlot(caseDir, G, filename, label = None, X = 0, Y = 1):
    # Generate plot of dense graph for quick visualization
    _ = plt.figure(figsize = [14, 16])
    ax = plt.gca()

    # In order to place the nodes in the visualization of the graph in a sagittal view, we use L and S coordinates (the view will be from the coronal plane, P axis)
    node_pos_dict_P = {}
    for n in G.nodes():
        node_pos_dict_P[n] = [G.nodes(data=True)[n]["pos"][X], G.nodes(data=True)[n]["pos"][Y]]

    if label is not None:
        edge_labels = nx.get_edge_attributes(G, label)
        nx.draw(G, node_pos_dict_P, node_size=20, ax=ax)
        nx.draw_networkx_edge_labels(G, node_pos_dict_P, edge_labels = edge_labels, ax=ax)
    else:
        nx.draw(G, node_pos_dict_P, node_size=20, ax=ax)

    if not os.path.exists(os.path.join(caseDir, 'pngGraph')): os.mkdir(os.path.join(caseDir, 'pngGraph'))

    plt.savefig(os.path.join(caseDir, 'pngGraph', filename))
    plt.close()

def make3DPlot(caseDir, finalSegmentsArray, filename):
    _ = plt.figure()
    ax = plt.axes(projection='3d')
    for idx in range(len(finalSegmentsArray)):
        ax.plot(-finalSegmentsArray[idx, 0][:, 0], finalSegmentsArray[idx, 0][:, 1], finalSegmentsArray[idx, 0][:, 2])
    ax.legend(range(len(finalSegmentsArray)))
    
    plt.savefig(os.path.join(caseDir, 'pngGraph', filename))
    plt.close()

def makePlanesImages(caseDir):
    
    # Load graph
    casePath = os.path.join(caseDir, 'graph.pickle')
    G = 0
    G = nx.readwrite.gpickle.read_gpickle(casePath)
    segmentsArray = np.load(os.path.join(caseDir, 'segmentsArray.npy'), allow_pickle = True)

    # Make plot to label the graph
    makeGraphPlot(caseDir, G, filename = 'graphAxial.png', label = 'cellId', X = 0, Y = 1)
    makeGraphPlot(caseDir, G, filename = 'graphCoronal.png', label = 'cellId', X = 1, Y = 2)
    makeGraphPlot(caseDir, G, filename = 'graphSagital.png', label = 'cellId', X = 0, Y = 2)
    make3DPlot(caseDir, segmentsArray, 'graph3D.png')

    # Generate overall figure
    pngPath = os.path.join(caseDir, "pngGraph")
    Axial = io.imread(pngPath +'/graphAxial.png')
    Coronal = io.imread(pngPath +'/graphCoronal.png')
    Sagital = io.imread(pngPath +'/graphSagital.png')
    triD = io.imread(pngPath +'/graph3D.png') 

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (20, 20))
    ax1.imshow(Axial)
    ax2.imshow(Coronal)
    ax3.imshow(Sagital)
    ax4.imshow(triD)
    plt.tight_layout()
    plt.savefig(os.path.join(pngPath, 'overall_fig'))
    plt.close()


databaseDir = "/home/marc/Desktop/Environment/databaseInference/"

for case in sorted(os.listdir(databaseDir)):    
            
    print()
    print(f'Labeling {case}')
    print()
    caseDir = os.path.join(databaseDir, case, 'cerebralArteries')


    graph_path = os.path.join(caseDir, "graph.pickle")
    if os.path.exists(graph_path):

        G = 0
        G = nx.readwrite.gpickle.read_gpickle(graph_path)


        vesselTypesDict = {
            0: "other", 
            1: "RVA", 
            2: "LVA", 
            3: "RICA", 
            4: "LICA", 
            5: "BA", 
            6: "LM1", 
            7: "RM1", 
            8: "LM2", 
            9: "RM2", 
            10: "LA1", 
            11: "RA1"}

        if os.path.isfile(os.path.join(caseDir, "graph_label.pickle")):
            print("Label is already made")
            try:
                G = nx.readwrite.gpickle.read_gpickle(os.path.join(caseDir, "graph_label.pickle"))
            except:
                with open(os.path.join(caseDir, "graph_label.pickle"), "rb") as pklFile:
                    G = pickle.load(pklFile)
        else:
            makePlanesImages(caseDir)
            for n0, n1 in G.edges:
                print("Edge", G[n0][n1]["cellId"])
                G[n0][n1]["vessel type"] = int(input("This edge's type is: "))
                G[n0][n1]["vessel type name"] = vesselTypesDict[G[n0][n1]["vessel type"]]

            nx.readwrite.gpickle.write_gpickle(G, os.path.join(caseDir, "graph_label.pickle"), protocol=4)

            nx.readwrite.gpickle.write_gpickle(G, os.path.join(caseDir, "graph_label.pickle"))

            _ = plt.figure(figsize = [5, 10])
            ax = plt.gca()

            # _, ax = plt.subplots(1, 2, figsize=[5, 10])

            node_pos_dict_P = {}
            for n in G.nodes():
                node_pos_dict_P[n] = [G.nodes(data=True)[n]["pos"][0], G.nodes(data=True)[n]["pos"][2]]

            edge_labels = nx.get_edge_attributes(G, 'vessel type name')

            nx.draw(G, node_pos_dict_P, node_size=20, ax = ax)
            nx.draw_networkx_edge_labels(G, node_pos_dict_P, edge_labels = edge_labels)

            # nx.draw(G, node_pos_dict_R, node_size=20)
            # nx.draw_networkx_edge_labels(G, node_pos_dict_R, edge_labels = edge_labels)

            plt.savefig(os.path.join(caseDir, "graph_label.png"))
            # plt.close()
            plt.show()

