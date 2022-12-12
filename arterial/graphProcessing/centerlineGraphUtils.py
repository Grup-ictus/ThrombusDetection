# -*-coding:Latin-1 -*
import os
import numpy as np
import networkx as nx

import nibabel as nib
from vtk.util.numpy_support import vtk_to_numpy

import matplotlib.pyplot as plt
from mycolorpy import colorlist as mcp

import math

import warnings
warnings.filterwarnings('ignore')

def getHierarchicalOrderingDense(G, access = "femoral", startNode = 0):
    ''' Computes hierarchization of graph. Associates each node to an index that
    indicates the number of nodes to the closest startpoint.

    Arguments:
        - G <nx.Graph>: graph
        - access <str>: can either be femoral or radial
        - startNode: 
    
    '''
    # We need a boolean variable to see if the analysis is finished
    hierarchyDone = False
    # We use this list to get the source nodes with the same hierarchy value at each iteration
    sourceNodes = [startNode]
    # We use this list to avoid repetition of any already analyzed nodes
    usedNodes = [startNode]
    # Hierarchy of startNode is 0
    G.nodes[startNode]["hierarchy {}".format(access)] = 0
    # Initialize hierarchy value
    hierarchy = 1 

    # Start hierarchization
    while not hierarchyDone:
        # Target nodes will be neighbors of sourceNodes that have not been yet analyzed (not in usedNodes)
        targetNodes = []
        # Loop through sourceNodes (nodes that share hierarchy value)
        for src in sourceNodes:
            # Append them to usedNodes
            usedNodes.append(src)
            # Check neighbors that have not been used yet
            for dst in G.neighbors(src):
                if dst not in usedNodes:
                    # Attribute them the corresponding hierarchy index
                    G.nodes[dst]["hierarchy {}".format(access)] = hierarchy
                    # Store them for next iteration
                    targetNodes.append(dst)
        # Update hierarchy index
        hierarchy += 1
        # Pass previous targetNodes to sourceNodes of next iteration
        sourceNodes = targetNodes.copy()

        # Check if analysis is finished
        # If no more target nodes are present and the number of used nodes is equal to the number of nodes in the graph, the analysis is done
        if len(targetNodes) == 0 and len(usedNodes) >= len(G.nodes()):
            hierarchyDone = True
        # If no more target nodes are present but there are still unused nodes, get the first unused node and attribute it with the following hierarchy value
        elif len(targetNodes) == 0 and len(usedNodes) < len(G.nodes()):
            for node in G.nodes():
                if node not in usedNodes:
                    # Add node to source nodes
                    sourceNodes = [node]
                    # Add hierarchy value to graph node
                    G.nodes[startNode]["hierarchy {}".format(access)] = hierarchy
                    # Update hierarchy index
                    hierarchy += 1
                    break

    for src, dst in G.edges:
        if G.nodes[src]["hierarchy {}".format(access)] < G.nodes[dst]["hierarchy {}".format(access)]:
            G[src][dst]["hierarchy {}".format(access)] = G.nodes[src]["hierarchy {}".format(access)]
        else:
            G[src][dst]["hierarchy {}".format(access)] = G.nodes[dst]["hierarchy {}".format(access)]

    return G

# def extractEdgeFeatures(G, niftiCTA, access = "femoral", edgesToRemove = None, cellIdToVesselType = None, featureExtractionForVesselLabelling = False):

#     # Get CTA data from nifti
#     arrayCTA = niftiCTA.get_fdata()

#     # If edges were artificially added for hierarchical ordering, remove them before feature extraction
#     if edgesToRemove is not None:
#         removedEdgesAttributes = []
#         for src, dst in edgesToRemove:
#             auxDict = {}
#             for key in G[src][dst].keys():
#                 auxDict[key] = G[src][dst][key]
#             removedEdgesAttributes.append(auxDict)
#             G.remove_edge(src, dst)

#     for src, dst in G.edges:
#         G[src][dst]["isArtificial"] = False
#         # Define empty feature array
#         G[src][dst]["features {}".format(access)] = {}
#         # Get positions and radius information for centerline points over graph edge
#         segmentPoints = G[src][dst]["segmentsCoordinateArray"]
#         # Eliminate repeated points
#         segmentPoints, indices = np.unique(segmentPoints, axis = 0, return_index=True)
#         segmentRadius = G[src][dst]["segmentsRadiusArray"][indices]
#         # If hierarchy does not flow in the direction of node placement, flip segment arrays
#         if np.sign(src - dst) != np.sign(G.nodes[src]["hierarchy {}".format(access)] - G.nodes[dst]["hierarchy {}".format(access)]):
#             segmentPoints = np.flip(segmentPoints, axis = 0)
#             segmentRadius = np.flip(segmentRadius)

#         # Mean radius
#         G[src][dst]["features {}".format(access)]["Mean radius"] = np.mean(segmentRadius)
#         # Maximum radius 
#         G[src][dst]["features {}".format(access)]["Maximum radius"] = np.max(segmentRadius)
#         # Minimum radius
#         G[src][dst]["features {}".format(access)]["Minimum radius"] = np.min(segmentRadius)
#         # Min/Max radius ratio
#         G[src][dst]["features {}".format(access)]["Min/Max radius ratio"] = G[src][dst]["features {}".format(access)]["Minimum radius"] / G[src][dst]["features {}".format(access)]["Maximum radius"]

#         # Curvature
#         try:
#             radiusOfCurvatureArray = computeCurvature(segmentPoints)
#             if len(radiusOfCurvatureArray) > 7:
#                 G[src][dst]["features {}".format(access)]["Maximum curvature"] = 1 / np.amin(radiusOfCurvatureArray[2:-2])
#             else:
#                 G[src][dst]["features {}".format(access)]["Maximum curvature"] = 1 / np.amin(radiusOfCurvatureArray)
#         except:
#             # If not enough points are found within the edge to compute gradients, set to nan and it will automatically set the closest edge value
#             G[src][dst]["features {}".format(access)]["Maximum curvature"] = math.nan

#         # Segment length
#         G[src][dst]["features {}".format(access)]["Segment length"] = sum([np.linalg.norm(segmentPoints[idx - 1] - segmentPoints[idx]) for idx in range(1, len(segmentPoints))])
#         # Number of points
#         G[src][dst]["features {}".format(access)]["Number of points"] = len(segmentPoints)
#         # Average distance between points
#         G[src][dst]["features {}".format(access)]["Average distance between points"] = G[src][dst]["features {}".format(access)]["Segment length"] / len(segmentPoints)
#         # Relative length
#         G[src][dst]["features {}".format(access)]["Relative length"] = np.linalg.norm(segmentPoints[-1] - segmentPoints[0]) / G[src][dst]["features {}".format(access)]["Segment length"]

#         # Directional features
#         segmentDirection = segmentPoints[-1] - segmentPoints[0]
#         module, polar, azimuth = sphericalAnglesFrom3DCartesian(segmentDirection)
#         # Module
#         G[src][dst]["features {}".format(access)]["Direction module"] = module
#         # Polar
#         G[src][dst]["features {}".format(access)]["Direction polar"] = polar
#         # Azimuth
#         G[src][dst]["features {}".format(access)]["Direction azimuth"] = azimuth

#         segmentHU = []
#         for pointPosition in segmentPoints:
#             segmentHU.append(arrayCTA[np.round(pointPosition).astype(int)[0], np.round(pointPosition).astype(int)[1], np.round(pointPosition).astype(int)[2]])
#         G[src][dst]["features {}".format(access)]["Min HU in segment"] = min(segmentHU)
#         G[src][dst]["features {}".format(access)]["Max HU in segment"] = max(segmentHU)

#         # When available, we also add the vessel label as node feature
#         if cellIdToVesselType is not None:
#             G[src][dst]["features {}".format(access)]["Vessel type"] = cellIdToVesselType[G[src][dst]["cellId"]]
            
#     # We finally check all edges not to have any nan or inf values
#     # If present, we choose the value from the neighboring edges
#     # We first check fist node (there is only one node with hierarchy equal to 0). This way, we ensure no error propagation and
#     # that all edges will not have nan or inf feature values:
#     for src, dst in G.edges:
#         if G[src][dst]["hierarchy {}".format(access)] == 0:
#             break
#     for featureKey in G[src][dst]["features {}".format(access)].keys():
#         # We create auxiliary node variables in case we have to look further than the first-degree neighborhood 
#         currentSrc = src
#         currentDst = dst
#         while math.isnan(G[currentSrc][currentDst]["features {}".format(access)][featureKey]) or math.isinf(G[currentSrc][currentDst]["features {}".format(access)][featureKey]):
#             for neighbor in G.neighbors(currentSrc):
#                 if neighbor != currentDst:
#                     # Now we choose first following node to also include first node
#                     if G[currentSrc][neighbor]["hierarchy {}".format(access)] > G[currentSrc][currentDst]["hierarchy {}".format(access)] and not math.isnan(G[currentSrc][neighbor]["features {}".format(access)][featureKey]) and not math.isinf(G[currentSrc][neighbor]["features {}".format(access)][featureKey]):
#                         print("        ", featureKey, currentSrc, neighbor, G[currentSrc][neighbor]["features {}".format(access)][featureKey])
#                         G[src][dst]["features {}".format(access)][featureKey] = G[currentSrc][neighbor]["features {}".format(access)][featureKey]
#                         break

#             # We update currentDst and currentSrc in case we do not find valid values for the nan or inf features. Search will continue from edge to edge until we find closest edge with valid values
#             # This is very unlikely to continue further than one iteration due to the low frequency of nan or inf values, but we are inclusive just in case
#             currentDst = currentSrc
#             currentSrc = neighbor
    
#     # Now we check all other edges (differently from looking at the first edge, we look at nodes with lower hierarchy)
#     for src, dst in G.edges:
#         if G[src][dst]["hierarchy {}".format(access)] > 0:
#             for featureKey in G[src][dst]["features {}".format(access)].keys():
#                 # We create auxiliary node variables in case we have to look further than the first-degree neighborhood 
#                 currentSrc = src
#                 currentDst = dst
#                 while math.isnan(G[src][dst]["features {}".format(access)][featureKey]) or math.isinf(G[src][dst]["features {}".format(access)][featureKey]):
#                     for neighbor in G.neighbors(currentSrc):
#                         # Now we choose first following node to also include first node
#                         if G[currentSrc][neighbor]["hierarchy {}".format(access)] <= G[currentSrc][currentDst]["hierarchy {}".format(access)] and not math.isnan(G[currentSrc][neighbor]["features {}".format(access)][featureKey]) and not math.isinf(G[currentSrc][neighbor]["features {}".format(access)][featureKey]):
#                             print("        ", featureKey, currentSrc, neighbor, G[currentSrc][neighbor]["features {}".format(access)][featureKey])
#                             G[src][dst]["features {}".format(access)][featureKey] = G[currentSrc][neighbor]["features {}".format(access)][featureKey]
#                             break
#                     # We update currentDst and currentSrc in case we do not find valid values for the nan or inf features. Search will continue from edge to edge until we find closest edge with valid values
#                     # This is very unlikely to continue further than one iteration due to the low frequency of nan or inf values, but we are inclusive just in case
#                     currentDst = currentSrc
#                     currentSrc = neighbor

#     # Add removed edges back
#     if edgesToRemove is not None:
#         for idx, edge in enumerate(edgesToRemove):
#             src, dst = edge
#             G.add_edge(src, dst, cellId = G.nodes[dst]["cellId"])
#             G[src][dst]["isArtificial"] = True
#             for key in removedEdgesAttributes[idx].keys():
#                 G[src][dst][key] = removedEdgesAttributes[idx][key]
#             G[src][dst]["features {}".format(access)] = {}
#             G[src][dst]["features {}".format(access)]["Segment length"] = np.linalg.norm(G.nodes[src]["pos"] - G.nodes[dst]["pos"])

#     if featureExtractionForVesselLabelling:
#         for src, dst in G.edges:
#             G[src][dst]["features"] = []
#             for key in G[src][dst]["features femoral"]:
#                  G[src][dst]["features"].append(G[src][dst]["features femoral"][key])
#             G[src][dst]["features"] = np.array(G[src][dst]["features"])
#             G[src][dst].pop("hierarchy femoral")
#             G[src][dst].pop("features femoral")

#     return G

# def extractNodeFeatures(G, niftiCTA, branchModel, access = "femoral", edgesToRemove = None, featureExtractionForVesselLabelling = False):
#     # Get CTA data from nifti
#     arrayCTA = niftiCTA.get_fdata()
#     # Translation
#     aff = niftiCTA.affine
#     translation = np.transpose(aff[:3, 3])

#     # Pool branchModel point coordinates. Get blanking for each point
#     branchModelCoordinates = np.ndarray([branchModel.GetNumberOfPoints(), 3])
#     blanking = np.ndarray([branchModel.GetNumberOfPoints()])
#     accumulatedNumberOfPoints = 0
#     for idx in range(branchModel.GetNumberOfCells()):         
#         for idx2 in range(branchModel.GetCell(idx).GetNumberOfPoints()):
#             branchModelCoordinates[idx2 + accumulatedNumberOfPoints] = branchModel.GetCell(idx).GetPoints().GetPoint(idx2) - translation
#             branchModelCoordinates[idx2 + accumulatedNumberOfPoints][0] = - branchModelCoordinates[idx2 + accumulatedNumberOfPoints][0]
#             blanking[idx2 + accumulatedNumberOfPoints] = vtk_to_numpy(branchModel.GetCellData().GetArray("Blanking"))[idx]
#         accumulatedNumberOfPoints += branchModel.GetCell(idx).GetNumberOfPoints()

#     # If edges were artificially added for hierarchical ordering, remove them before feature extraction
#     if edgesToRemove is not None:
#         removedEdgesAttributes = []
#         for src, dst in edgesToRemove:
#             auxDict = {}
#             for key in G[src][dst].keys():
#                 auxDict[key] = G[src][dst][key]
#             removedEdgesAttributes.append(auxDict)
#             G.remove_edge(src, dst)

#     for node in G:
#         # Define empty feature array
#         G.nodes[node]["features {}".format(access)] = {}

#         # Hierarchy
#         G.nodes[node]["features {}".format(access)]["Hierarchy"] = G.nodes[node]["hierarchy {}".format(access)]
#         # Node position
#         G.nodes[node]["features {}".format(access)]["Node position i"] = G.nodes[node]["pos"][0]
#         G.nodes[node]["features {}".format(access)]["Node position j"] = G.nodes[node]["pos"][1]
#         G.nodes[node]["features {}".format(access)]["Node position k"] = G.nodes[node]["pos"][2]
#         # Degree
#         G.nodes[node]["features {}".format(access)]["Degree"] = G.degree(node)
#         # Radius
#         G.nodes[node]["features {}".format(access)]["Radius"] = G.nodes[node]["radius"]

#         # Curvature
#         if G.degree(node) == 1:
#             for nodeEnd in G.neighbors(node):
#                 # During the graph building phase, we stored the centerline points between each graph node
#                 # We only want to keep those closer to the node as part of the feature computations
#                 segmentPoints = G[node][nodeEnd]["segmentsCoordinateArray"]
#                 # Eliminate repeated points
#                 segmentPoints = np.unique(segmentPoints, axis = 0)
#                 # Compute curvature
#                 try:
#                     radiusOfCurvatureArray = computeCurvature(segmentPoints)
#                     G.nodes[node]["features {}".format(access)]["Curvature at node"] = 1 / radiusOfCurvatureArray[len(radiusOfCurvatureArray) // 2]
#                 except:
#                     # If not enough points are found within the edge to compute gradients, set to nan and it will automatically set the closest node value
#                     G.nodes[node]["features {}".format(access)]["Curvature at node"] = math.nan
#         elif G.degree(node) == 2:
#             # We define these empty arrays in case there is a rare event like a startpoint originating two centerlines (extremely rare)
#             segmentPointsProx = np.ndarray([0, 3])
#             segmentPointsDist = np.ndarray([0, 3])
#             for nodeAux in G.neighbors(node):
#                 if G.nodes[nodeAux]["hierarchy {}".format(access)] < G.nodes[node]["hierarchy {}".format(access)]:
#                     nodeProx = nodeAux
#                     # During the graph building phase, we stored the centerline points between each graph node
#                     # We only want to keep those closer to the node as part of the feature computations
#                     segmentPointsProx = G[node][nodeProx]["segmentsCoordinateArray"]
#                     # We have to see the direction for the hierarchization in comparison to the centerline to see if we have to flip the position and radius arrays 
#                     # for a correct computation of the directional features
#                     if np.sign(node - nodeProx) != np.sign(G.nodes[node]["hierarchy {}".format(access)] - G.nodes[nodeProx]["hierarchy {}".format(access)]):
#                         segmentPointsProx = np.flip(segmentPointsProx, axis = 0)
#                 elif G.nodes[nodeAux]["hierarchy {}".format(access)] > G.nodes[node]["hierarchy {}".format(access)]:
#                     nodeDist = nodeAux
#                     # During the graph building phase, we stored the centerline points between each graph node
#                     # We only want to keep those closer to the node as part of the feature computations
#                     segmentPointsDist = G[node][nodeDist]["segmentsCoordinateArray"]
#                     # We have to see the direction for the hierarchization in comparison to the centerline to see if we have to flip the position and radius arrays 
#                     # for a correct computation of the directional features
#                     if np.sign(node - nodeDist) != np.sign(G.nodes[node]["hierarchy {}".format(access)] - G.nodes[nodeDist]["hierarchy {}".format(access)]):
#                         segmentPointsDist = np.flip(segmentPointsDist, axis = 0)
#             # Finally we concatenate both the coordinates and radius of both ends of the segment
#             segmentPoints = np.append(segmentPointsProx, segmentPointsDist, axis = 0)
#             # Eliminate repeated points
#             segmentPoints = np.unique(segmentPoints, axis = 0)
#                 # Compute curvature
#             try:
#                 radiusOfCurvatureArray = computeCurvature(segmentPoints)
#                 G.nodes[node]["features {}".format(access)]["Curvature at node"] = 1 / radiusOfCurvatureArray[np.argmin(np.linalg.norm(segmentPoints - G.nodes[node]["pos"], axis = 1))]
#             except:
#                 # If not enough points are found within the edge to compute gradients, set to nan and it will automatically set the closest node value
#                 G.nodes[node]["features {}".format(access)]["Curvature at node"] = math.nan
#         # Multi-furcations (degree > 2)
#         elif G.degree[node] > 2:
#             # We treat multi-furcation as endpoints (since there is no preference a priori for the path that needs to be taken). We only look at the preceeding node
#             # First we search the neighboring nodes and we get the candidate centerline points for segment feature extraction from the preceeding node
#             # To find the preceeding node, we check the hierarchical order (there should be one node with a lower hierarchical index than the bifurcation point)
#             # We have to initialize the segment arrays just in case we are in the rare event of a multifurcation with hierarchy = 0
#             segmentPoints = np.ndarray([0, 3])
#             for nodeAux in G.neighbors(node):
#                 if G.nodes[nodeAux]["hierarchy {}".format(access)] < G.nodes[node]["hierarchy {}".format(access)]:
#                     nodeProx = nodeAux
#                     # During the graph building phase, we stored the centerline points between each graph node
#                     # We only want to keep those closer to the node as part of the feature computations
#                     segmentPoints = G[node][nodeProx]["segmentsCoordinateArray"]
#                     # We have to see the direction for the hierarchization in comparison to the centerline to see if we have to flip the position and radius arrays 
#                     # for a correct computation of the directional features
#                     if np.sign(node - nodeProx) != np.sign(G.nodes[node]["hierarchy {}".format(access)] - G.nodes[nodeProx]["hierarchy {}".format(access)]):
#                         segmentPoints = np.flip(segmentPoints, axis = 0)
#             # Multi-furcation as start node. We just take the first neighbor to compute all variables. In this case, we invert the roles of node and nodeProx
#             if len(segmentPoints) == 0:
#                 originalNode = node
#                 for nodeAux in G.neighbors(node):
#                     # During the graph building phase, we stored the centerline points between each graph node
#                     # We only want to keep those closer to the node as part of the feature computations
#                     segmentPoints = G[node][nodeAux]["segmentsCoordinateArray"]
#                     break

#             # Eliminate repeated points
#             segmentPoints = np.unique(segmentPoints, axis = 0)
#             # Compute curvature
#             try:
#                 radiusOfCurvatureArray = computeCurvature(segmentPoints)
#                 G.nodes[node]["features {}".format(access)]["Curvature at node"] = 1 / radiusOfCurvatureArray[len(radiusOfCurvatureArray) // 2]
#             except:
#                 # If not enough points are found within the edge to compute gradients, set to nan and it will automatically set the closest node value
#                 G.nodes[node]["features {}".format(access)]["Curvature at node"] = math.nan

#         # Blanking
#         G.nodes[node]["features {}".format(access)]["Blanking"] = blanking[findPointId(G.nodes[node]["pos"], branchModelCoordinates)]
#         # HU at node
#         G.nodes[node]["features {}".format(access)]["HU at node"] = arrayCTA[np.round(G.nodes[node]["pos"]).astype(int)[0], np.round(G.nodes[node]["pos"]).astype(int)[1], np.round(G.nodes[node]["pos"]).astype(int)[2]]

#     # Add removed edges back
#     if edgesToRemove is not None:
#         for idx, edge in enumerate(edgesToRemove):
#             src, dst = edge
#             G.add_edge(src, dst, cellId = G.nodes[dst]["cellId"])
#             G[src][dst]["isArtificial"] = True
#             for key in removedEdgesAttributes[idx].keys():
#                 G[src][dst][key] = removedEdgesAttributes[idx][key]

#     # We finally check all nodes not to have any nan or inf values
#     # If present, we choose the value from the neighboring nodes
#     # We first check fist node (there is only one node with hierarchy equal to 0). This way, we ensure no error propagation and
#     # that all nodes will not have nan or inf feature values:
#     for node in G:
#         if G.nodes[node]["hierarchy {}".format(access)] == 0:
#             break
#     for featureKey in G.nodes[node]["features {}".format(access)].keys():
#         # We create an auxiliary node variable in case we have to look further than the first-degree neighborhood 
#         currentNode = node
#         while math.isnan(G.nodes[node]["features {}".format(access)][featureKey]) or math.isinf(G.nodes[node]["features {}".format(access)][featureKey]):
#             for neighbor in G.neighbors(currentNode):
#                 # Now we choose first following node to also include first node
#                 if G.nodes[currentNode]["hierarchy {}".format(access)] < G.nodes[neighbor]["hierarchy {}".format(access)] and not math.isnan(G.nodes[neighbor]["features {}".format(access)][featureKey]) and not math.isinf(G.nodes[neighbor]["features {}".format(access)][featureKey]):
#                     # print("        ", featureKey, node, neighbor, G.nodes[neighbor]["features {}".format(access)][featureKey])
#                     G.nodes[node]["features {}".format(access)][featureKey] = G.nodes[neighbor]["features {}".format(access)][featureKey]
#                     break

#             # We update currentnode in case we do not find valid values for the nan or inf features. Search will continue from node to node until we find closes node with valid values
#             # This is very unlikely to continue further than one node doe to the low frequency of nan or inf values, but we are inclusive just in case
#             currentNode = neighbor
    
#     # Now we check all other nodes (differently from looking at the first node, we look at nodes with lower hierarchy)
#     for node in G:
#         if G.nodes[node]["hierarchy {}".format(access)] > 0:
#             for featureKey in G.nodes[node]["features {}".format(access)].keys():
#                 # We create an auxiliary node variable in case we have to look further than the first-degree neighborhood 
#                 currentNode = node
#                 while math.isnan(G.nodes[node]["features {}".format(access)][featureKey]) or math.isinf(G.nodes[node]["features {}".format(access)][featureKey]):
#                     for neighbor in G.neighbors(currentNode):
#                         # Now we choose first following node to also include first node
#                         if G.nodes[currentNode]["hierarchy {}".format(access)] > G.nodes[neighbor]["hierarchy {}".format(access)] and not math.isnan(G.nodes[neighbor]["features {}".format(access)][featureKey]) and not math.isinf(G.nodes[neighbor]["features {}".format(access)][featureKey]):
#                             # print("        ", featureKey, node, neighbor, G.nodes[neighbor]["features {}".format(access)][featureKey])
#                             G.nodes[node]["features {}".format(access)][featureKey] = G.nodes[neighbor]["features {}".format(access)][featureKey]
#                         # We update currentnode in case we do not find valid values for the nan or inf features. Search will continue from node to node until we find closes node with valid values
#                         # This is very unlikely to continue further than one node doe to the low frequency of nan or inf values, but we are inclusive just in case
#                         elif G.nodes[currentNode]["hierarchy {}".format(access)] > G.nodes[neighbor]["hierarchy {}".format(access)]:
#                             currentNode = neighbor

#     if not featureExtractionForVesselLabelling:
#         # Accumulative features
#         for hierarchy in range(getMaxHierarchy(G, access) + 1):
#             for node in G:
#                 if G.nodes[node]["hierarchy {}".format(access)] == hierarchy:
#                     # For the first node, it is just 0
#                     if hierarchy == 0:
#                         G.nodes[node]["features {}".format(access)]["Accumulated length from access"] = 0.0
#                     else:
#                         for nodeAux in G.neighbors(node):
#                             # For successive nodes, we add the segment lenght to the previously accumulated length from access
#                             if G.nodes[node]["hierarchy {}".format(access)] > G.nodes[nodeAux]["hierarchy {}".format(access)]:
#                                 G.nodes[node]["features {}".format(access)]["Accumulated length from access"] = G.nodes[nodeAux]["features {}".format(access)]["Accumulated length from access"] + G[node][nodeAux]["features {}".format(access)]["Segment length"]
#     # If we are preparing the graph for labelling, pop hierarchy and make feature array for GNN processing
#     else:
#         for node in G:
#             G.nodes[node]["features"] = []
#             for key in G.nodes[node]["features femoral"]:
#                 G.nodes[node]["features"].append(G.nodes[node]["features femoral"][key])
#             G.nodes[node]["features"] = np.array(G.nodes[node]["features"])
#             G.nodes[node].pop("hierarchy femoral")
#             G.nodes[node].pop("features femoral")

#     return G

def extractNodeFeatures(G, segmentsCoordinateArray, segmentsRadiusArray, niftiCTA, branchModel, access = "femoral", edgesToRemove = None, featureExtractionForVesselLabelling = False):
    ''' Feature extractor method for graph featurization. Inputs a networkx graph as well as the segmentsArray
    and the CTA nifti from a case and returns the same graph with node attributes.

    For vessel labelling, hierarchical ordering may not be accurate, specially if parts of the vascular segments
    were missed in the segmentation process. This causes some features that rely on a correct hierarchical ordering
    not to be reliable. This is fixed for the final feature extraction, where separate segments and missed segmentations
    at the base of the VAs are joint to the rest of the graph to perform an accurate feature extraction.

    Features groups include (35-37 total features):
        * Node position
        * Radius
        * Curvature
        * Relative position to neighbor nodes
        * Segment distance
        * Directional features
        * Other features (CTA intensity, blanking)
        * Vessel type (if original graph is labelled)

    Inputs:
        - caseDir <str, path>: path to case directory.
        - G <nx.Graph>: non-featurized graph with hierarchical ordering for `access`.
        - access <str>: access site for thrombectomy configuration. Can be either "femoral" or "radial".
        - edgesToRemove <None or list>: if not None (then list), represents artificial edges of G that were added for a
        correct hierarchical ordering. These are removed for feature extraction (to respect node degree of original G),
        and are finally added for accumulative features computation. Output graph keeps these edges.
        - featureExtractionForVesselLabelling <bool>: if True, hierarchical ordering is not reliable and features are only used for vessel labelling.
        If False, then hierarchical ordering is reliable and features are extracted for the final characterization of G.

    Returns:
        - G <nx.Graph>: featurized G with node attributes.

    '''
    # Get CTA data from nifti
    arrayCTA = niftiCTA.get_fdata()
    # Translation
    aff = niftiCTA.affine
    translation = np.transpose(aff[:3, 3])

    # Pool branchModel point coordinates. Get blanking for each point
    branchModelCoordinates = np.ndarray([branchModel.GetNumberOfPoints(), 3])
    blanking = np.ndarray([branchModel.GetNumberOfPoints()])
    accumulatedNumberOfPoints = 0
    for idx in range(branchModel.GetNumberOfCells()):         
        for idx2 in range(branchModel.GetCell(idx).GetNumberOfPoints()):
            branchModelCoordinates[idx2 + accumulatedNumberOfPoints] = branchModel.GetCell(idx).GetPoints().GetPoint(idx2) - translation
            branchModelCoordinates[idx2 + accumulatedNumberOfPoints][0] = - branchModelCoordinates[idx2 + accumulatedNumberOfPoints][0]
            blanking[idx2 + accumulatedNumberOfPoints] = vtk_to_numpy(branchModel.GetCellData().GetArray("Blanking"))[idx]
        accumulatedNumberOfPoints += branchModel.GetCell(idx).GetNumberOfPoints()

    # If edges were artificially added for hierarchical ordering, remove them before feature extraction
    if edgesToRemove is not None:
        for src, dst in edgesToRemove:
            G.remove_edge(src, dst)

    for src, dst in G.edges:
        G[src][dst]["isArtificial"] = False

    # We need to iterate over all graph nodes and generalize the feature extraction process depending on the degree of the node
    for node in G:
        # We accumulate the distance to every neighbor node to compute the average distance to neighboring nodes
        distanceToNeighborNodes = []

        # Endpoints (degree == 1)
        # For endpoints, we gather information from the endpoint node's relationship to its neighbor (nodeEnd)
        if G.degree[node] == 1:
            # We get the position of the node
            nodePos = G.nodes[node]["pos"]
            # First we search the neighboring node and we get the candidate centerline points for segment feature extraction
            for nodeEnd in G.neighbors(node):
                # During the graph building phase, we stored the centerline points between each graph node
                # We only want to keep those closer to the node as part of the feature computations
                segmentPointsCandidates = segmentsCoordinateArray[G[node][nodeEnd]["cellId"]][G[node][nodeEnd]["segmentsArrayIndices"]]
                segmentRadiusCandidates = segmentsRadiusArray[G[node][nodeEnd]["cellId"]][G[node][nodeEnd]["segmentsArrayIndices"]]
                # We have to see the direction for the hierarchization in comparison to the centerline to see if we have to flip the position and radius arrays 
                # for a correct computation of the directional features
                if np.sign(node - nodeEnd) != np.sign(G.nodes[node]["hierarchy {}".format(access)] - G.nodes[nodeEnd]["hierarchy {}".format(access)]):
                    segmentPointsCandidates = np.flip(segmentPointsCandidates, axis = 0)
                    segmentRadiusCandidates = np.flip(segmentRadiusCandidates)
                # We get the position of the alternative node
                nodeEndPos = G.nodes[nodeEnd]["pos"]
                distanceToNeighborNodes.append(np.linalg.norm(nodePos - nodeEndPos))
            # We compute the distance along the centerline points
            distance = sum([np.linalg.norm(segmentPointsCandidates[idx - 1] - segmentPointsCandidates[idx]) for idx in range(1, len(segmentPointsCandidates))])
            # We compute the accumulated distance of the centerline over each candidate point and stop when we surpass half of the distance of the whole segment between both nodes
            for idx in range(1, len(segmentPointsCandidates) - 1):
                distanceEnd = sum([np.linalg.norm(segmentPointsCandidates[idxAux] - segmentPointsCandidates[idxAux + 1]) for idxAux in range(idx)])
                if distanceEnd > distance / 2:
                    break
            # Since we are analyzing an endpoint segment, we have to check if the analyzed node is an endpoint or a startpoint
            # We can use the node hierarchy computed for the dense graph for that
            # If the node hierarchy is greater than the alternative node, then we need to keep the distal part of the centerline segment for the feature extraction
            if G.nodes[node]["hierarchy {}".format(access)] > G.nodes[nodeEnd]["hierarchy {}".format(access)]: 
                segmentPoints = segmentPointsCandidates[idx:, :]
                segmentRadius = segmentRadiusCandidates[idx:]
                # From the candidate centerline points, we take a minimum of 3 points
                if len(segmentPoints) < 3:
                    segmentPoints = segmentPointsCandidates[-3:, :]
                    segmentRadius = segmentRadiusCandidates[-3:]
                # We also keep the proximal and distance vectors from node to node
                proxDirectionCartesian = nodeEndPos - nodePos
                distDirectionCartesian = nodePos - nodeEndPos
                # To compute the curvature, we take the position of the last three nodes (we compute curvature at a scale of node distances)
                for nodeEnd2 in G.neighbors(nodeEnd):
                    if nodeEnd2 != node:
                        break
                segmentPointsCurvature = np.array([
                    G.nodes[nodeEnd2]["pos"],
                    G.nodes[nodeEnd]["pos"],
                    G.nodes[node]["pos"]
                    
                ])
            # Otherwise (it is a startpoint), we keep the proximal part
            else:
                segmentPoints = segmentPointsCandidates[:idx - 1, :]
                segmentRadius = segmentRadiusCandidates[:idx - 1]
                # From the candidate centerline points, we take a minimum of 3 points
                if len(segmentPoints) < 3:
                    segmentPoints = segmentPointsCandidates[:3, :]
                    segmentRadius = segmentRadiusCandidates[:3]
                # We also keep the proximal and distance vectors from node to node
                proxDirectionCartesian = nodePos - nodeEndPos
                distDirectionCartesian = nodeEndPos - nodePos
                # To compute the curvature, we take the position of the first three nodes (we compute curvature at a scale of node distances)
                for nodeEnd2 in G.neighbors(nodeEnd):
                    if nodeEnd2 != node and G.nodes[nodeEnd2]["hierarchy {}".format(access)] < G.nodes[nodeEnd]["hierarchy {}".format(access)]:
                        break
                segmentPointsCurvature = np.array([
                    G.nodes[node]["pos"],
                    G.nodes[nodeEnd]["pos"],
                    G.nodes[nodeEnd2]["pos"]
                    
                ])

        # Normal segment (degree == 2)
        elif G.degree[node] == 2:
            # We get the node position    
            nodePos = G.nodes[node]["pos"]
            # This is the regular segment case. First we recognize the proximal and distal nodes depending on the node hierarchical index
            # Curvature is computed with preferably 5 point (if available) for nodes with degree == 2
            segmentPointsCurvature = np.array([G.nodes[node]["pos"]])
            # We define these empty arrays in case there is a rare event like a startpoint originating two centerlines (extremely rare)
            segmentPointsCandidatesProx = np.ndarray([0, 3])
            segmentRadiusCandidatesProx = np.array([])
            segmentPointsCandidatesDist = np.ndarray([0, 3])
            segmentRadiusCandidatesDist = np.array([])
            for nodeAux in G.neighbors(node):
                if G.nodes[nodeAux]["hierarchy {}".format(access)] < G.nodes[node]["hierarchy {}".format(access)]:
                    nodeProx = nodeAux
                    # During the graph building phase, we stored the centerline points between each graph node
                    # We only want to keep those closer to the node as part of the feature computations
                    segmentPointsCandidatesProx = segmentsCoordinateArray[G[node][nodeProx]["cellId"]][G[node][nodeProx]["segmentsArrayIndices"]]
                    segmentRadiusCandidatesProx = segmentsRadiusArray[G[node][nodeProx]["cellId"]][G[node][nodeProx]["segmentsArrayIndices"]]
                    # We have to see the direction for the hierarchization in comparison to the centerline to see if we have to flip the position and radius arrays 
                    # for a correct computation of the directional features
                    if np.sign(node - nodeProx) != np.sign(G.nodes[node]["hierarchy {}".format(access)] - G.nodes[nodeProx]["hierarchy {}".format(access)]):
                        segmentPointsCandidatesProx = np.flip(segmentPointsCandidatesProx, axis = 0)
                        segmentRadiusCandidatesProx = np.flip(segmentRadiusCandidatesProx)
                    # We get the position of the proximal node
                    nodeProxPos = G.nodes[nodeProx]["pos"]
                    distanceToNeighborNodes.append(np.linalg.norm(nodePos - nodeProxPos))
                    # To compute the curvature, we take the position of the two more proximal nodes (if available) (we compute curvature at a scale of node distances)
                    segmentPointsCurvature = np.insert(segmentPointsCurvature, 0, [nodeProxPos], axis = 0)
                    if G.degree(nodeProx) == 2:
                        for nodeProx2 in G.neighbors(nodeProx):
                            if nodeProx2 != node and G.nodes[nodeProx2]["hierarchy {}".format(access)] < G.nodes[nodeProx]["hierarchy {}".format(access)]:
                                segmentPointsCurvature = np.insert(segmentPointsCurvature, 0, [G.nodes[nodeProx2]["pos"]], axis = 0)
                elif G.nodes[nodeAux]["hierarchy {}".format(access)] > G.nodes[node]["hierarchy {}".format(access)]:
                    nodeDist = nodeAux
                    # During the graph building phase, we stored the centerline points between each graph node
                    # We only want to keep those closer to the node as part of the feature computations
                    segmentPointsCandidatesDist = segmentsCoordinateArray[G[node][nodeDist]["cellId"]][G[node][nodeDist]["segmentsArrayIndices"]]
                    segmentRadiusCandidatesDist = segmentsRadiusArray[G[node][nodeDist]["cellId"]][G[node][nodeDist]["segmentsArrayIndices"]]
                    # We have to see the direction for the hierarchization in comparison to the centerline to see if we have to flip the position and radius arrays 
                    # for a correct computation of the directional features
                    if np.sign(node - nodeDist) != np.sign(G.nodes[node]["hierarchy {}".format(access)] - G.nodes[nodeDist]["hierarchy {}".format(access)]):
                        segmentPointsCandidatesDist = np.flip(segmentPointsCandidatesDist, axis = 0)
                        segmentRadiusCandidatesDist = np.flip(segmentRadiusCandidatesDist)
                    # We get the position of the distal node
                    nodeDistPos = G.nodes[nodeDist]["pos"]
                    distanceToNeighborNodes.append(np.linalg.norm(nodePos - nodeDistPos))
                    # To compute the curvature, we take the position of the two more distal nodes (if available) (we compute curvature at a scale of node distances)
                    segmentPointsCurvature = np.append(segmentPointsCurvature, [nodeDistPos], axis = 0)
                    if G.degree(nodeDist) == 2:
                        for nodeDist2 in G.neighbors(nodeDist):
                            if nodeDist2 != node and G.nodes[nodeDist2]["hierarchy {}".format(access)] > G.nodes[nodeDist]["hierarchy {}".format(access)]:
                                segmentPointsCurvature = np.append(segmentPointsCurvature, [G.nodes[nodeDist2]["pos"]], axis = 0)

            # We define these empty arrays in case there is a rare event like a startpoint originating two centerlines (extremely rare)
            if len(segmentPointsCandidatesProx) < 1:
                distanceProx = 0
                segmentPointsProx = np.ndarray([0, 3])
                segmentRadiusProx = np.array([])
                nodeProxPos = nodePos
            else:
                # We compute the distance along the centerline between the proximal node and the node, to get the centerline points within the node segment
                distanceProx = sum([np.linalg.norm(segmentPointsCandidatesProx[idx - 1] - segmentPointsCandidatesProx[idx]) for idx in range(1, len(segmentPointsCandidatesProx))])
                # We keep the distal part of the centerline points (closer to node)
                for idx in range(1, len(segmentPointsCandidatesProx) - 1):
                    distanceAccProx = sum([np.linalg.norm(segmentPointsCandidatesProx[idxAux] - segmentPointsCandidatesProx[idxAux + 1]) for idxAux in range(idx)])
                    if distanceAccProx > distanceProx / 2:
                        break
                segmentPointsProx = segmentPointsCandidatesProx[idx:, :]
                segmentRadiusProx = segmentRadiusCandidatesProx[idx:]
            # From the candidate centerline points, we take a minimum of 3 points
            if len(segmentPointsProx) < 3:
                segmentPointsProx = segmentPointsCandidatesProx[-3:, :]
                segmentRadiusProx = segmentRadiusCandidatesProx[-3:]
            
            # We define these empty arrays in case there is a rare event like a startpoint originating two centerlines (extremely rare)
            if len(segmentPointsCandidatesDist) < 1:
                distanceDist = 0
                segmentPointsDist = np.ndarray([0, 3])
                segmentRadiusDist = np.array([])
                nodeDistPos = nodePos
            else:
                # We compute the distance along the centerline between the node and the distal node, to get the centerline points within the node segment
                distanceDist = sum([np.linalg.norm(segmentPointsCandidatesDist[idx - 1] - segmentPointsCandidatesDist[idx]) for idx in range(1, len(segmentPointsCandidatesDist))])
                # We keep the proximal part of the centerline points (closer to node)
                for idx in range(1, len(segmentPointsCandidatesDist) - 1):
                    distanceAccDist = sum([np.linalg.norm(segmentPointsCandidatesDist[idxAux] - segmentPointsCandidatesDist[idxAux + 1]) for idxAux in range(idx)])
                    if distanceAccDist > distanceDist / 2:
                        break
                segmentPointsDist = segmentPointsCandidatesDist[:idx - 1, :]
                segmentRadiusDist = segmentRadiusCandidatesDist[:idx - 1]
                # From the candidate centerline points, we take a minimum of 3 points
                if len(segmentPointsDist) < 3:
                    segmentPointsDist = segmentPointsCandidatesDist[:3, :]
                    segmentRadiusDist = segmentRadiusCandidatesDist[:3]

            # Finally we concatenate both the coordinates and radius of both ends of the segment
            segmentPoints = np.append(segmentPointsProx, segmentPointsDist, axis = 0)
            segmentRadius = np.append(segmentRadiusProx, segmentRadiusDist)
            # We also keep the proximal and distance vectors from node to node
            proxDirectionCartesian = nodeProxPos - nodePos
            distDirectionCartesian = nodeDistPos - nodePos

        # Multi-furcations (degree > 2)
        elif G.degree[node] > 2:
            # We get the node position    
            nodePos = G.nodes[node]["pos"]
            # We treat multi-furcation as endpoints (since there is no preference a priori for the path that needs to be taken). We only look at the preceeding node
            # First we search the neighboring nodes and we get the candidate centerline points for segment feature extraction from the preceeding node
            # To find the preceeding node, we check the hierarchical order (there should be one node with a lower hierarchical index than the bifurcation point)
            # We have to initialize the segment arrays just in case we are in the rare event of a multifurcation with hierarchy = 0
            segmentPointsCandidates = []
            for nodeAux in G.neighbors(node):
                if G.nodes[nodeAux]["hierarchy {}".format(access)] < G.nodes[node]["hierarchy {}".format(access)]:
                    nodeProx = nodeAux
                    # During the graph building phase, we stored the centerline points between each graph node
                    # We only want to keep those closer to the node as part of the feature computations
                    segmentPointsCandidates = segmentsCoordinateArray[G[node][nodeProx]["cellId"]][G[node][nodeProx]["segmentsArrayIndices"]]
                    segmentRadiusCandidates = segmentsRadiusArray[G[node][nodeProx]["cellId"]][G[node][nodeProx]["segmentsArrayIndices"]]
                    # We have to see the direction for the hierarchization in comparison to the centerline to see if we have to flip the position and radius arrays 
                    # for a correct computation of the directional features
                    if np.sign(node - nodeProx) != np.sign(G.nodes[node]["hierarchy {}".format(access)] - G.nodes[nodeProx]["hierarchy {}".format(access)]):
                        segmentPointsCandidates = np.flip(segmentPointsCandidates, axis = 0)
                        segmentRadiusCandidates = np.flip(segmentRadiusCandidates)
                    # We get the position of the proximal node
                    nodeProxPos = G.nodes[nodeProx]["pos"]
                distanceToNeighborNodes.append(np.linalg.norm(nodePos - G.nodes[nodeAux]["pos"]))
            # Multi-furcation as start node. We just take the first neighbor to compute all variables. In this case, we invert the roles of node and nodeProx
            if len(segmentPointsCandidates) == 0:
                originalNode = node
                for nodeAux in G.neighbors(node):
                    nodeProx = originalNode
                    node = nodeAux
                    # During the graph building phase, we stored the centerline points between each graph node
                    # We only want to keep those closer to the node as part of the feature computations
                    segmentPointsCandidates = segmentsCoordinateArray[G[node][nodeProx]["cellId"]][G[node][nodeProx]["segmentsArrayIndices"]]
                    segmentRadiusCandidates = segmentsRadiusArray[G[node][nodeProx]["cellId"]][G[node][nodeProx]["segmentsArrayIndices"]]
                    # We have to see the direction for the hierarchization in comparison to the centerline to see if we have to flip the position and radius arrays 
                    # for a correct computation of the directional features
                    if np.sign(node - nodeProx) != np.sign(G.nodes[node]["hierarchy {}".format(access)] - G.nodes[nodeProx]["hierarchy {}".format(access)]):
                        segmentPointsCandidates = np.flip(segmentPointsCandidates, axis = 0)
                        segmentRadiusCandidates = np.flip(segmentRadiusCandidates)
                    # We get the position of the proximal node
                    nodeProxPos = G.nodes[nodeProx]["pos"]
                    # In this case, we also update the node position
                    nodePos = G.nodes[node]["pos"]
                    # We just take a look at any random node
                    break

                # We compute the distance along the centerline points
                distance = sum([np.linalg.norm(segmentPointsCandidates[idx - 1] - segmentPointsCandidates[idx]) for idx in range(1, len(segmentPointsCandidates))])
                # We compute the accumulated distance of the centerline over each candidate point and stop when we surpass half of the distance of the whole segment between both nodes
                for idx in range(1, len(segmentPointsCandidates) - 1):
                    distanceProx = sum([np.linalg.norm(segmentPointsCandidates[idxAux] - segmentPointsCandidates[idxAux + 1]) for idxAux in range(idx)])
                    if distanceProx > distance / 2:
                        break
                # We can use the node hierarchy computed for the dense graph for that
                # If the node hierarchy is greater than the alternative node, then we need to keep the distal part of the centerline segment for the feature extraction
                segmentPoints = segmentPointsCandidates[idx:]
                segmentRadius = segmentRadiusCandidates[idx:]
                # From the candidate centerline points, we take a minimum of 3 points
                if len(segmentPoints) < 3:
                    segmentPoints = segmentPointsCandidates[-3:, :]
                    segmentRadius = segmentRadiusCandidates[-3:]
                # We also keep the proximal and distance vectors from node to node
                proxDirectionCartesian = nodeProxPos - nodePos
                distDirectionCartesian = nodePos - nodeProxPos
                # To compute the curvature, we take the position of the first three nodes (we compute curvature at a scale of node distances)
                for nodeAux2 in G.neighbors(nodeAux):
                    if nodeAux2 != node:
                        break
                segmentPointsCurvature = np.array([
                    G.nodes[node]["pos"],
                    G.nodes[nodeAux]["pos"],
                    G.nodes[nodeAux2]["pos"]
                    
                ])

                # At the end, we set node and nodePos back to their original value
                node = originalNode
                nodePos = G.nodes[node]["pos"]

            else:
                # We compute the distance along the centerline points
                distance = sum([np.linalg.norm(segmentPointsCandidates[idx - 1] - segmentPointsCandidates[idx]) for idx in range(1, len(segmentPointsCandidates))])
                # We compute the accumulated distance of the centerline over each candidate point and stop when we surpass half of the distance of the whole segment between both nodes
                for idx in range(1, len(segmentPointsCandidates) - 1):
                    distanceProx = sum([np.linalg.norm(segmentPointsCandidates[idxAux] - segmentPointsCandidates[idxAux + 1]) for idxAux in range(idx)])
                    if distanceProx > distance / 2:
                        break
                # We can use the node hierarchy computed for the dense graph for that
                # If the node hierarchy is greater than the alternative node, then we need to keep the distal part of the centerline segment for the feature extraction
                segmentPoints = segmentPointsCandidates[idx:]
                segmentRadius = segmentRadiusCandidates[idx:]
                # From the candidate centerline points, we take a minimum of 3 points
                if len(segmentPoints) < 3:
                    segmentPoints = segmentPointsCandidates[-3:, :]
                    segmentRadius = segmentRadiusCandidates[-3:]
                # We also keep the proximal and distance vectors from node to node
                proxDirectionCartesian = nodeProxPos - nodePos
                distDirectionCartesian = nodePos - nodeProxPos
                # To compute the curvature, we take the position of the first three nodes (we compute curvature at a scale of node distances)
                for nodeAux2 in G.neighbors(nodeAux):
                    if nodeAux2 != node and G.nodes[nodeAux2]["hierarchy {}".format(access)] < G.nodes[nodeAux]["hierarchy {}".format(access)]:
                        break
                segmentPointsCurvature = np.array([
                    G.nodes[node]["pos"],
                    G.nodes[nodeAux]["pos"],
                    G.nodes[nodeAux2]["pos"]
                    
                ])
            
        # Eliminate repeated points
        segmentPoints, indices = np.unique(segmentPoints, axis = 0, return_index = True)
        segmentRadius = segmentRadius[indices]

        # Feature extraction           
        G.nodes[node]["features {}".format(access)] = {}

        # Hierarchy
        G.nodes[node]["features {}".format(access)]["Hierarchy"] = G.nodes[node]["hierarchy {}".format(access)]
        
        # Node position
        # We have to compute the ijk positions for normalization purposes (either this or subtract the translation from the affine matrix)
        # nodePosIJK = np.matmul(np.linalg.inv(aff), np.append(nodePos, 1.0))[:3]
        G.nodes[node]["features {}".format(access)]["Node position i"] = nodePos[0]
        G.nodes[node]["features {}".format(access)]["Node position j"] = nodePos[1]
        G.nodes[node]["features {}".format(access)]["Node position k"] = nodePos[2]
        
        # Radius features
        G.nodes[node]["features {}".format(access)]["Mean radius"] = np.mean(segmentRadius)
        G.nodes[node]["features {}".format(access)]["Proximal radius"] = segmentRadius[0]
        G.nodes[node]["features {}".format(access)]["Middle radius"] = segmentRadius[len(segmentRadius) // 2]
        G.nodes[node]["features {}".format(access)]["Distal radius"] = segmentRadius[-1]
        G.nodes[node]["features {}".format(access)]["Maximum radius"] = np.amax(segmentRadius)
        G.nodes[node]["features {}".format(access)]["Minimum radius"] = np.amin(segmentRadius)
        G.nodes[node]["features {}".format(access)]["Min/Max radius ratio"] = np.amin(segmentRadius) / np.amax(segmentRadius)
        
        # Curvature
        radiusOfCurvatureArray = computeCurvature(segmentPointsCurvature)
        if len(radiusOfCurvatureArray) == 3:
            G.nodes[node]["features {}".format(access)]["Curvature at node"] = 1 / radiusOfCurvatureArray[1]
            G.nodes[node]["features {}".format(access)]["Maximum curvature in segment"] = 1 / radiusOfCurvatureArray[1]
            # G.nodes[node]["features {}".format(access)]["Radius of curvature at node"] = radiusOfCurvatureArray[1]
            # G.nodes[node]["features {}".format(access)]["Minimum radius of curvature in segment"] = radiusOfCurvatureArray[1]
        else:
            G.nodes[node]["features {}".format(access)]["Curvature at node"] = 1 / radiusOfCurvatureArray[np.argmin(np.linalg.norm(segmentPointsCurvature - nodePos, axis = 1))]
            G.nodes[node]["features {}".format(access)]["Maximum curvature in segment"] = 1 / np.amin(radiusOfCurvatureArray)
            # G.nodes[node]["features {}".format(access)]["Radius of curvature at node"] = radiusOfCurvatureArray[np.argmin(np.linalg.norm(segmentPointsCurvature - nodePos, axis = 1))]
            # G.nodes[node]["features {}".format(access)]["Minimum radius of curvature in segment"] = np.amin(radiusOfCurvatureArray)
        
        # Neighbor nodes features
        G.nodes[node]["features {}".format(access)]["Degree"] = G.degree[node]
        G.nodes[node]["features {}".format(access)]["Average distance to neighbor nodes"] = np.mean(distanceToNeighborNodes)
        
        # Segment distance features
        G.nodes[node]["features {}".format(access)]["Segment length"] = sum([np.linalg.norm(segmentPoints[idx - 1] - segmentPoints[idx]) for idx in range(1, len(segmentPoints))])
        G.nodes[node]["features {}".format(access)]["Number of points"] = len(segmentPoints)
        G.nodes[node]["features {}".format(access)]["Average distance between points"] = G.nodes[node]["features {}".format(access)]["Segment length"] / len(segmentPoints)
        G.nodes[node]["features {}".format(access)]["Relative length"] = np.linalg.norm(segmentPoints[-1] - segmentPoints[0]) / G.nodes[node]["features {}".format(access)]["Segment length"]

        # Directional features
        segmentDirection = segmentPoints[-1] - segmentPoints[0]
        module, polar, azimuth = sphericalAnglesFrom3DCartesian(segmentDirection)
        moduleProx, polarProx, azimuthProx = sphericalAnglesFrom3DCartesian(proxDirectionCartesian)
        moduleDist, polarDist, azimuthDist = sphericalAnglesFrom3DCartesian(distDirectionCartesian)
        G.nodes[node]["features {}".format(access)]["Direction module"] = module
        G.nodes[node]["features {}".format(access)]["Direction polar"] = polar
        G.nodes[node]["features {}".format(access)]["Direction azimuth"] = azimuth
        G.nodes[node]["features {}".format(access)]["Proximal direction module"] = moduleProx
        G.nodes[node]["features {}".format(access)]["Proximal direction polar"] = polarProx
        G.nodes[node]["features {}".format(access)]["Proximal direction azimuth"] = azimuthProx
        G.nodes[node]["features {}".format(access)]["Distal direction module"] = moduleDist
        G.nodes[node]["features {}".format(access)]["Distal direction polar"] = polarDist
        G.nodes[node]["features {}".format(access)]["Distal direction azimuth"] = azimuthDist
        
        # Other features
        G.nodes[node]["features {}".format(access)]["Blanking"] = blanking[findPointId(nodePos, branchModelCoordinates)]
        G.nodes[node]["features {}".format(access)]["HU at node position"] = arrayCTA[np.round(nodePos).astype(int)[0], np.round(nodePos).astype(int)[1], np.round(nodePos).astype(int)[2]]
        segmentHU = []
        for idx, pointPosition in enumerate(segmentPoints):
            segmentHU.append(arrayCTA[np.round(pointPosition).astype(int)[0], np.round(pointPosition).astype(int)[1], np.round(pointPosition).astype(int)[2]])
        G.nodes[node]["features {}".format(access)]["Min HU in segment"] = min(segmentHU)
        G.nodes[node]["features {}".format(access)]["Max HU in segment"] = max(segmentHU)

    # If we had removed edges in the beggining, we add them again to compute accumulated features
    if edgesToRemove is not None:
        for src, dst in edgesToRemove:
            G.add_edge(src, dst, cellId = G.nodes[dst]["cellId"])
            G[src][dst]["isArtificial"] = True
            # Empty segmentsArrayIndices to identify artificial edges
            G[src][dst]["segmentsArrayIndices"] = np.array([])

    # We finally check all nodes not to have any nan or inf values
    # If present, we choose the value from the neighboring nodes
    # We first check fist node (there is only one node with hierarchy equal to 0). This way, we ensure no error propagation and
    # that all nodes will not have nan or inf feature values:
    for node in G:
        if G.nodes[node]["hierarchy {}".format(access)] == 0:
            break
    for featureKey in G.nodes[node]["features {}".format(access)].keys():
        # We create an auxiliary node variable in case we have to look further than the first-degree neighborhood 
        currentNode = node
        while math.isnan(G.nodes[node]["features {}".format(access)][featureKey]) or math.isinf(G.nodes[node]["features {}".format(access)][featureKey]):
            for neighbor in G.neighbors(currentNode):
                # Now we choose first following node to also include first node
                if G.nodes[currentNode]["hierarchy {}".format(access)] < G.nodes[neighbor]["hierarchy {}".format(access)] and not math.isnan(G.nodes[neighbor]["features {}".format(access)][featureKey]) and not math.isinf(G.nodes[neighbor]["features {}".format(access)][featureKey]):
                    print(featureKey, node, neighbor, G.nodes[neighbor]["features {}".format(access)][featureKey])
                    G.nodes[node]["features {}".format(access)][featureKey] = G.nodes[neighbor]["features {}".format(access)][featureKey]
                    break
            # We update currentnode in case we do not find valid values for the nan or inf features. Search will continue from node to node until we find closes node with valid values
            # This is very unlikely to continue further than one node doe to the low frequency of nan or inf values, but we are inclusive just in case
            currentNode = neighbor
    
    # Now we check all other nodes (differently from looking at the first node, we look at nodes with lower hierarchy)
    for node in G:
        if G.nodes[node]["hierarchy {}".format(access)] > 0:
            for featureKey in G.nodes[node]["features {}".format(access)].keys():
                # We create an auxiliary node variable in case we have to look further than the first-degree neighborhood 
                currentNode = node
                while math.isnan(G.nodes[node]["features {}".format(access)][featureKey]) or math.isinf(G.nodes[node]["features {}".format(access)][featureKey]):
                    for neighbor in G.neighbors(currentNode):
                        # Now we choose first following node to also include first node
                        if G.nodes[currentNode]["hierarchy {}".format(access)] > G.nodes[neighbor]["hierarchy {}".format(access)] and not math.isnan(G.nodes[neighbor]["features {}".format(access)][featureKey]) and not math.isinf(G.nodes[neighbor]["features {}".format(access)][featureKey]):
                            print(featureKey, node, neighbor, G.nodes[neighbor]["features {}".format(access)][featureKey])
                            G.nodes[node]["features {}".format(access)][featureKey] = G.nodes[neighbor]["features {}".format(access)][featureKey]
                            break
                    # We update currentnode in case we do not find valid values for the nan or inf features. Search will continue from node to node until we find closes node with valid values
                    # This is very unlikely to continue further than one node doe to the low frequency of nan or inf values, but we are inclusive just in case
                    currentNode = neighbor

    # If we are now performing final feature extraction for supersegment treatment, we compute accumulative features that rely on a proper hierarchical ordering (need subgraph union)
    if not featureExtractionForVesselLabelling:
        # Accumulative features
        for hierarchy in range(getMaxHierarchy(G, access) + 1):
            for node in G:
                if G.nodes[node]["hierarchy {}".format(access)] == hierarchy:
                    # For the first node, we just compute the segment length
                    if hierarchy == 0:
                        G.nodes[node]["features {}".format(access)]["Accumulated length from access"] = G.nodes[node]["features {}".format(access)]["Segment length"]
                    else:
                        for nodeAux in G.neighbors(node):
                            # For successive nodes, we addthe segment lenght to the previously accumulated length from access
                            if G.nodes[node]["hierarchy {}".format(access)] > G.nodes[nodeAux]["hierarchy {}".format(access)]:
                                # For artificial nodes, we add the distance between nodes instead (marked by empty segmentsArrayIndices vector in edge features)
                                if len(G[node][nodeAux]["segmentsArrayIndices"]) > 0:
                                    G.nodes[node]["features {}".format(access)]["Accumulated length from access"] = G.nodes[node]["features {}".format(access)]["Segment length"] + G.nodes[nodeAux]["features {}".format(access)]["Accumulated length from access"]
                                else:
                                    G.nodes[node]["features {}".format(access)]["Accumulated length from access"] = G.nodes[nodeAux]["features {}".format(access)]["Accumulated length from access"] + np.linalg.norm(G.nodes[node]["pos"] - G.nodes[nodeAux]["pos"])
        # When available, we also add the vessel label as node feature
        for node in G:
            G.nodes[node]["features {}".format(access)]["Vessel type"] = G.nodes[node]["Vessel type"]
    # If we are preparing the graph for labelling, pop hierarchy and make feature array for GNN processing
    else:
        for node in G:
            G.nodes[node].pop("hierarchy femoral")
            G.nodes[node]["features"] = []
            for key in G.nodes[node]["features femoral"]:
                G.nodes[node]["features"].append(G.nodes[node]["features femoral"][key])
            G.nodes[node]["features"] = np.array(G.nodes[node]["features"])
            G.nodes[node].pop("features femoral")

    return G

def extractGlobalFeatures(G):

    def getAANodeGoordinates(G):
        AApositions = np.ndarray([0, 3], dtype = float)
        for node in G:
            if G.nodes[node]["Vessel type name"] == "AA":
                AApositions = np.append(AApositions, [G.nodes[node]["pos"]], axis = 0)

        return AApositions
    
    def getAAType(G):
        AApositions = getAANodeGoordinates(G)
        closestBTNodeDistance = 100
        closestLCCANodeDistance = 100
        A = 0
        B, D = None, None
        for node in G:
            if G.nodes[node]["Vessel type name"] == "AA" and G.nodes[node]["pos"][2] + G.nodes[node]["radius"] > A:
                A = G.nodes[node]["pos"][2] + G.nodes[node]["radius"]
            if G.nodes[node]["Vessel type name"] == "BT" and G.nodes[node]["features femoral"]["Blanking"] == 0 and np.amin(np.linalg.norm(AApositions - G.nodes[node]["pos"], axis = 1)) < closestBTNodeDistance:
                B = G.nodes[node]["pos"][2]
                closestBTNodeDistance = np.amin(np.linalg.norm(AApositions - G.nodes[node]["pos"], axis = 1))
            if G.nodes[node]["Vessel type name"] == "LCCA" and G.nodes[node]["features femoral"]["Blanking"] == 0 and np.amin(np.linalg.norm(AApositions - G.nodes[node]["pos"], axis = 1)) < closestLCCANodeDistance:
                D = 2 * G.nodes[node]["radius"]
                closestLCCANodeDistance = np.amin(np.linalg.norm(AApositions - G.nodes[node]["pos"], axis = 1))

        if A > 0 and B is not None and D is not None:
            value = np.abs(A - B) / D
            if value < 1:
                AAType = 1
            elif value > 1 and value < 2:
                AAType = 2
            elif value > 2:
                AAType = 3
        else:
            AAType = 1

        return AAType

    def getBovineArch(G):
        AApositions = getAANodeGoordinates(G)
        closestLCCANodeDistance = 100
        closestLCCANode = None
        for node in G:
            if G.nodes[node]["Vessel type name"] == "LCCA" and np.amin(np.linalg.norm(AApositions - G.nodes[node]["pos"], axis = 1)) < closestLCCANodeDistance:
                closestLCCANode = node
                closestLCCANodeDistance = np.amin(np.linalg.norm(AApositions - G.nodes[node]["pos"], axis = 1))

        if closestLCCANode is not None:
            vesselTypeNamesInContact = []
            for neighbor in G.neighbors(closestLCCANode):
                for src, dst in G.edges(neighbor):
                    if not G[src][dst]["isArtificial"]:
                        vesselTypeNamesInContact.append(G[src][dst]["Vessel type name"])

            if "BT" in vesselTypeNamesInContact:
                bovineArch = 1
            else:
                bovineArch = 0
        else:
            bovineArch = 0

        return bovineArch

    def getARSA(G):
        # Add extra criteria to make it more reliable. 
        # RSA linked to an AA point closer to feamoral startpoint than LSA?
            # Get AA closest node: point A
            # Get point A hierarchy
            # Repeat computation with LSA: point B
            # Get pointy B hierarchy
            # If hierarchy B > hierarchy A, and AA in vesselTypeNamesInContact, ARSA. Else, not ARSA

        AApositions = getAANodeGoordinates(G)
        closestRSANodeDistance = 100
        closestLSANodeDistance = 100
        closestRSANode = None
        closestLSANode = None

        for node in G:
            if G.nodes[node]["Vessel type name"] == "RSA" and np.amin(np.linalg.norm(AApositions - G.nodes[node]["pos"], axis = 1)) < closestRSANodeDistance:
                closestRSANode = node
                closestRSANodeDistance = np.amin(np.linalg.norm(AApositions - G.nodes[node]["pos"], axis = 1))
            if G.nodes[node]["Vessel type name"] == "LSA" and np.amin(np.linalg.norm(AApositions - G.nodes[node]["pos"], axis = 1)) < closestLSANodeDistance:
                closestLSANode = node
                closestLSANodeDistance = np.amin(np.linalg.norm(AApositions - G.nodes[node]["pos"], axis = 1))

        if closestRSANode is not None:
            vesselTypeNamesInContact = []
            for neighbor in G.neighbors(closestRSANode):
                for src, dst in G.edges(neighbor):
                    if not G[src][dst]["isArtificial"]:
                        vesselTypeNamesInContact.append(G[src][dst]["Vessel type name"])

            if "AA" in vesselTypeNamesInContact and G.nodes[closestRSANode]["hierarchy femoral"] < G.nodes[closestLSANode]["hierarchy femoral"]:
                ARSA = 1
            else:
                ARSA = 0
        else:
            ARSA = 0

        return ARSA

    G.graph["AA type"] = getAAType(G)
    G.graph["Bovine arch"] = getBovineArch(G)
    G.graph["ARSA"] = getARSA(G)

    return G
    
    # def getARSA(G):
    #     # Add extra criteria to make it more reliable. 
    #     # RSA linked to an AA point closer to feamoral startpoint than LSA?
    #         # Get AA closest node: point A
    #         # Get point A hierarchy
    #         # Repeat computation with LSA: point B
    #         # Get pointy B hierarchy
    #         # If hierarchy B > hierarchy A, and AA in vesselTypeNamesInContact, ARSA. Else, not ARSA

    #     AApositions = getAANodeGoordinates(G)
    #     closestRSANodeDistance = 100
    #     closestRSANode = None

    #     for node in G:
    #         if G.nodes[node]["Vessel type name"] == "RSA" and np.amin(np.linalg.norm(AApositions - G.nodes[node]["pos"], axis = 1)) < closestRSANodeDistance:
    #             closestRSANode = node
    #             closestRSANodeDistance = np.amin(np.linalg.norm(AApositions - G.nodes[node]["pos"], axis = 1))

    #     if closestRSANode is not None:
    #         vesselTypeNamesInContact = []
    #         for neighbor in G.neighbors(closestRSANode):
    #             for src, dst in G.edges(neighbor):
    #                 if not G[src][dst]["isArtificial"]:
    #                     vesselTypeNamesInContact.append(G[src][dst]["Vessel type name"])

    #         if "AA" in vesselTypeNamesInContact:
    #             ARSA = 1
    #         else:
    #             ARSA = 0
    #     else:
    #         ARSA = 0

    #     return ARSA

    # G.graph["AA type"] = getAAType(G)
    # G.graph["Bovine arch"] = getBovineArch(G)
    # G.graph["ARSA"] = getARSA(G)

    # return G

def extractFeatures(G, segmentsCoordinateArray, segmentsRadiusArray, niftiCTA, branchModel, access = "femoral", edgesToRemove = None, cellIdToVesselType = None, featureExtractionForVesselLabelling = False):
    ''' Feature extractor method for graph featurization. Inputs a networkx graph as well as the segmentsArray
    and the CTA nifti from a case and returns the same graph with node attributes.

    For vessel labelling, hierarchical ordering may not be accurate, specially if parts of the vascular segments
    were missed in the segmentation process. This causes some features that rely on a correct hierarchical ordering
    not to be reliable. This is fixed for the final feature extraction, where separate segments and missed segmentations
    at the base of the VAs are joint to the rest of the graph to perform an accurate feature extraction.

    Features groups include (35-37 total features):
        * Node position
        * Radius
        * Curvature
        * Relative position to neighbor nodes
        * Segment distance
        * Directional features
        * Other features (CTA intensity, blanking)
        * Vessel type (if original graph is labelled)

    Inputs:
        - caseDir <str, path>: path to case directory.
        - G <nx.Graph>: non-featurized graph with hierarchical ordering for `access`.
        - access <str>: access site for thrombectomy configuration. Can be either "femoral" or "radial".
        - edgesToRemove <None or list>: if not None (then list), represents artificial edges of G that were added for a
        correct hierarchical ordering. These are removed for feature extraction (to respect node degree of original G),
        and are finally added for accumulative features computation. Output graph keeps these edges.
        - featureExtractionForVesselLabelling <bool>: if True, hierarchical ordering is not reliable and features are only used for vessel labelling.
        If False, then hierarchical ordering is reliable and features are extracted for the final characterization of G.

    Returns:
        - G <nx.Graph>: featurized G with node attributes.

    '''

    # G = extractEdgeFeatures(G, niftiCTA = niftiCTA, access = access, edgesToRemove = edgesToRemove, cellIdToVesselType = cellIdToVesselType, featureExtractionForVesselLabelling = featureExtractionForVesselLabelling)
    G = extractNodeFeatures(G, segmentsCoordinateArray = segmentsCoordinateArray, segmentsRadiusArray = segmentsRadiusArray, niftiCTA = niftiCTA, branchModel = branchModel, access = access, edgesToRemove = edgesToRemove, featureExtractionForVesselLabelling = featureExtractionForVesselLabelling)
    if not featureExtractionForVesselLabelling:
        G = extractGlobalFeatures(G)

    return G

def getMaxHierarchy(G, access = "femoral"):
    ''' Iterates over nodes to find max hierarchy for each access configuration.

    Arguments: 
        - G <nx.Graph>: graph.
        - access <str>: access site for thrombectomy configuration. Can be either "femoral" or "radial".

    Returns:
        - maxHierarchy <int>: maximum hierarchical index for `access`.
    
    '''
    maxHierarchy = 0
    for node in G:
        if G.nodes[node]["hierarchy {}".format(access)] > maxHierarchy:
            maxHierarchy = G.nodes[node]["hierarchy {}".format(access)]
                   
    return maxHierarchy

def sphericalAnglesFrom3DCartesian(vec):
    ''' Returns spherical angles of a vector in cartesian coordinates.

    Arguments:
        - vec: numpy array of shape [3] or equivalent.

    Returns:
        - polar: polar angle in spherical coordinates. Contained between -pi / 2 and pi / 2.
        - azimuth: azimuth angle in spherical coordinates. Contained between -pi and pi.

    '''
    x, y, z = vec
    # We compute the module of the vector
    module = np.linalg.norm(vec)

    # For the polar angle, we consider the case when z could be 0
    if abs(z) < 1e-5:
        # If any x or y is different than 0, polar is pi / 2
        if abs(x) > 1e-5 or abs(y) > 1e-5:
            polar = 0
        # Otherwise, we are in the case when vec == [0, 0, 0]
        else:
            polar = math.nan
    # Otherwise compute polar angle normally
    else: 
        polar = np.sign(z) * math.pi / 2 - math.atan((x ** 2 + y ** 2) ** 0.5 / z) 

    if polar > math.pi / 2:
        polar -= math.pi
    
    # For the azimuth angle, let's check the case when x could be 0
    if abs(x) < 1e-5:
        # If y is not 0, azimuth is either pi / 2 (y > 0) or 3 * pi / 2 (y < 0)
        if abs(y) > 1e-5:
            azimuth = (2 - np.sign(y)) * (math.pi / 2)
        # If y is 0, azimuth is undetermined
        else:
            azimuth = math.nan
    # Otherwise compute azimuth angle normally
    else:
        azimuth = math.atan(y / x)
        # Depending on the quadrant, we have to add additional rotation
        if x > 0 and y >= 0: # First quadrant (x / y > 0, atan in [0, pi / 2], angle should be between 0 and pi / 2). We add 0
                             # Also included the possibility that y = 0 and x > 0. Then, azimuth should be 0
            azimuth += 0
        elif x < 0 and y >= 0: # Second quadrant (x / y < 0, atan in [-pi / 2, 0], angle should be between pi / 2 and pi). We add pi
                               # Also included the possibility that y = 0 and x < 0. Then, azimuth should be pi
            azimuth += math.pi
        elif x < 0 and y < 0: # Third quadrant (x / y > 0, atan in [0, pi / 2], angle should be between -pi and -pi / 2). We add -pi
            azimuth -= math.pi
        elif x > 0 and y < 0: # Fourth quadrant (x / y < 0, atan in [-pi / 2, 0], angle should be between -pi / 2 and 0). We add 0
            azimuth += 0

    return module, polar, azimuth

def computeCurvature(curve):
    # Compute tangent vector
    velocity = np.gradient(curve, axis = 0)
    # Compute speed (module of tangent vector)
    speed = np.linalg.norm(velocity, axis = 1)
    # Compute unitary tangent vector at each point
    unitTangent = np.einsum('ij,i->ij', velocity[:, :], 1 / speed)
    # Compute curvature
    curvature = np.linalg.norm(np.gradient(unitTangent, axis = 0), axis = 1) / speed
    # Compute radius of curvature
    radiusOfCurvature = 1 / curvature
    
    return radiusOfCurvature

def findPointId(point, modelCoordinates):
    return np.argmin(np.linalg.norm(modelCoordinates - point, axis = 1))

def vesselTypeSequenceToOneHot(sequence, highlights = None, counterHighlights = None):
    vesselTypeOneHotCode = {
        "other": 0,
        "AA": 1,
        "BT": 2,
        "RCCA": 3,
        "LCCA": 4,
        "RSA": 5,
        "LSA": 6,
        "RVA": 7,
        "LVA": 8,
        "RICA": 9,
        "LICA": 10,
        "RECA": 11,
        "LECA": 12,
        "BA": 13
    }
    
    supersegmentCandidateOneHot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for vesselType in sequence:
        supersegmentCandidateOneHot[vesselTypeOneHotCode[vesselType]] = 1

    if highlights is not None:
        for highlight in highlights:
            supersegmentCandidateOneHot[vesselTypeOneHotCode[highlight]] += 1

    if counterHighlights is not None:
        for counterHighlight in counterHighlights:
            supersegmentCandidateOneHot[vesselTypeOneHotCode[counterHighlight]] -= 1
        
    return supersegmentCandidateOneHot

def cosineSimilarity(u, v):
    ''' Cosine of the angle difference between two vectors. 
    We use it as a similarity measure between supersegment candidates
    and reference thrombectomy configurations. We use the labelled vessel sequence 
    and pass it to one-hot encoding to perform this measurement.
    
    '''
    # Vectors have to be the same length (one-hot encoding ensures this)
    assert len(u) == len(v)
    # The cosine distance or similarity between two vectors is simply the dot product of both vectors
    # divided by the product of their modules
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def makeGraphPlot(caseDir, G, filename, label = None):
    # Generate plot of dense graph for quick visualization
    _ = plt.figure(figsize = [5, 10])
    ax = plt.gca()

    # In order to place the nodes in the visualization of the graph in a sagittal view, we use L and S coordinates (the view will be from the coronal plane, P axis)
    node_pos_dict_P = {}
    for n in G.nodes():
        node_pos_dict_P[n] = [G.nodes(data=True)[n]["pos"][0], G.nodes(data=True)[n]["pos"][2]]

    if label is not None:
        edge_labels = nx.get_edge_attributes(G, label)
        nx.draw(G, node_pos_dict_P, node_size=20, ax=ax)
        nx.draw_networkx_edge_labels(G, node_pos_dict_P, edge_labels = edge_labels, ax=ax)
    else:
        nx.draw(G, node_pos_dict_P, node_size=20, ax=ax)

    plt.savefig(os.path.join(caseDir, filename))
    plt.close()

def makeSupersegmentPlots(caseDir, supersegments):
    # We can visualize first the original hierarchic dense graph
    configurationTitles = ["Femoral + right + anterior",
                        "Femoral + right + posterior",
                        "Femoral + left + anterior",
                        "Femoral + left + posterior",
                        "Radial + right + anterior",
                        "Radial + right + posterior",
                        "Radial + left + anterior",
                        "Radial + left + posterior"]

    rows = 2
    columns = len(configurationTitles) // rows

    _, ax = plt.subplots(rows, columns, figsize = [16, 18])

    for idxAccess, access in enumerate(supersegments.keys()):
        for idx, supersegment in enumerate(supersegments[access]):
            highlightNode = None

            for node in supersegment:
                if supersegment.nodes[node]["hierarchy"] == 0:
                    highlightNode = node
                
            colorPalette = mcp.gen_color(cmap = "bwr", n = 2)
            colorMap = [colorPalette[not supersegment.nodes[node]["isSupersegment"]] for node in supersegment] 
            
            if highlightNode is not None:
                colorMap[highlightNode] = "chartreuse"

            # In order to place the nodes in the visualization of the graph in a sagittal view, we use L and S coordinates (the view will be from the coronal plane, P axis)
            node_pos_dict_P = {}
            for n in supersegment.nodes():
                node_pos_dict_P[n] = [supersegment.nodes(data=True)[n]["pos"][0], supersegment.nodes(data=True)[n]["pos"][2]]

            nx.draw(supersegment, node_pos_dict_P, node_size=10, node_color=colorMap, ax=ax[(4 * idxAccess + idx) // columns, (4 * idxAccess + idx) % columns])
            ax[(4 * idxAccess + idx) // columns, (4 * idxAccess + idx) % columns].set_title(configurationTitles[(4 * idxAccess + idx)], fontsize=12)
            ax[(4 * idxAccess + idx) // columns, (4 * idxAccess + idx) % columns].set_xlim([8, 200])
            ax[(4 * idxAccess + idx) // columns, (4 * idxAccess + idx) % columns].set_ylim([-10, 280])
        
    plt.savefig(os.path.join(caseDir, "supersegments.png"))

def makeSupersegmentPlot(caseDir, supersegment, patientConfiguration):

    _ = plt.figure(figsize = [5, 10])
    ax = plt.gca()

    highlightNode = None
    for node in supersegment:
        if supersegment.nodes[node]["hierarchy"] == 0:
            highlightNode = node
        
    colorPalette = mcp.gen_color(cmap = "bwr", n = 2)
    colorMap = [colorPalette[not supersegment.nodes[node]["isSupersegment"]] for node in supersegment] 
    
    if highlightNode is not None:
        colorMap[highlightNode] = "chartreuse"

    # In order to place the nodes in the visualization of the graph in a sagittal view, we use L and S coordinates (the view will be from the coronal plane, P axis)
    node_pos_dict_P = {}
    for n in supersegment.nodes():
        node_pos_dict_P[n] = [supersegment.nodes(data=True)[n]["pos"][0], supersegment.nodes(data=True)[n]["pos"][2]]

    nx.draw(supersegment, node_pos_dict_P, node_size=10, node_color=colorMap)
    ax.set_title(patientConfiguration["Access"] + " + " + patientConfiguration["Laterality"] + " + " + patientConfiguration["Antero-posterior"] + ". Time: " + str(patientConfiguration["Time first angiography"]), fontsize=12)
    ax.set_xlim([8, 200])
    ax.set_ylim([-10, 280])
        
    plt.savefig(os.path.join(caseDir, "thrombectomyConfiguration", "supersegment.png"))

def addConfigurationFeatures(G, patientConfiguration):
    ''' Adds global features from patient configuration to graph.
    
    '''
    # Interventionalist
    # Define dict
    interventionalistsDict = {}
    interventionalistsDict["David"] = 0
    interventionalistsDict["Tomasello"] = 1
    interventionalistsDict["Ribo"] = 2
    interventionalistsDict["Piñana"] = 3
    interventionalistsDict["Coscojuela"] = 4
    interventionalistsDict["Remullo"] = 5
    interventionalistsDict["Bellvitge"] = 6
    interventionalistsDict["Requena"] = 7
    interventionalistsDict["Marta"] = 8
    internventionalistsOneHot = np.zeros(9)
    if patientConfiguration["Interventionalist"] in interventionalistsDict.keys():
        internventionalistsOneHot[interventionalistsDict[patientConfiguration["Interventionalist"]]] = 1.
    G.graph["Interventionalist"] = internventionalistsOneHot

    # Date
    year, month, _ = patientConfiguration["Date"].split("-")
    monthsFromJan2018 = 12 * (int(year) - 2018) + int(month) - 1
    G.graph["Months from Jan 2018"] = monthsFromJan2018

    # Access
    if patientConfiguration["Access"] == "Femoral":
        G.graph["Access"] = 0
    elif patientConfiguration["Access"] == "Radial":
        G.graph["Access"] = 1
    else:
        G.graph["Access"] = 2

    # DCP
    if not math.isnan(patientConfiguration["DCP"]):
        G.graph["DCP"] = 1
    else:
        G.graph["DCP"] = 0

    # Antero-posterior
    if patientConfiguration["Antero-posterior"] == "Anterior":
        G.graph["Antero-posterior"] = 0
    elif patientConfiguration["Antero-posterior"] == "Posterior":
        G.graph["Antero-posterior"] = 1
    else:
        G.graph["Antero-posterior"] = 2
        
    # Laterality
    if patientConfiguration["Laterality"] == "Right":
        G.graph["Laterality"] = 0
    elif patientConfiguration["Laterality"] == "Left":
        G.graph["Laterality"] = 1
    else:
        G.graph["Laterality"] = 2

    # Impossible accesses
    if not math.isnan(patientConfiguration["TFA impossible"]):
        G.graph["TFA impossible"] = 1
    else:
        G.graph["TFA impossible"] = 0
        
    if not math.isnan(patientConfiguration["TRA impossible"]):
        G.graph["TRA impossible"] = 1
    else:
        G.graph["TRA impossible"] = 0

    # Total supersegment length
    totalLength = 0
    for node in G:
        if G.nodes[node]["isSupersegment"] > 0.5:
            totalLength += G.nodes[node]["features"]["Segment length"]    
    G.graph["Total length"] = totalLength

    return G

def selectVertebrobasilarLaterality(G, laterality):
    radiusR, radiusL = [], []

    for node in G:
        if G.nodes[node]["Vessel type name"] == "RVA":
            try:
                radiusR.append(G.nodes[node]["features femoral"]["Mean radius"])
            except:
                pass
        elif G.nodes[node]["Vessel type name"] == "LVA":
            try:
                radiusL.append(G.nodes[node]["features femoral"]["Mean radius"])
            except:
                pass

    if len(radiusR) == 0 and len(radiusL) != 0:
        return "Left"
    elif len(radiusR) != 0 and len(radiusL) == 0:
        return "Right"
    elif len(radiusR) == 0 and len(radiusL) == 0:
        return laterality
    else:
        if len(radiusR) > 3 * len(radiusL):
            return "Right"
        elif len(radiusL) > 3 * len(radiusR):
            return "Left"
        else:
            if np.mean(radiusR) >= np.mean(radiusL):
                return "Right"
            else:
                return "Left"
