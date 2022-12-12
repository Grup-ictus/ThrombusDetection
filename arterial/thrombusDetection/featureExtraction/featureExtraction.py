import numpy as np
import pandas as pd
import SimpleITK as sitk
import radiomics as rx
import json as js
import pickle
import networkx as nx
import os

def thrombusFeatures(case_dir):
    prev_dir = case_dir
    case_id = os.path.basename(case_dir)
    case_dir = os.path.join(case_dir, 'cerebralArteries','thrombus')
    casePath = os.path.join(case_dir, 'thrombusPrediction.nii.gz')

    volume = getThrombusVolume(casePath)
    resultDict = radiomicsExtractor(case_dir)

    resultDict['Volume_mm3'] = volume

    resultPath = os.path.join(os.path.dirname(case_dir), 'radiomicFeatures.pickle')
    with open(resultPath, 'wb') as f:
        pickle.dump(resultDict ,f)

def getThrombusVolume(casePath):
    stats = sitk.StatisticsImageFilter()
    img = sitk.ReadImage(casePath)
    stats.Execute(img)

    nvoxels = stats.GetSum()

    spacing = img.GetSpacing()
    voxvol = spacing[0]*spacing[1]*spacing[2]

    volume = nvoxels * voxvol
    return volume


def radiomicsExtractor(case_dir):
    casePath = os.path.join(case_dir, 'thrombusPrediction.nii.gz')
    casePathCT = os.path.join(case_dir, 'croppedCT.nii.gz')
    casePathCTA = os.path.join(case_dir, 'croppedCTA.nii.gz')

    yamlDir = os.path.join(os.environ["arterial_dir"], "Params.yaml")
    extractor = rx.featureextractor.RadiomicsFeatureExtractor(yamlDir)

    extractor.enableAllFeatures

    resultCT = dict(extractor.execute(casePathCT, casePath))
    resultCTA = dict(extractor.execute(casePathCTA, casePath))

    result = {}
    for key in resultCTA: 
        result[key + '_CTA'] = resultCTA[key]
    for key in resultCT: 
        result[key + '_CT'] = resultCT[key]

    return result

def getTypeOfVessel(prev_dir, case_id):

    # Get segemntation affine and remove translation
    segmentation = nib.load(os.path.join(prev_dir, case_id + '_segmentation.nii.gz'))
    
    affine = segmentation.affine
    affine[0,0] = abs(affine[0,0])
    affine[:3,3] = [0, 0, 0]

    # Get thrombus patch coordiantes
    patch_coords = np.loadtxt(os.path.join(prev_dir, 'thrombus', 'patchCoordinates.txt'))
    x, y, z = (coords[0]+coords[1])//2, (coords[2]+coords[3])//2, (coords[4]+coords[5])//2

    # Convert coords from RAS to mm
    centerpoint = [x,y,z]
    centerpoint_ijk = nib.affines.apply_affine(affine, centerpoint)

    # Load graph
    G = nx.gpickle.read_gpickle(os.path.join(prev_dir, 'graph_predXGB.pickle'))

    # Create dictionary to store content
    distance = {'cell_id':[],
            'vessel type':[],
            'vessel type name': [],
            'distance':[]
           }
    # Iterate over the graph to identify the graph closer to the centerpoint
    for n0, n1 in G.edges:
        distance['cell_id'].append(G[n0][n1]['cell_id'])
        distance['vessel type'].append(G[n0][n1]['Vessel type'])
        distance['vessel type name'].append(G[n0][n1]['Vessel type name'])
        distance['distance'].append(euclidean(G[n0][n1]['pos'], centerpoint_ijk))
        
    df = pd.DataFrame(distance)
    min_row = df.loc[df['distance'].idxmin()]
    return min_row

    