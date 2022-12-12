import os 
import nibabel as nib
import pandas as pd 
import numpy as np
import scipy
import json

from scipy.spatial.distance import euclidean

from skimage.measure import regionprops,regionprops_table, label


def patchExtraction(case_dir, side, confidence = 0, size = 96):
    """
    
    """
    if case_dir[-1:] == '/':
        case_dir = case_dir[:-1]

    previousDir = case_dir
    case_id = os.path.basename(case_dir)
    case_dir = os.path.join(case_dir, 'cerebralArteries')
    
    case_dir_cerebral = os.path.join(case_dir, 'thrombus')
    if not os.path.exists(case_dir_cerebral): os.mkdir(case_dir_cerebral)

    G = openpkl(os.path.join(case_dir, 'graph_predXGB.pickle'))
    
    # Coordinates  
    print('Obtaining vessel coordinates...')
    coordinates = getCoords(G, side)
    if side == 'L':
        altern_coordiantes = getCoords(G, 'R')
        sideX = 'R'
    else:
        altern_coordiantes = getCoords(G, 'L')
        sideX = 'L'
        
    # Load nifti images
    print('Loading images')
    print(case_id)
    
    nifti = nib.load(os.path.join(case_dir, case_id + "_NCCT.nii.gz"))
    CT = nifti.get_fdata()
    CTA = nib.load(os.path.join(case_dir, case_id + ".nii.gz")).get_fdata()
    SEG = nib.load(os.path.join(case_dir, case_id + "_TH.nii.gz")).get_fdata()

#     labelMask = label(SEG)
#     properties = regionprops(labelMask)
# #     x, y, z = [x.centroid for x in properties][0]
#     for i in properties: centroide = i.centroid
#     xim, yim, zim = centroide

    z_org = SEG.shape[2]
    aff = nifti.affine
    header = nifti.header
    
    if not CT.shape == CTA.shape:

        CT_new = np.empty(shape = (CT.shape[0], CT.shape[1], CTA.shape[2]))
        CT_new[:,:,CTA.shape[2]-CT.shape[2]:] = CT
        CT = CT_new    

    # Reduce bounding box
    print('Computing Bounding Box limits')
    x_min, x_max, y_min, y_max, z_min, z_max = getReducedBBox(coordinates, 
                                                              side = side, 
                                                              affine = aff, 
                                                              Graph = G)
        # Get prediction for alternate coordinates                                         
    x_minX, x_maxX, y_minX, y_maxX, z_minX, z_maxX = getReducedBBox(altern_coordiantes, 
                                                              side = sideX, 
                                                              affine = aff, 
                                                              Graph = G)
    
    # Set size  to size x size x size
    if x_max - x_min < 128 or y_max - y_min < size or  z_max - z_min < size:
         x_min, x_max, y_min, y_max, z_min, z_max = ensureSize(size, x_min, x_max, y_min, y_max, z_min, z_max, CTA)
    if x_maxX - x_minX < 128 or y_maxX - y_minX < size or  z_maxX - z_minX < size:
         x_minX, x_maxX, y_minX, y_maxX, z_minX, z_maxX = ensureSize(size, x_minX, x_maxX, y_minX, y_maxX, z_minX, z_maxX, CTA)

    # print()
    # print('Predicted')
    # checkIn(xim, x_max, x_min, yim, y_max, y_min, zim, z_max, z_min)
    # print()
    # print('Alternate')
    # checkIn(xim, x_maxX, x_minX, yim, y_maxX, y_minX, zim, z_maxX, z_minX)

    # Crop nifti images
    CT  = CT[x_min:x_max,y_min:y_max,z_min:z_max]
    CTA = CTA[x_min:x_max,y_min:y_max,z_min:z_max]
    SEG = SEG[x_min:x_max,y_min:y_max,z_min:z_max]

    CTX  = CT[x_minX:x_maxX,y_minX:y_maxX,z_minX:z_maxX]
    CTAX = CTA[x_minX:x_maxX,y_minX:y_maxX,z_minX:z_maxX]
    SEGX = SEG[x_minX:x_maxX,y_minX:y_maxX,z_minX:z_maxX]

    # Save nifti images
    print('Saving...')
    SEG = np.round(SEG, 0).astype(np.int32)
    SEGX = np.round(SEGX, 0).astype(np.int32)

    CT = nib.Nifti1Image(CT, aff, header)
    CTA = nib.Nifti1Image(CTA, aff, header)
    SEG = nib.Nifti1Image(SEG, aff, header)

    CTX = nib.Nifti1Image(CTX, aff, header)
    CTAX = nib.Nifti1Image(CTAX, aff, header)
    SEGX = nib.Nifti1Image(SEGX, aff, header)

    case_dir_cerebral_Altern = os.path.join(case_dir_cerebral, 'alternative')
    if not os.path.exists(case_dir_cerebral_Altern): os.mkdir(case_dir_cerebral_Altern)

    patchCoordinates = np.array([x_min, x_max, y_min, y_max, z_min, z_max])
    np.savetxt(os.path.join(case_dir_cerebral, 'patchCoordinates.txt'), patchCoordinates, fmt='%1.3f')

    nib.save(CT, os.path.join(case_dir_cerebral, 'croppedCT.nii.gz'))
    nib.save(CTA, os.path.join(case_dir_cerebral, 'croppedCTA.nii.gz'))
    nib.save(SEG, os.path.join(case_dir_cerebral, 'croppedSEG.nii.gz'))

    patchCoordinates = np.array([x_minX, x_maxX, y_minX, y_maxX, z_minX, z_maxX])
    np.savetxt(os.path.join(case_dir_cerebral_Altern, 'patchCoordinates.txt'), patchCoordinates, fmt='%1.3f')

    nib.save(CT, os.path.join(case_dir_cerebral_Altern, 'croppedCT.nii.gz'))
    nib.save(CTA, os.path.join(case_dir_cerebral_Altern, 'croppedCTA.nii.gz'))
    nib.save(SEG, os.path.join(case_dir_cerebral_Altern, 'croppedSEG.nii.gz'))

def checkIn(x, x_max, x_min, y, y_max, y_min, z, z_max, z_min):
    x_true, y_true, z_true = False, False, False
    
    if int(x) in range(x_min, x_max):
        x_true = True
        print('-------------x bien')
    if int(y) in range(y_min, y_max):
        y_true = True
        print('-------------y bien')
    if int(z) in range(z_min, z_max):
        z_true = True
        print('-------------z bien')
    
def getReducedBBox(coordinates, side, affine, Graph):
        #    X coordiantes 
    offset_x = 0.23*512
    #offset_x = 0
    if side == 'R':

        maxx = int(512//2)
        minx = int(0 + offset_x) 
        medium = int(512/2) - int((maxx-minx)/2)
        x_min = medium - 48
        x_max = medium + 48
    #         x_min = minx
    #         x_max = x_min + 96
    else:
        minx = int(512//2) 
        maxx = int(512 - offset_x)
        medium = int(minx + int((maxx-minx)/2))
        x_min = medium - 48
        x_max = medium + 48

    if len(coordinates) == 0:
        print('Not graph in this side: Taking data from inverse')
        if side == 'R':
            coordinates = getCoords(Graph, 'L')
        else:
            coordinates = getCoords(Graph, 'R')
    T = 0        
    YCoord = [case[1] for case in coordinates]
    ZCoord = [case[2] for case in coordinates]
    XCoord = [case[0] for case in coordinates]

    x_min = int((1 - T) * min(XCoord))
    x_max = int((1 + T) * max(XCoord))

    y_min = int((1 - T) * min(YCoord))
    y_max = int((1 + T) * max(YCoord))

    z_min = int((1 - T) * min(ZCoord))
    z_max = int((1 + T) * max(ZCoord))

    # Contralateral coordinates
    if side == 'R':
        coordinatesL = getCoords(Graph, 'L')
        YCoordL = [case[1] for case in coordinatesL]
        YCoord += YCoordL

        ZCoordL = [case[2] for case in coordinatesL]
        ZCoord += ZCoordL

    else:
        coordinatesR = getCoords(Graph, 'R')
        YCoordR = [case[1] for case in coordinatesR]
        YCoord += YCoordR
        
        ZCoordR = [case[2] for case in coordinatesR]
        ZCoord += ZCoordR


    XI, YI, ZI = int((x_max+x_min)/2), int(np.mean(YCoord)), int(1.15 * np.mean(ZCoord))
    coordMM = [XI, YI, ZI]
    affine_with = affine.copy()
    affine = affine_with
    affine[0,0] = abs(affine_with[0,0])
    affine[:3,3] = [0, 0, 0]

    point_vox = nib.affines.apply_affine(np.linalg.inv(affine), coordMM)
    # print(point_vox)
    point_vox[0] = medium
    return int(point_vox[0]),int(point_vox[0]),int(point_vox[1]),int(point_vox[1]),int(point_vox[2]),int(point_vox[2])

def ensureSize(size, x_min, x_max, y_min, y_max, z_min, z_max, CTOriginal):
    shape = CTOriginal.shape
    n = 0
    while x_max - x_min < 128:
        if n == 0:
            if shape[0] > x_max:
                x_max += 1
            n = 1
        else:
            x_min -= 1
            n = 0
            
    n = 0
    while y_max - y_min < size:
        if n == 0:
            if shape[1] > y_max:
                y_max += 1
            n = 1
        else:
            y_min -= 1
            n = 0
    
    n = 0
    while z_max - z_min < size:
        if n == 0:
            if shape[2] > z_max:
                z_max += 1
            n = 1
        else:
            z_min -= 1
            n = 0
    
    return x_min, x_max, y_min, y_max, z_min, z_max

    
def get_ICACoord(G, side):
    coordD = []
#     coordP = []
    if side == 'R':
        for n0, n1 in G.edges:
            if G[n0][n1]['Vessel type name'] == 'RICA':
                coordD.append([G[n0][n1]['features']['distal bifurcation position i'], G[n0][n1]['features']['distal bifurcation position j'], G[n0][n1]['features']['distal bifurcation position k']])
    #             coordP.append(G[n0][n1]['proximal bifurcation position'])
    
    if side == 'L':
        for n0, n1 in G.edges:
            if G[n0][n1]['Vessel type name'] == 'LICA':
                coordD.append([G[n0][n1]['features']['distal bifurcation position i'], G[n0][n1]['features']['distal bifurcation position j'], G[n0][n1]['features']['distal bifurcation position k']])
    #             coordP.append(G[n0][n1]['proximal bifurcation position'])
    def get_z(coordinate):
        return coordinate[2]
    
    coordICA = []
    if len(coordD) != 0:
        coordICA.append(max(coordD, key=get_z))
        
    return coordICA

def get_M1Coord(G, side):
    coordD = []
    coordP = []
    
    if side == 'R':
        for n0, n1 in G.edges:
            if G[n0][n1]['Vessel type name'] == 'RM1':
                coordD.append([G[n0][n1]['features']['distal bifurcation position i'], G[n0][n1]['features']['distal bifurcation position j'], G[n0][n1]['features']['distal bifurcation position k']])
                coordP.append([G[n0][n1]['features']['proximal bifurcation position i'], G[n0][n1]['features']['proximal bifurcation position j'], G[n0][n1]['features']['proximal bifurcation position k']])
    if side == 'L':
        for n0, n1 in G.edges:
            if G[n0][n1]['Vessel type name'] == 'LM1':
                coordD.append([G[n0][n1]['features']['distal bifurcation position i'], G[n0][n1]['features']['distal bifurcation position j'], G[n0][n1]['features']['distal bifurcation position k']])
                coordP.append([G[n0][n1]['features']['proximal bifurcation position i'], G[n0][n1]['features']['proximal bifurcation position j'], G[n0][n1]['features']['proximal bifurcation position k']])            
    return coordD, coordP

def get_A1Coord(G, side):
    coordP = []
    
    if side == 'L':
        for n0, n1 in G.edges:
            if G[n0][n1]['Vessel type name'] == 'LA1':
                coordP.append([G[n0][n1]['features']['proximal bifurcation position i'], G[n0][n1]['features']['proximal bifurcation position j'], G[n0][n1]['features']['proximal bifurcation position k']])
    
    def get_z(coordinate):
        return coordinate[2]
    
    coordA1 = []
    coordA1.append(min(coordP, key=get_z))
    
    return coordA1

def get_M2Coord(G, side):
    coordD_M1 = []
    coordD = []
    coordP = []
    dist = []
    final_coordD = []
    final_coordP = []
    
    if side == 'R':
    
        for n0, n1 in G.edges:
            if G[n0][n1]['Vessel type name'] == 'RM1':
                coordD_M1.append([G[n0][n1]['features']['distal bifurcation position i'], G[n0][n1]['features']['distal bifurcation position j'], G[n0][n1]['features']['distal bifurcation position k']])
            if G[n0][n1]['Vessel type name'] == 'RM2':
                coordD.append([G[n0][n1]['features']['distal bifurcation position i'], G[n0][n1]['features']['distal bifurcation position j'], G[n0][n1]['features']['distal bifurcation position k']])
                coordP.append([G[n0][n1]['features']['proximal bifurcation position i'], G[n0][n1]['features']['proximal bifurcation position j'], G[n0][n1]['features']['proximal bifurcation position k']])

        new_coordP = []
        new_coordD = []

        if len(coordD_M1)>0 and len(coordP)>0:
            for i in range(len(coordP)):
                distPresent = euclidean(coordD_M1[0], coordP[i])
                dist.append(distPresent)
                if min(dist) == distPresent:

                    dist = [distPresent]
                    new_coordP = [coordP[i]]
                    new_coordD = [coordD[i]]
                else:
                    dist.remove(distPresent)
            if len(new_coordD)>0 and len(new_coordP)>0:
                final_coordD.append(new_coordD[0])
                final_coordP.append(new_coordP[0])
            
    coordD_M1 = []
    coordD = []
    coordP = []
    
    if side == 'L':
    
        for n0, n1 in G.edges:
            if G[n0][n1]['Vessel type name'] == 'LM1':
                coordD_M1.append([G[n0][n1]['features']['distal bifurcation position i'], G[n0][n1]['features']['distal bifurcation position j'], G[n0][n1]['features']['distal bifurcation position k']])
            if G[n0][n1]['Vessel type name'] == 'LM2':
                coordD.append([G[n0][n1]['features']['distal bifurcation position i'], G[n0][n1]['features']['distal bifurcation position j'], G[n0][n1]['features']['distal bifurcation position k']])
                coordP.append([G[n0][n1]['features']['proximal bifurcation position i'], G[n0][n1]['features']['proximal bifurcation position j'], G[n0][n1]['features']['proximal bifurcation position k']])

        new_coordP = []
        new_coordD = []

        if len(coordD_M1)>0 and len(coordP)>0:
            for i in range(len(coordP)):
                distPresent = euclidean(coordD_M1[0], coordP[i])
                dist.append(distPresent)
                if min(dist) == distPresent:

                    dist = [distPresent]
                    new_coordP = [coordP[i]]
                    new_coordD = [coordD[i]]
                else:
                    dist.remove(distPresent)
            if len(new_coordD)>0 and len(new_coordP)>0:
                final_coordD.append(new_coordD[0])
                final_coordP.append(new_coordP[0])
                
    return final_coordD, final_coordP
            
def getCoords(G, side, typex = False):
    
    # ICA    
    distal_ICA = get_ICACoord(G, side)
#     print(distal_ICA)
#     print()
    # M1 
    distal_M1, proximal_M1 = get_M1Coord(G, side)
#     print(distal_M1, proximal_M1)
#     print()
    # M2
    distal_M2, proximal_M2 = get_M2Coord(G, side)
#     print(distal_M2, proximal_M2)
#     print()
    # A1
#     proximal_A1 = get_A1Coord(G, side)
#     print(distal_M2, proximal_M2)
#     print()
    # Overall
#     print('M1', proximal_M1)
#     print(proximal_M2)
    
    coordinates = distal_ICA + distal_M1 + proximal_M1 +  proximal_M2 #distal_M2 + + proximal_A1
    
    return coordinates

def openpkl(datadir):
    import pickle5 as pickle
    with open(datadir, "rb") as fh:
        data = pickle.load(fh)
    return data

