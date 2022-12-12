import numpy as np
import os
import pickle5
import pickle
import xgboost as xgb
import pandas as pd
import nibabel as nib


def OcclusionSide(case_dir, case_id, model, G):
    
    # Get labelled graph
    # case_dir = os.path.join(case_dir, case_id, 'cerebralArteries')
    casePath = os.path.join(case_dir, 'graph_predXGB.pickle')
    # G = 0
    # G = openpkl(casePath)
    
    # Init. dict
    dictio_lr = {'LM1_Presence': [],
                'RM1_Presence': [],
                'LM1_Length': [],
                'RM1_Length': [],
                'LM1_Radius': [],
                'RM1_Radius': [],
                'DistanceLICA_LM1': [],
                'DistanceRICA_RM1': [],
                'Distance_LM1_LM2': [],
                'Distance_RM1_RM2': [],
                'LM2_NumPoints': [],
                'RM2_NumPoints': [],
                'LM2_Length': [],
                'RM2_Length': [],
                'ProxDistRatioLM1': [],
                'ProxDistRatioRM1': [],
                'LICA_Length': [],
                'RICA_Length': [] }
    
    # Get graph features
    
    dictio_lr['LM1_Presence'], dictio_lr['RM1_Presence']=detectM1(G)
    dictio_lr['LM1_Length'], dictio_lr['RM1_Length']=getDistances(G)
    dictio_lr['LM1_Radius'], dictio_lr['RM1_Radius']=getM1Radius(G)
    dictio_lr['DistanceLICA_LM1'], dictio_lr['DistanceRICA_RM1']=connection_M1_ICA(G)
    dictio_lr['Distance_LM1_LM2'], dictio_lr['Distance_RM1_RM2']=connection_M1_M2(G)
    dictio_lr['LM2_NumPoints'], dictio_lr['RM2_NumPoints']=getNumPointsM2(G)
    dictio_lr['LM2_Length'], dictio_lr['RM2_Length']=getM2Length(G)
    dictio_lr['ProxDistRatioLM1'], dictio_lr['ProxDistRatioRM1']=getProximalDistalRatioM1(G)
    dictio_lr['LICA_Length'], dictio_lr['RICA_Length']=getICALenght(G)
    
    df = pd.DataFrame(dictio_lr, [str(case_id)])
     
    # Predict Side
    # file_name = "./THROMBUS_DETECTOR/SideDetection/GradientBoosting/GradientBoosting_SideDetection_Model.model"
    # model = xgb.Booster({'nthread': 4})  # init model
    # model.load_model(file_name)  # load data
    logit = np.round(model.predict(xgb.DMatrix(df)).item(), 3)
    predicted = np.round(logit, 0) 
    
    # Add Side
    if predicted == 0:
        dictio_lr['Occlusion']  = 'R'
        dictio_lr['Confidence'] = 1- logit
    else:
        dictio_lr['Occlusion']  = 'L'
        dictio_lr['Confidence'] = logit
        
    file_name = os.path.join(case_dir, 'SideDetection.pkl')
    
    df = pd.DataFrame(dictio_lr, [str(case_id)])
        
    # pickle.dump(df, open(file_name, "wb"))

    return df

def detectM1(G):
    # Iterate to find LM1 adn RM1 
    presence = [0, 0]
    for n0, n1 in G.edges:
        if G[n0][n1]['Vessel type name'] == 'LM1':
            presence[0] = 1
        if G[n0][n1]['Vessel type name'] == 'RM1':
            presence[1] = 1
    return presence

def getDistances(G):
    # Calculate distances of each branch
    distanceLM1 = []
    distanceRM1 = []
    for n0, n1 in G.edges:
        # 
        if G[n0][n1]['Vessel type name'] == 'LM1':
            distanceLM1.append(G[n0][n1]['features']['distance'])
        if G[n0][n1]['Vessel type name'] == 'RM1':
            distanceRM1.append(G[n0][n1]['features']['distance'])

    return [round(sum(distanceLM1), 2), round(sum(distanceRM1), 2)]

def getM1Radius(G):
    
    radiusLM1 = []
    radiusRM1 = []
    for n0, n1 in G.edges:
        # 
        if G[n0][n1]['Vessel type name'] == 'LM1':
            radiusLM1.append(G[n0][n1]['features']['mean radius'])
        if G[n0][n1]['Vessel type name'] == 'RM1':
            radiusRM1.append(G[n0][n1]['features']['mean radius'])
    if len(radiusRM1)>0:
        RmeanRad = round(np.mean(radiusRM1), 2)
    else:
        RmeanRad = 0
        
    if len(radiusLM1)>0:
        LmeanRad = round(np.mean(radiusLM1), 2)
    else:
        LmeanRad = 0
    
    return [LmeanRad, RmeanRad]

def getM2Length(G):
    lengthRM2 = []
    lengthLM2 = []
    
    for n0, n1 in G.edges: 
        if G[n0][n1]['Vessel type name'] == 'RM2':
            lengthRM2.append(G[n0][n1]['features']['distance'])
        if G[n0][n1]['Vessel type name'] == 'LM2':
            lengthLM2.append(G[n0][n1]['features']['distance'])
            
    if len(lengthRM2)>0:
        RLength = round(sum(lengthRM2), 2)
    else:
        RLength = 0
    if len(lengthLM2)>0:
        LLength = round(sum(lengthLM2), 2)
    else:
        LLength = 0    
    return [LLength, RLength]

def getProximalDistalRatioM1(G):
    
    LM1 = []
    RM1 = []
    for n0, n1 in G.edges:
        # 
        if G[n0][n1]['Vessel type name'] == 'LM1':
            LM1.append(G[n0][n1]['features']['proximal/distal radius ratio'])
        if G[n0][n1]['Vessel type name'] == 'RM1':
            RM1.append(G[n0][n1]['features']['proximal/distal radius ratio'])
            
    if len(LM1)>0:
        L_RATE = round(np.mean(LM1), 2)
    else:
        L_RATE = 1
        
    if len(RM1)>0:
        R_RATE = round(np.mean(RM1), 2)
    else:
        R_RATE = 1
    
    return [L_RATE, R_RATE]

def getICALenght(G):
    lengthRICA = []
    lengthLICA = []
    
    for n0, n1 in G.edges: 
        if G[n0][n1]['Vessel type name'] == 'RICA':
            lengthRICA.append(G[n0][n1]['features']['distance'])
        if G[n0][n1]['Vessel type name'] == 'LICA':
            lengthLICA.append(G[n0][n1]['features']['distance'])
            
    return [round(sum(lengthLICA), 2), round(sum(lengthRICA), 2)]

def getNumPointsM2(G):
    RM2 = []
    LM2 = []
    for n0, n1 in G.edges: 
        if G[n0][n1]['Vessel type name'] == 'RM2':
            RM2.append(G[n0][n1]['features']['number of points'])
        if G[n0][n1]['Vessel type name'] == 'LM2':
            LM2.append(G[n0][n1]['features']['number of points'])
    return [sum(LM2), sum(RM2)]
            
def openpkl(datadir):
    import pickle5 as pickle
    with open(datadir, "rb") as fh:
        data = pickle.load(fh)
    return data

def connection_M1_ICA(G):
    from scipy.spatial.distance import euclidean

    RICA = []
    LICA = []
    RM1 = []
    LM1 = []

    # LICA//RICA//LM1//RM1 proximal & distal points
    for n0, n1 in G.edges:
        if G[n0][n1]['Vessel type name'] == 'RICA':
            RICA.append([G[n0][n1]['features']['distal bifurcation position i'], G[n0][n1]['features']['distal bifurcation position j'], G[n0][n1]['features']['distal bifurcation position k']])
            RICA.append([G[n0][n1]['features']['proximal bifurcation position i'], G[n0][n1]['features']['proximal bifurcation position j'], G[n0][n1]['features']['proximal bifurcation position k']])
        if G[n0][n1]['Vessel type name'] == 'LICA':
            LICA.append([G[n0][n1]['features']['distal bifurcation position i'], G[n0][n1]['features']['distal bifurcation position j'], G[n0][n1]['features']['distal bifurcation position k']])
            LICA.append([G[n0][n1]['features']['proximal bifurcation position i'], G[n0][n1]['features']['proximal bifurcation position j'], G[n0][n1]['features']['proximal bifurcation position k']])
        if G[n0][n1]['Vessel type name'] == 'RM1':
            RM1.append([G[n0][n1]['features']['distal bifurcation position i'], G[n0][n1]['features']['distal bifurcation position j'], G[n0][n1]['features']['distal bifurcation position k']])
            RM1.append([G[n0][n1]['features']['proximal bifurcation position i'], G[n0][n1]['features']['proximal bifurcation position j'], G[n0][n1]['features']['proximal bifurcation position k']])
        if G[n0][n1]['Vessel type name'] == 'LM1':
            LM1.append([G[n0][n1]['features']['distal bifurcation position i'], G[n0][n1]['features']['distal bifurcation position j'], G[n0][n1]['features']['distal bifurcation position k']])
            LM1.append([G[n0][n1]['features']['proximal bifurcation position i'], G[n0][n1]['features']['proximal bifurcation position j'], G[n0][n1]['features']['proximal bifurcation position k']])
   
    if len(RICA)>0  and len(RM1)>0:
        # Calculate distance between all RICA and RM1 points
        distance_R = []
        for coord in RICA:
            for endpoint in RM1:
                distance_R.append(euclidean(coord, endpoint))
        dist_R = round(min(distance_R), 2)
    else:
        dist_R = 0
    if len(LICA)>0 and len(LM1)>0:
        # Calculate distance between all LICA and LM1 points
        distance_L = []
        for coord in LICA:
            for endpoint in LM1:
                distance_L.append(euclidean(coord, endpoint))
        dist_L = round(min(distance_L), 2)
    else:
        dist_L = 0
    
    return [dist_L, dist_R]

def connection_A1_M1(G):
    from scipy.spatial.distance import euclidean

    RA1 = []
    LA1 = []
    RM1 = []
    LM1 = []
    LA1 = []
    RA1 = []
    RICA = []
    LICA = []

    # LICA//RICA//LM1//RM1 proximal & distal points
    for n0, n1 in G.edges:
        if G[n0][n1]['Vessel type name'] == 'RA1':
            RA1.append([G[n0][n1]['features']['distal bifurcation position i'], G[n0][n1]['features']['distal bifurcation position j'], G[n0][n1]['features']['distal bifurcation position k']])
            RA1.append([G[n0][n1]['features']['proximal bifurcation position i'], G[n0][n1]['features']['proximal bifurcation position j'], G[n0][n1]['features']['proximal bifurcation position k']])
        if G[n0][n1]['Vessel type name'] == 'LA1':
            LA1.append([G[n0][n1]['features']['distal bifurcation position i'], G[n0][n1]['features']['distal bifurcation position j'], G[n0][n1]['features']['distal bifurcation position k']])
            LA1.append([G[n0][n1]['features']['proximal bifurcation position i'], G[n0][n1]['features']['proximal bifurcation position j'], G[n0][n1]['features']['proximal bifurcation position k']])
        if G[n0][n1]['Vessel type name'] == 'RM1':
            RM1.append([G[n0][n1]['features']['distal bifurcation position i'], G[n0][n1]['features']['distal bifurcation position j'], G[n0][n1]['features']['distal bifurcation position k']])
            RM1.append([G[n0][n1]['features']['proximal bifurcation position i'], G[n0][n1]['features']['proximal bifurcation position j'], G[n0][n1]['features']['proximal bifurcation position k']])
        if G[n0][n1]['Vessel type name'] == 'LM1':
            LM1.append([G[n0][n1]['features']['distal bifurcation position i'], G[n0][n1]['features']['distal bifurcation position j'], G[n0][n1]['features']['distal bifurcation position k']])
            LM1.append([G[n0][n1]['features']['proximal bifurcation position i'], G[n0][n1]['features']['proximal bifurcation position j'], G[n0][n1]['features']['proximal bifurcation position k']])
        if G[n0][n1]['Vessel type name'] == 'LA1':
            LA1.append([G[n0][n1]['features']['distal bifurcation position i'], G[n0][n1]['features']['distal bifurcation position j'], G[n0][n1]['features']['distal bifurcation position k']])
            LA1.append([G[n0][n1]['features']['proximal bifurcation position i'], G[n0][n1]['features']['proximal bifurcation position j'], G[n0][n1]['features']['proximal bifurcation position k']])
        if G[n0][n1]['Vessel type name'] == 'RA1':
            RA1.append([G[n0][n1]['features']['distal bifurcation position i'], G[n0][n1]['features']['distal bifurcation position j'], G[n0][n1]['features']['distal bifurcation position k']])
            RA1.append([G[n0][n1]['features']['proximal bifurcation position i'], G[n0][n1]['features']['proximal bifurcation position j'], G[n0][n1]['features']['proximal bifurcation position k']])
        if G[n0][n1]['Vessel type name'] == 'RICA':
            RICA.append([G[n0][n1]['features']['distal bifurcation position i'], G[n0][n1]['features']['distal bifurcation position j'], G[n0][n1]['features']['distal bifurcation position k']])
            RICA.append([G[n0][n1]['features']['proximal bifurcation position i'], G[n0][n1]['features']['proximal bifurcation position j'], G[n0][n1]['features']['proximal bifurcation position k']])
        if G[n0][n1]['Vessel type name'] == 'LICA':
            LICA.append([G[n0][n1]['features']['distal bifurcation position i'], G[n0][n1]['features']['distal bifurcation position j'], G[n0][n1]['features']['distal bifurcation position k']])
            LICA.append([G[n0][n1]['features']['proximal bifurcation position i'], G[n0][n1]['features']['proximal bifurcation position j'], G[n0][n1]['features']['proximal bifurcation position k']])
            

    if len(RA1)>0 and len(RM1)>0:
        # Calculate distance between all RICA and RM1 points
        distance_R = []
        for coord in RA1:
            for endpoint in RM1:
                distance_R.append(euclidean(coord, endpoint))
        dist_R = round(min(distance_R), 2)
    else:
        distance_R = []
        for coord in RA1:
            for endpoint in RM1:
                distance_R.append(euclidean(coord, endpoint))
        dist_R = round(min(distance_R), 2)
    
    if len(LA1)>0 and len(LM1)>0:
        # Calculate distance between all LICA and LM1 points
        distance_L = []
        for coord in LA1:
            for endpoint in LM1:
                distance_L.append(euclidean(coord, endpoint))
        dist_L = round(min(distance_L), 2)
    else:
        dist_L = np.nan
    
    return [dist_L, dist_R]

def connection_M1_M2(G):
    from scipy.spatial.distance import euclidean

    RM2 = []
    LM2 = []
    RM1 = []
    LM1 = []
    LA1 = []
    RA1 = []
    RICA = []
    LICA = []

    # LICA//RICA//LM1//RM1 proximal & distal points
    for n0, n1 in G.edges:
        if G[n0][n1]['Vessel type name'] == 'RM2':
            RM2.append([G[n0][n1]['features']['distal bifurcation position i'], G[n0][n1]['features']['distal bifurcation position j'], G[n0][n1]['features']['distal bifurcation position k']])
            RM2.append([G[n0][n1]['features']['proximal bifurcation position i'], G[n0][n1]['features']['proximal bifurcation position j'], G[n0][n1]['features']['proximal bifurcation position k']])
        if G[n0][n1]['Vessel type name'] == 'LM2':
            LM2.append([G[n0][n1]['features']['distal bifurcation position i'], G[n0][n1]['features']['distal bifurcation position j'], G[n0][n1]['features']['distal bifurcation position k']])
            LM2.append([G[n0][n1]['features']['proximal bifurcation position i'], G[n0][n1]['features']['proximal bifurcation position j'], G[n0][n1]['features']['proximal bifurcation position k']])
        if G[n0][n1]['Vessel type name'] == 'RM1':
            RM1.append([G[n0][n1]['features']['distal bifurcation position i'], G[n0][n1]['features']['distal bifurcation position j'], G[n0][n1]['features']['distal bifurcation position k']])
#             RM1.append(G[n0][n1]['proximal bifurcation position'])
        if G[n0][n1]['Vessel type name'] == 'LM1':
            LM1.append([G[n0][n1]['features']['distal bifurcation position i'], G[n0][n1]['features']['distal bifurcation position j'], G[n0][n1]['features']['distal bifurcation position k']])
#             LM1.append(G[n0][n1]['proximal bifurcation position'])
        if G[n0][n1]['Vessel type name'] == 'LA1':
            LA1.append([G[n0][n1]['features']['distal bifurcation position i'], G[n0][n1]['features']['distal bifurcation position j'], G[n0][n1]['features']['distal bifurcation position k']])
            LA1.append([G[n0][n1]['features']['proximal bifurcation position i'], G[n0][n1]['features']['proximal bifurcation position j'], G[n0][n1]['features']['proximal bifurcation position k']])
        if G[n0][n1]['Vessel type name'] == 'RA1':
            RA1.append([G[n0][n1]['features']['distal bifurcation position i'], G[n0][n1]['features']['distal bifurcation position j'], G[n0][n1]['features']['distal bifurcation position k']])
            RA1.append([G[n0][n1]['features']['proximal bifurcation position i'], G[n0][n1]['features']['proximal bifurcation position j'], G[n0][n1]['features']['proximal bifurcation position k']])
        if G[n0][n1]['Vessel type name'] == 'RICA':
            RICA.append([G[n0][n1]['features']['distal bifurcation position i'], G[n0][n1]['features']['distal bifurcation position j'], G[n0][n1]['features']['distal bifurcation position k']])
            RICA.append([G[n0][n1]['features']['proximal bifurcation position i'], G[n0][n1]['features']['proximal bifurcation position j'], G[n0][n1]['features']['proximal bifurcation position k']])
        if G[n0][n1]['Vessel type name'] == 'LICA':
            LICA.append([G[n0][n1]['features']['distal bifurcation position i'], G[n0][n1]['features']['distal bifurcation position j'], G[n0][n1]['features']['distal bifurcation position k']])
            LICA.append([G[n0][n1]['features']['proximal bifurcation position i'], G[n0][n1]['features']['proximal bifurcation position j'], G[n0][n1]['features']['proximal bifurcation position k']])

    if len(RM1)>0 and len(RM2)>0:
        # Calculate distance between all RICA and RM1 points
        distance_R = []
        for coord in RM2:
            for endpoint in RM1:
                distance_R.append(euclidean(coord, endpoint))
        dist_R = round(min(distance_R), 2)
        
    else:
        if len(RM2)<1:
            dist_R = 10
            
        else:
            distance_RA1 = []
            
            for coord in RA1:
                for endpoint in RM2:
                    distance_RA1.append(euclidean(coord, endpoint))
            
            distance_RICA = []
            for coord in RICA:
                for endpoint in RM2:
                    distance_RICA.append(euclidean(coord, endpoint))
                    
            RICARA1dist = distance_RICA + distance_RA1
            if len(RICARA1dist) > 0:
                dist_R = round(min(RICARA1dist), 2)
            else:
                dist_R = 10            
        
    if len(LM1)>0 and len(LM2)>0:   
        # Calculate distance between all LICA and LM1 points
        distance_L = []
        for coord in LM2:
            for endpoint in LM1:
                distance_L.append(euclidean(coord, endpoint))
        dist_L = round(min(distance_L), 2)
    else:
        if len(LM2)<1:
            dist_L = 10
        else:
            distance_LA1 = []
            for coord in LA1:
                for endpoint in LM2:
                    distance_LA1.append(euclidean(coord, endpoint))
            

            distance_LICA = []
            for coord in LICA:
                for endpoint in LM2:
                    distance_LICA.append(euclidean(coord, endpoint))
                    
            LICALA1dist = distance_LICA + distance_LA1
            
            if len(LICALA1dist)>0:
                dist_L = round(min(LICALA1dist), 2)
            else:
                dist_L = 10   

    return [dist_L, dist_R]


def lateralityPrediction(case_dir):
    """
    This code perform the prediction of the laterality of the occlusion. 
    First, it loads the automatic labelled graph, whcih is passed to the 
    OcclusionSide fucntion where the required features are computed and 
    stroed in a DataFrame so that the XGBoost model can predict the side
    of the occlusion.

    Parmeters
    ---------
        case_dir : string or path-like object
        Path to case directory. 

    Returns
    -------
        Side of the occluion as a string 'L' or 'R' (Left/Right)

        Confidence of the predictive model as float
    """  
    
    case_id = os.path.basename(case_dir)
    case_dir = os.path.join(case_dir, 'cerebralArteries')

    # Load graph
    G = openpkl(os.path.join(case_dir, 'graph_predXGB.pickle'))
    #     G_label = openpkl(os.path.join(case_dir, case_id, 'cerebralArteries', 'graph_label.pickle'))
    #    
    # print('Opening graph...')

    # Load XGB model & Infer occlusion side
    file_name = os.path.join(os.environ["arterial_dir"], "./thrombusDetection/thrombusBbox/GradientBoosting_SideDetection_Model1.model")
    model = xgb.Booster({'nthread': 4})  
    model.load_model(file_name)

    df = OcclusionSide(case_dir, case_id, model, G)
    side = df['Occlusion'].item()
    confidence = df['Confidence'].item() 
    #     side = 'L'
    print('Occluion detected at'+' '+ str(side) + ' ' + 'side with confidence' + '' + str(confidence))

    return side, confidence 