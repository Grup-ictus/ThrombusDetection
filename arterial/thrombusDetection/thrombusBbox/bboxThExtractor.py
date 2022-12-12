#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

from arterial.thrombusDetection.thrombusBbox.patchExtraction import patchExtraction
from arterial.thrombusDetection.thrombusBbox.SideDetection import lateralityPrediction


class bboxThExtractor():

    def __init__(self, case_dir):
        """
        Initializes object of the bboxThExtractor class.

        Parameters
        ----------
        case_dir : string or path-like object
            Path to case directory. 

        Returns
        -------
        
        """
        self.case_dir = case_dir 

    def get_occlusion_side(self):
        """
        This method calls the lateralityPredeiction in order to
        predict where the occlusion has more probability to be 
        placed by using an XGboost model that takes a set of constructed 
        graph features. This code expects as a convention:

        >>> {caseDir}/cerebralArteries/graph_predXGB.pickle

        At the end of the segmentation prediction, a DataFrame is saved 
        with the format:

        >>> {case_dir}/cerebralArteries/SideDetection.pkl

        Parmeters
        ---------
            case_dir : string or path-like object
            Path to case directory. 

        Returns
        -------
            Side of the occluion as a string 'L' or 'R' (Left/Right)

            Confidence of the predictive model as float

        
        """
        side, confidence = lateralityPrediction(self.case_dir)

        self.side = side
        self.confidence = confidence
    
    def get_thrombus_patch(self):
        """
        This method calls patchExtraction which crops the 
        original image given a set of heuristics applied on 
        the graph model that enables to obtain a 128 x 96 x 96 
        patch where the thrombus is supposed to be within the 
        bbox limits. This method exxpects a coregistered CTA 
        and CT in nifti format with the same shape and pixel diemnsions 

        At the end of the execution these files should be generated:

        >>> {case_dir}/cerebralArteries/thrombus/cropped_CT.nii.gz
        >>> {case_dir}/cerebralArteries/thrombus/cropped_CTA.nii.gz
        >>> {case_dir}/cerebralArteries/thrombus/patchCoordinates.txt
        >>> {case_dir}/cerebralArteries/thrombus/alternative/cropped_CT.nii.gz
        >>> {case_dir}/cerebralArteries/thrombus/alternative/cropped_CTA.nii.gz
        >>> {case_dir}/cerebralArteries/thrombus/alternative/patchCoordinates.txt
        
        Parmeters
        ---------
            case_dir : string or path-like object
            Path to case directory. 

        Returns
        -------
            

        """
        patchExtraction(self.case_dir, self.side, self.confidence)
