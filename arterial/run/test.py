#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os

from arterial.thrombusDetection.pipelinePreparation.cerebralPreparator import cerebralPreparator
from arterial.thrombusDetection.segmentation.cerebralSegmenter import cerebralSegmenter
from arterial.thrombusDetection.centerline_extraction.centerline_extractor import CenterlineExtractor
from arterial.thrombusDetection.vesselLabelling.GrahEdgePredictor import edgePredictor
from arterial.thrombusDetection.thrombusBbox.bboxThExtractor import bboxThExtractor
from arterial.thrombusDetection.featureExtraction.featuresExtractor import FeatureExtractor

class ArterialProcessor():
    """
    ArterialProcessor class to perform the desired analysis specified by the parsed arguments upon
    command line call. Wraps all modules' classes and their methods within one object.
    
    """
    def __init__(self, parser):
        """
        Initializes object of the ArterialProcessor class, and creates objects for all module central classes
        as attributes for the ArterialProcessor object.

        Parameters
        ----------
        parser : argparse.ArgumentParser object
            Contains all parsed arguments from the perform_analysis.py call as attributes.
            These are:
            - case_dir : string or path-like object
                Path to case directory. 
            - no_display : bool, default = False
                Boolean variable to be used when running analysis on a headless server.
                In addition, add ```$xvfb-run --auto-servernum --server-num=1``` at the beggining
                of the command line call when executing the script from the command line.
                E.g.: ```$xvfb-run --auto-servernum --server-num=1 python perform_analysis.py -case_dir {case_dir} -no_display {True}```
            - skip_segmentation: bool, default = False
                If True, it will skip the segmentation process. Useful if segmentation is already done,
                as segmentation is time-consuming. False by default.

        Returns
        -------
        
        """
         # Parameters from parser
        self.case_dir = parser.case_dir
        self.no_display = parser.no_display
        self.skip_segmentation = parser.skip_segmentation

        # Initialize module classes
        self.cropper = cerebralPreparator(self.case_dir)
        self.cerebral_segmenter = cerebralSegmenter(self.case_dir)  
        self.centerline_extractor = CenterlineExtractor(self.case_dir, self.no_display)
        self.edge_predictor = edgePredictor(self.case_dir)
        self.bbox_extractor = bboxThExtractor(self.case_dir)
        self.feature_extractor = FeatureExtractor(self.case_dir)

    def perform_analysis(self):
        """
        Calls wrapper method from each of the Thrombus Detection Arterial sub-modules.

        Binary arguments from parser are used to specify 

        """
        print("Performing analysis over case {}.".format(os.path.basename(self.case_dir)))
        
        # Crop image and get the cerebral part of the original CTA
        self.perform_cerebral_pipeline_preparation()

        # Use nnUNet framework to get a bianry mask of the cerebral arteries
        self.perform_segmentation()

        # Process with VMTK, VTK the previous binary mask to obtain a mesh and centerlines format
        self.perform_centerline_extraction()

        # Generate a graph and predict which type of cerebral vessel corresponds to each edge contained in it
        self.perform_cerebral_vessel_labelling()

        # Predict the side of the occlusion and extracts a reduced image where the thrombus has most probability to be pllaced
        self.perform_thrombus_patch_extraction()

        # Predict thrombus bianry mask from cropped CT and CTA images
        self.perfom_thrombus_segmentation()

        # Sanity check


        # Extract thrombus centerline
        self.perform_thrombus_centerline()

        # Get and save thrombus features
        self.perform_feature_extraction()

    def perform_cerebral_pipeline_preparation(self):
        """
        Wrapper method of the coregistering and cropper module. Calls method to coregsiter CT image to CTA
        and perform cerebral cropping from both CT and CTA images.

        At the end of the execution, the following files should be generated:

        >>> {case_dir}/cerebralArteries/{os.path.basename(case_dir)}_CTA.nii.gz
        >>> {case_dir}/cerebralArteries/{os.path.basename(case_dir)}_label.nii.gz
    
        Parmeters
        ---------

        Returns
        -------

        """
        # self.cropper.coregister() ########################Un-comment line if coregister is needed#####################################
        self.cropper.crop()
    
    def perform_segmentation(self):
        """
        Wrapper method of the segmentation module. Calls method to perform segmentation.

        At the end of the execution, the following files should be generated:

        >>> {case_dir}/{os.path.basename(case_dir)}_segmentation.nii.gz
    
        Parmeters
        ---------

        Returns
        -------

        """
        # Predicts segmentation by nnunet inference
        if not self.skip_segmentation:
            print("Predicting segmentation...")
            self.cerebral_segmenter.predict()
            print("done \n")
        else:
            print("Skipping segmentation \n")

    
    def perform_centerline_extraction(self):
        """
        Wrapper method of the centerline_extraction module. Calls methods to perform centerline
        extraction, model branching and clipping as well as postprocessing of the centerline models.

        At the end of the execution, the following files should be generated:
        
        >>> case_dir/centerlines/centerlines{idx}.vtk
        >>> case_dir/segmentations/segmentation{idx}.vtk
        >>> case_dir/branch_models/branch_model{idx}.vtk
        >>> case_dir/branch_model.vtk
        >>> case_dir/clipped_models/clipped_model{idx}.vtk
        >>> case_dir/clipped_model.vtk
        >>> case_dir/centerline_segments_array.npy

        Parmeters
        ---------

        Returns
        -------
        
        """
        # Applies centerline preprocessing and extraction using Slicer and VMTK
        self.centerline_extractor.extract_centerline()
        # Performs centerline model branching with VMTK
        self.centerline_extractor.extract_branch_model()
        # Performs surface mdoel clipping with VMTK
        # self.centerline_extractor.extract_clipped_model()
        # Creates array for easier centerline analysis
        self.centerline_extractor.postprocess_centerline()

    def perform_cerebral_vessel_labelling(self):
        """
        Wrapper method of the vesselLabelling module. Calls method to perform cerevbal vessels automatic
        labelling from the graph previously generated.
        
        At the end of the execution, the following files should be generated:

        >>> {case_dir}/cerebralArteries/graph_predXGB.pickle
        >>> {case_dir}/cerebralArteries/graph_predXGB.png

        
        Parmeters
        ---------

        Returns
        -------

        """
        self.edge_predictor.preprocessing()
        self.edge_predictor.predict_vessel_types()

    def perform_thrombus_patch_extraction(self):
        
        """
        Wrapper method of the bboxThExtractor module which predicts the side of the
        occlusion as well as it extracts the patch of the image where the thrombus is
        predicted to be.
        
        At the end of the execution, the following files should be generated:

        >>> {case_dir}/cerebralArteries/SideDetection.pkl
        >>> {case_dir}/cerebralArteries/thrombus/cropped_CT.nii.gz
        >>> {case_dir}/cerebralArteries/thrombus/cropped_CTA.nii.gz
        >>> {case_dir}/cerebralArteries/thrombus/patchCoordinates.txt
        >>> {case_dir}/cerebralArteries/thrombus/alternative/cropped_CT.nii.gz
        >>> {case_dir}/cerebralArteries/thrombus/alternative/cropped_CTA.nii.gz
        >>> {case_dir}/cerebralArteries/thrombus/alternative/patchCoordinates.txt
        
        Parmeters
        ---------

        Returns
        -------

        """
        self.bbox_extractor.get_occlusion_side()
        self.bbox_extractor.get_thrombus_patch()

    def perfom_thrombus_segmentation(self):
        """
        Wrapper method of the segmentation module. Calls method to perform thrombussegmentation.

        At the end of the execution, the following files should be generated:

        >>> {case_dir}/cerebralArteries/thrombus/thrombusPrediction.nii.gz
    
        Parmeters
        ---------

        Returns
        -------
        """
        self.cerebral_segmenter.predictThrombus()
    
    def perform_thrombus_centerline(self):

        """
        Wrapper method of the centerline_extraction module. Calls methods to perform centerline
        extraction for thrombus bianry mask.
        At the end of the execution, the following files should be generated:
        
        >>> {case_dir}/cerebralArteries/thrombus/centerlines/centerlines{idx}.vtk
        >>> {case_dir}/cerebralArteries/thrombus/segmentations/segmentation{idx}.vtk

        Parmeters
        ---------

        Returns
        -------

        """

        self.centerline_extractor.extract_centerline_thrombus()


    def perform_feature_extraction(self):
        """
        Wrapper method fo the feature extraction module. Call method to perform extraction 
        of thrombus features from the thrombus, CTA and CT nifit files.

        At the end of the execution this files should be created:

        >>> {case_dir}/cerebralArteries/thrombus/radiomicFeatures.pickle

        Parmeters
        ---------

        Returns
        -------

        """
        self.feature_extractor.extract_thrombus_features()
