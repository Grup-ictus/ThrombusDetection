#    Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

from arterial.thrombusDetection.segmentation.cerebralSegmentation import nnUNetInference, nnUNetThrombusInference, nnUNetAlternateThrombusInference
from arterial.thrombusDetection.segmentation.sanityCheck import sanityChecks

class cerebralSegmenter():
     
    
    def __init__(self, case_dir):
        """
        Initializes object of the cerebralSegmenter class.

        Parameters
        ----------
        case_dir : string or path-like object
            Path to case directory. 

        Returns
        -------
        
        """
        self.case_dir = case_dir
    
    def predict(self):
        """
        This method calls perform_inference to perform inference using a trained nnunet
        model over the cerebral part of the cropped CTA. The CTA should be in nifti format, within the 
        self.case_dir directory and with case_id being the basename of the self.case_dir,
        the name convention used should be:

        >>> {case_dir}/cerebralArteries/{case_id}.nii.gz

        At the end of the segmentation prediction, a nifti file with the format:
        
        >>> {case_dir}/cerebralArteries/{case_id}_segmentation.nii.gz
        
        should be generated in the self.case_dir. This class acts as a wrapper
        for the arterial.thrombusDetection.segmentation.nnUNetInference() function. 

        Parmeters
        ---------
        case_dir : string or path-like object
            Path to case directory.

        Returns
        -------

        """
        nnUNetInference(self.case_dir)

    def predictThrombus(self):
        """
        This method calls nnUNetThrombusInference to perform inference using a trained nnunet
        model over the cropped CTA and CT to segemnt the thrombus. The images should be in nifti 
        format, within the self.case_dir/cerebralArteries/thrombus/cropped/ directory,
        the name convention used should be:

        >>> {case_dir}/cerebralArteries/thrombus/cropped/croppedCT.nii.gz
        >>> {case_dir}/cerebralArteries/thrombus/cropped/croppedCTA.nii.gz

        At the end of the segmentation prediction, a nifti file with the format:
        
        >>> {case_dir}/cerebralArteries/thrombus/cropped/thrombusPrediction.nii.gz
        
        should be generated. This class acts as a wrapper
        for the arterial.thrombusDetection.segmentation.nnUNetThrombusInference() function. 

        Parmeters
        ---------
        case_dir : string or path-like object
            Path to case directory.

        Returns
        -------

        """
        nnUNetThrombusInference(self.case_dir)
    
    
    def predictAlternateThrombus(self):
        """
        This method calls nnUNetThrombusInference to perform inference using a trained nnunet
        model over the cropped CTA and CT to segemnt the thrombus. The images should be in nifti 
        format, within the self.case_dir/cerebralArteries/thrombus/cropped/ directory,
        the name convention used should be:

        >>> {case_dir}/cerebralArteries/thrombus/alternate/croppedCT.nii.gz
        >>> {case_dir}/cerebralArteries/thrombus/alternate/croppedCTA.nii.gz

        At the end of the segmentation prediction, a nifti file with the format:
        
        >>> {case_dir}/cerebralArteries/thrombus/alternate/thrombusPrediction.nii.gz
        
        should be generated. This class acts as a wrapper
        for the arterial.thrombusDetection.segmentation.nnUNetThrombusInference() function. 

        Parmeters
        ---------
        case_dir : string or path-like object
            Path to case directory.

        Returns
        -------

        """
        passport = sanityChecks(self.case_dir)
        nnUNetAlternateThrombusInference(self.case_dir)

