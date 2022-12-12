#    Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

from arterial.thrombusDetection.pipelinePreparation.cerebralCropping import cerebral_cropping
from arterial.thrombusDetection.pipelinePreparation.Coregistration import registeringCT2CTA

class cerebralPreparator():
     
    
    def __init__(self, case_dir):
        """
        Initializes object of the cerebralPreparator class.

        Parameters
        ----------
        case_dir : string or path-like object
            Path to case directory. 

        Returns
        -------
        
        """
        self.case_dir = case_dir
    
    def coregister(self):
        """
        This method calls regiteringCT2CTA to apply the corregistration algorithm to CT to fit 
        it with the CTA as a referencce

        -------------------------IN DEVELOPMENT--------------------------------------
        Parmeters
        ---------

        Returns
        -------
        """

        registeringCT2CTA(self.case_dir)
        pass

    def crop(self):
        """
        This method calls cerebral_cropping to generate a reduced bounding box of the origianl CTA 
        and the manually annotated segmentation mask so that the cranium region is only kept. The 
        CTA should be in nifti format, in the path self.case_path which is conformed by a case_dir 
        and a case_id, the name convention used should be:

        >>> {case_id}.nii.gz

        At the end of the segmentation prediction, the nifti files with the format:
        
        >>> {case_dir}/cerebralArteries/{case_id}_CTA.nii.gz
        >>> {case_dir}/cerebralArteries/{case_id}_label.nii.gz

        Parmeters
        ---------

        Returns
        -------

        """
        cerebral_cropping(self.case_dir)

