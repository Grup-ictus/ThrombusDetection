from arterial.thrombusDetection.featureExtraction.featureExtraction import thrombusFeatures


class FeatureExtractor():
    """
    """
    def __init__(self, case_dir):
        """
        Initializes object of the FeatureExtractor class

        Parameters
        ----------
        case_dir : string or path-like object
            Path to case directory. 

        Returns
        -------

        """

        self.case_dir = case_dir
    
    def extract_thrombus_features(self):
        """
        Calls function to extract radiomic and geometrical
        features of the segmented thrombus. 

        After the execution a pickle file si generated with 
        a dataframe encompassing all the extracted attributes
        
        Parameters
        ----------
        case_dir : string or path-like object
            Path to case directory. 

        Returns
        -------
        Pickle filem

        """
        thrombusFeatures(self.case_dir)