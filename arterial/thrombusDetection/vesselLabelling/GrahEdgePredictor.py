#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

from arterial.thrombusDetection.vesselLabelling.predictCV import predictVesselTypes
from arterial.thrombusDetection.vesselLabelling.preprocessing.preprocessing import build_simple_centerline_graph

class edgePredictor():
    
    def __init__(self, case_dir):
        """
        Initializes object to predict the cerebral vessel types per each edge
        """
        self.case_dir = case_dir
    
    def preprocessing(self):
        """
        Runs preprocessing for vessel labelling. Generates a networkx graph where
        vessel segments are represented by edges, and nodes correspond to bifurcations.
        We call these "simple" graphs. Also performs featurization at a segment level.

        At the end of the preprocessing, a graph and an image should be generated:
        
        >>> {case_dir}/cerebralArteries/graph_simple.pickle
        >>> {case_dir}/cerebralArteries/graph_simple.png

        Acts as a wrapper for the arterial.vessel_labelling.preprocessing.preprocessing 
            make_simple_centerline_graph() function.

        Parameters
        ----------

        Returns
        -------

        """
        build_simple_centerline_graph(self.case_dir)

    def predict_vessel_types(self):
        """
        Inputs the graph derived from the segmentsArray in edge form (edges as centerline segments), 
        performs the prediction by inference with the XGboost model and returns the original graph 
        in edge form with the predicted vessel types for each edge. Saves a pickle file with the 
        predicted graph at os.path.join(caseDir, "graph_predXGB.pickle").

        At the end of the execution, the following files should be generated:

        >>> {case_dir}/cerebralArteries/graph_predXGB.pickle
        >>> {case_dir}/cerebralArteries/graph_predXGB.png
        

        Parameters
        ----------
            case_dir : string or path-like object
                Path to case directory.     
    

        Returns
        -------

        """

        predictVesselTypes(self.case_dir)