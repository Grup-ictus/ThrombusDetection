#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os
import nibabel as nib

def perform_preprocessing_and_centerline_extraction_thrombus(case_dir, no_display):
    """
    Wrapper function to make terminal command calls for centerline preprocessing and extraction,
    using Slicer [1] and VMTK [2-4]. Since PythonSlicer is needed, we need to call terminal commands to execute
    the corresponding script. When this script is executed as __main__ (upon terminal command call),
    it performs preprocessing and terminal extraction. This function basically calls itself from the 
    terminal line using Slicer's Python interpreter.

    Assumes that enviroment variables slicer_path and arterial_dir are correctly set according to your
    own system. This should be set in the ~/.bashrc (or ~/.zshrc) and should look something like:

    >>> export slicer_path="/path/to/Slicer"
    >>> arterial_dir="/path/to/arterial/arterial"

    References:
    [1]     Fedorov, Andriy, Reinhard Beichel, Jayashree Kalpathy-Cramer, Julien Finet, Jean-Christophe Fillion-Robin, 
    Sonia Pujol, Christian Bauer, et al. 2012. "3D Slicer as an Image Computing Platform for the Quantitative Imaging 
    Network." Magnetic Resonance Imaging 30 (9): 1323-41. https://doi.org/10.1016/j.mri.2012.05.001.
    [2]     Antiga, Luca, Bogdan Ene-Iordache, and Andrea Remuzzi. 2003. "Centerline Computation and Geometric Analysis 
    of Branching Tubular Surfaces with Application to Blood Vessel Modeling." Wscg. 
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.14.671&rep=rep1&type=pdf.
    [3]     Antiga, Luca, Marina Piccinelli, Lorenzo Botti, Bogdan Ene-Iordache, Andrea Remuzzi, and David A. Steinman. 
    2008. "An Image-Based Modeling Framework for Patient-Specific Computational Hemodynamics." Medical and Biological 
    Engineering and Computing 46 (11): 1097-1112. https://doi.org/10.1007/s11517-008-0420-1.
    [4]     Antiga, Luca, and David A. Steinman. 2004. "Robust and Objective Decomposition and Mapping of Bifurcating 
    Vessels." IEEE Transactions on Medical Imaging 23 (6): 704-13. https://doi.org/10.1109/TMI.2004.826946.

    Parameters
    ----------
    case_dir : string or path-like object
        Path to case directory. 
    no_display : bool, default = False
        Boolean variable to be used when running analysis on a headless server.
        In addition, add ```$xvfb-run --auto-servernum --server-num=1``` at the beggining
        of the command line call when executing the script from the command line.
        E.g.: ```$xvfb-run --auto-servernum --server-num=1 python perform_analysis.py -case_dir {case_dir} -no_display {True}```

    Return
    ------
    
    """
    # Define paths from enviornment variables. These should be set prior to execution in ~/.bashrc or equivalent
    SLICER_PATH = os.environ["slicer_path"]
    RUN_CENTERLINE_EXTRACTION_SCRIPT = os.path.join(os.environ["arterial_dir"], "thrombusDetection/centerline_extraction/run_centerline_extraction_slicer_thrombus.py")
    # Perform segmentation and centerline extraction. This generates decimatedSegmentations and centerlines in caseDir
    if no_display: # Use if remote server is used, in combination with xvfb-run --auto-servernum --server-num=1  --disable-terminal-outputs
        os.system("{}  --python-script {} -case_dir {} --exit-after-startup".format(SLICER_PATH, RUN_CENTERLINE_EXTRACTION_SCRIPT, case_dir))
    else:
        os.system("{} --no-main-window --no-splash  --python-script {} -case_dir {} --exit-after-startup".format(SLICER_PATH, RUN_CENTERLINE_EXTRACTION_SCRIPT, case_dir))

if __name__ ==  "__main__":
    # Script to be executed by PythonSlicer interpreter
    # Will only be executed when this script is called from a direct terminal command
    import slicer

    import argparse

    from preprocessing.preprocessing_thrombus import preprocessing_thrombus
    from centerline_extraction_thrombus import centerline_extraction_thrombus

    parser = argparse.ArgumentParser()

    parser.add_argument('-case_dir', '--case_dir', type=str, required=True, 
        help='path binary nifti to be processed. Required.')

    args = parser.parse_args()

    case_dir = args.case_dir
    
    case_id = os.path.basename(case_dir)
    
    nifti = nib.load(os.path.join(case_dir, 'cerebralArteries', 'thrombus', "thrombusPrediction.nii.gz")).get_fdata()

    if sum(nifti.flatten())>0:
        # Load volume and associate to node
        slicer.util.loadLabelVolume(os.path.join(case_dir,'cerebralArteries', 'thrombus', "thrombusPrediction.nii.gz"))        
        master_volume_node = getNode("thrombusPrediction")

        # Perform segmentation from binary mask
        segmentation_node, masked_volume_array = preprocessing_thrombus(master_volume_node)
        # Perform centerline extraction. Creates centerlines.vtk
        centerline_extraction_thrombus(case_dir, segmentation_node, masked_volume_array)