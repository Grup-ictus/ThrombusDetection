#   Copyright 2022 Stroke Research at Vall d"Hebron Research Institute (VHIR), Barcelona, Spain.

 
import argparse

from arterial.run.test import ArterialProcessor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-case_dir", "--case_dir", type=str, required=True,
        help="Path to directory containing the nifti image (assumes that the nifti file has the basename of the dir). Required.")
    parser.add_argument("-no_display", "--no_display", type=bool, required=False, default=False, 
        help="If using a remote Linux, this should be used following correct Slicer installation, and should be coulpled "
        "with the use of `xvfb-run --auto-servernum --server-num=1` upon use before calling this script (prior to the python command). "
        "Not required, default = False.")
    parser.add_argument("-ss", "--skip_segmentation", type=bool, required=False, default=False, 
        help="Boolean argument to determine if segmentation is predicted or not. Use (True) if segmentation is already "
        "predicted, to skip nnUNet inference and save time. If True, there should exist a nifti file with the binary map "
        "with the following naming convention: {os.path.basename}_segmentation.nii.gz. Not required, default = False.")

    # skip_centerline_extraction
    # skip_circular_centerlines
    # skip_branching
    # skip_clipping
    # skip_vessel_labelling
    
    parser = parser.parse_args()

    processor = ArterialProcessor(parser)
    processor.perform_analysis()

if __name__ == "__main__":
    main()