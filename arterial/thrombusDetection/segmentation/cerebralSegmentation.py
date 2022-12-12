import os
import shutil

from time import time


def nnUNetInference(case_dir):
    ''' Performs inference of casePath (nifti, CTA) with the best performing model
    to output a binary mask in a nifti format. The resulting nifti will be placed in
    output_path.

    Arguments:
        - casePath <str>: path to nifti image (a CTA) that we want to segment.

    Returns:

    '''
    case_id = os.path.basename(case_dir)
    case_dir = os.path.join(case_dir, 'cerebralArteries')
    # Paths used by nnUNet
    input_path = os.path.join(case_dir, "input")
    output_path = os.path.join(case_dir, "segmentation")
    if not os.path.isdir(input_path): os.mkdir(input_path)
    if not os.path.isdir(output_path): os.mkdir(output_path)
    
    # The casePath will be placed in a newly created dir with the case_id as name
    if not os.path.isfile(os.path.join(input_path, case_id + "_0000.nii.gz")):
        # os.rename(os.path.join(case_dir, case_id + '.nii.gz'), os.path.join(input_path, case_id + "_0000.nii.gz")) # Do to move the file
        shutil.copyfile(os.path.join(case_dir, case_id + '.nii.gz'), os.path.join(input_path, case_id + "_0000.nii.gz")) # Do to copy the file

    # We have to get rid of potential spaces in paths for the terminal commands to work 
    input_path = input_path.replace("\\", "")
    input_path = input_path.replace(" ", "\ ")
    output_path = output_path.replace("\\", "")
    output_path = output_path.replace(" ", "\ ")

    start = time()
   
    # os.environ['RESULTS_FOLDER'] = '/home/marc/Desktop/Environment/arterial1/arterial/segmentation/models'
    os.environ['RESULTS_FOLDER'] = os.path.join(os.environ["arterial_dir"], 'thrombusDetection/segmentation/models')

    if not os.path.isfile(os.path.join(output_path, case_id + ".nii.gz")):
        os.system("nnUNet_predict -i " + input_path + " -o " + output_path + " -t Task555_Cerebral -m 3d_fullres -f 0")

    # Set paths back to normal
    input_path = input_path.replace("\\", "")
    output_path = output_path.replace("\\", "")

    # shutil.copyfile(os.path.join(input_path, case_id + "_0000.nii.gz"), os.path.join(case_dir, case_id + ".nii.gz"))
    shutil.copyfile(os.path.join(output_path, case_id + ".nii.gz"), os.path.join(case_dir, case_id + "_segmentation.nii.gz"))
    shutil.rmtree(input_path)
    shutil.rmtree(output_path)
    print("Inference took {timex} s".format(timex = (time() - start)))
    print("                                  ")

def nnUNetThrombusInference(case_dir):

    ''' Performs inference of casePath (nifti, CTA) with the best performing model
    to output a binary mask in a nifti format. The resulting nifti will be placed in
    output_path.

    Arguments:
        - casePath <str>: path to nifti image (a CTA and a CT) that we want to segment.

    Returns:

    '''
    case_id = os.path.basename(case_dir)
    case_dir = os.path.join(case_dir, 'cerebralArteries')

    case_dir = os.path.join(case_dir, 'thrombus')

    # Paths used by nnUNet
    input_path = os.path.join(case_dir, "input")
    output_path = os.path.join(case_dir, "segmentation")

    CT = os.path.join(case_dir, 'croppedCT.nii.gz')
    CTA = os.path.join(case_dir, 'croppedCTA.nii.gz')

    if not os.path.isdir(input_path): os.mkdir(input_path)
    if not os.path.isdir(output_path): os.mkdir(output_path)
    
    # The casePath will be placed in a newly created dir with the case_id as name
    if not os.path.isfile(os.path.join(input_path, case_id + "_0000.nii.gz")):
        shutil.copyfile(CT, os.path.join(input_path, case_id + "_0000.nii.gz"))
        shutil.copyfile(CTA, os.path.join(input_path, case_id + "_0001.nii.gz"))

    # We have to get rid of potential spaces in paths for the terminal commands to work 
    input_path = input_path.replace("\\", "")
    input_path = input_path.replace(" ", "\ ")
    output_path = output_path.replace("\\", "")
    output_path = output_path.replace(" ", "\ ")

    start = time()
   
    os.environ['RESULTS_FOLDER'] = os.path.join(os.environ["arterial_dir"], 'thrombusDetection/segmentation/models')

    if not os.path.isfile(os.path.join(output_path, case_id + ".nii.gz")):
        os.system("nnUNet_predict -i " + input_path + " -o " + output_path + " -t Task267_Thrombus -m 3d_fullres -f 0") 

    # Set paths back to normal
    input_path = input_path.replace("\\", "")
    output_path = output_path.replace("\\", "")

    # shutil.copyfile(os.path.join(input_path, case_id + "_0000.nii.gz"), os.path.join(case_dir, case_id + "_CTA.nii.gz"))
    shutil.copyfile(os.path.join(output_path, case_id + ".nii.gz"), os.path.join(case_dir, "thrombusPrediction.nii.gz"))

    shutil.rmtree(input_path)
    shutil.rmtree(output_path)

    print("Inference took {timex} s".format(timex = (time() - start)))
    print("                                  ")

def nnUNetAlternateThrombusInference(case_dir):

    ''' Performs inference of casePath (nifti, CTA) with the best performing model
    to output a binary mask in a nifti format. The resulting nifti will be placed in
    output_path.

    Arguments:
        - casePath <str>: path to nifti image (a CTA and a CT) that we want to segment.

    Returns:

    '''
    case_id = os.path.basename(case_dir)
    case_dir = os.path.join(case_dir, 'cerebralArteries')

    case_dir = os.path.join(case_dir, 'thrombus', 'alternative')

    # Paths used by nnUNet
    input_path = os.path.join(case_dir, "input")
    output_path = os.path.join(case_dir, "segmentation")

    CT = os.path.join(case_dir, 'croppedCT.nii.gz')
    CTA = os.path.join(case_dir, 'croppedCTA.nii.gz')

    if not os.path.isdir(input_path): os.mkdir(input_path)
    if not os.path.isdir(output_path): os.mkdir(output_path)
    
    # The casePath will be placed in a newly created dir with the case_id as name
    if not os.path.isfile(os.path.join(input_path, case_id + "_0000.nii.gz")):
        shutil.copyfile(CT, os.path.join(input_path, case_id + "_0000.nii.gz"))
        shutil.copyfile(CTA, os.path.join(input_path, case_id + "_0001.nii.gz"))

    # We have to get rid of potential spaces in paths for the terminal commands to work 
    input_path = input_path.replace("\\", "")
    input_path = input_path.replace(" ", "\ ")
    output_path = output_path.replace("\\", "")
    output_path = output_path.replace(" ", "\ ")

    start = time()
    
    os.environ['RESULTS_FOLDER'] = os.path.join(os.environ["arterial_dir"], 'thrombusDetection/segmentation/models')
    if not os.path.isfile(os.path.join(output_path, case_id + ".nii.gz")):
        os.system("nnUNet_predict -i " + input_path + " -o " + output_path + " -t Task267_Thrombus -m 3d_fullres -f 1") 

    # Set paths back to normal
    input_path = input_path.replace("\\", "")
    output_path = output_path.replace("\\", "")

    # shutil.copyfile(os.path.join(input_path, case_id + "_0000.nii.gz"), os.path.join(case_dir, case_id + "_CTA.nii.gz"))
    shutil.copyfile(os.path.join(output_path, case_id + ".nii.gz"), os.path.join(case_dir, "thrombusPrediction.nii.gz"))

    shutil.rmtree(input_path)
    shutil.rmtree(output_path)

    print("Inference took {timex} s".format(timex = (time() - start)))
    print("                                  ")

