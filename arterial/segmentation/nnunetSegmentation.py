#    Copyright 2021 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import shutil

from time import time


def nnUNetInference(casePath):
    ''' Performs inference of casePath (nifti, CTA) with the best performing model
    to output a binary mask in a nifti format. The resulting nifti will be placed in
    outputPath.

    Arguments:
        - casePath <str>: path to nifti image (a CTA) that we want to segment.

    Returns:

    '''

    caseId = os.path.basename(casePath)[:-7] # Name of the nifti casePath except the .nii.gz extension
    caseDir = os.path.dirname(casePath)
    print(caseId, caseDir)

    # Paths used by nnUNet
    inputPath = os.path.join(caseDir, "input")
    outputPath = os.path.join(caseDir, "segmentation")
    if not os.path.isdir(inputPath): os.mkdir(inputPath)
    if not os.path.isdir(outputPath): os.mkdir(outputPath)
    
    # The casePath will be placed in a newly created dir with the caseId as name
    if not os.path.isfile(os.path.join(inputPath, caseId + "_0000.nii.gz")):
        os.rename(casePath, os.path.join(inputPath, caseId + "_0000.nii.gz"))

    # We have to get rid of potential spaces in paths for the terminal commands to work 
    inputPath = inputPath.replace("\\", "")
    inputPath = inputPath.replace(" ", "\ ")
    outputPath = outputPath.replace("\\", "")
    outputPath = outputPath.replace(" ", "\ ")

    start = time()
    os.environ['RESULTS_FOLDER'] = '/home/marc/Desktop/Environment/arterial1/arterial/segmentation/models'
    if not os.path.isfile(os.path.join(outputPath, caseId + ".nii.gz")):
        os.system("nnUNet_predict -i " + inputPath + " -o " + outputPath + " -t Task555_Cerebral -m 3d_fullres -f 0")

    # Set paths back to normal
    inputPath = inputPath.replace("\\", "")
    outputPath = outputPath.replace("\\", "")

    shutil.copyfile(os.path.join(inputPath, caseId + "_0000.nii.gz"), os.path.join(caseDir, caseId + "_CTA.nii.gz"))
    shutil.copyfile(os.path.join(outputPath, caseId + ".nii.gz"), os.path.join(caseDir, caseId + ".nii.gz"))

    print("Inference took {timex} s".format(timex = (time() - start)))
    print("                                  ")


def nnUNetThrombusInference(casePath):
    ''' Performs inference of casePath (nifti, CTA) with the best performing model
    to output a binary mask in a nifti format. The resulting nifti will be placed in
    outputPath.

    Arguments:
        - casePath <str>: path to nifti image (a CTA) that we want to segment.

    Returns:

    '''

    caseId = os.path.basename(casePath)[:-7] # Name of the nifti casePath except the .nii.gz extension
    caseDir = os.path.join(os.path.dirname(casePath), 'thrombus', 'cropped')
    print(caseId, caseDir)

    # Paths used by nnUNet
    inputPath = os.path.join(caseDir, "input")
    outputPath = os.path.join(caseDir, "segmentation")

    CT = os.path.join(caseDir, 'croppedCT.nii.gz')
    CTA = os.path.join(caseDir, 'croppedCTA.nii.gz')

    if not os.path.isdir(inputPath): os.mkdir(inputPath)
    if not os.path.isdir(outputPath): os.mkdir(outputPath)
    
    # The casePath will be placed in a newly created dir with the caseId as name
    if not os.path.isfile(os.path.join(inputPath, caseId + "_0000.nii.gz")):
        shutil.copyfile(CT, os.path.join(inputPath, caseId + "_0000.nii.gz"))
        shutil.copyfile(CTA, os.path.join(inputPath, caseId + "_0001.nii.gz"))

    # We have to get rid of potential spaces in paths for the terminal commands to work 
    inputPath = inputPath.replace("\\", "")
    inputPath = inputPath.replace(" ", "\ ")
    outputPath = outputPath.replace("\\", "")
    outputPath = outputPath.replace(" ", "\ ")

    start = time()
    os.environ['RESULTS_FOLDER'] = '/home/marc/Desktop/Environment/arterial1/arterial/segmentation/models'
    if not os.path.isfile(os.path.join(outputPath, caseId + ".nii.gz")):
        os.system("nnUNet_predict -i " + inputPath + " -o " + outputPath + " -t Task267_Thrombus -m 3d_fullres -f 1") 

    # Set paths back to normal
    inputPath = inputPath.replace("\\", "")
    outputPath = outputPath.replace("\\", "")

    # shutil.copyfile(os.path.join(inputPath, caseId + "_0000.nii.gz"), os.path.join(caseDir, caseId + "_CTA.nii.gz"))
    shutil.copyfile(os.path.join(outputPath, caseId + ".nii.gz"), os.path.join(caseDir, "thrombusPrediction.nii.gz"))

    print("Inference took {timex} s".format(timex = (time() - start)))
    print("                                  ")


def nnUNetEnsemble(casePath):
    ''' Performs inference of casePath (nifti, CTA) by ensembling all folds of 
    the best performing model to output a binary mask in a nifti format. The resulting 
    nifti will be placed in outputDir.

    Arguments:
        - casePath <str>: path to nifti image (a CTA) that we want to segment.

    Returns:

    '''

    caseId = os.path.basename(casePath)[:-7]
    caseDir = os.path.dirname(casePath)

    inputPath = os.path.join(caseDir, "input")
    if not os.path.isdir(inputPath): os.mkdir(inputPath)
    if not os.path.isdir(os.path.join(caseDir, "ensemble")): os.mkdir(os.path.join(caseDir, "ensemble"))

    if not os.path.isfile(os.path.join(inputPath, caseId + "_0000.nii.gz")):
        os.rename(casePath, os.path.join(inputPath, caseId + "_0000.nii.gz"))

    # We have to get rid of potential spaces in paths for the terminal commands to work 
    inputPath = inputPath.replace("\\", "")
    inputPath = inputPath.replace(" ", "\ ")

    npzDirs = []

    start = time()

    for fold in range(5):
        print("Predicting {} fold {}".format(caseId, fold))
        print("                               ")

        outputPath = os.path.join(caseDir, "ensemble", "fold_{}".format(fold))
        if not os.path.isdir(outputPath): os.mkdir(outputPath)

        outputPathAux = outputPath.replace("\\", "")
        outputPathAux = outputPath.replace(" ", "\ ")
            
        # We have to get rid of potential spaces in paths for the terminal commands to work 
        os.system("nnUNet_predict -i " + inputPath + " -o " + outputPathAux + " -t Task001_Arterial -z -m 3d_lowres -f " + str(fold))

        npzDirs.append(outputPath)

        print("Fold {} completed".format(fold))
        print("                      ")

    outputDir = os.path.join(caseDir, "ensemble", "output")
    if not os.path.isdir(outputDir): os.mkdir(outputDir)

    outputDir = outputDir.replace("\\", "")

    print("Starting ensembling")

    os.system("nnUNet_ensemble -f {} {} {} {} {} -o {}".format(npzDirs[0], npzDirs[1], npzDirs[2], npzDirs[3], npzDirs[4], outputDir))

    # Set paths back to normal
    inputPath = inputPath.replace("\\", "")
    outputDir = outputDir.replace("\\", "")

    shutil.copyfile(os.path.join(inputPath, caseId + "_0000.nii.gz"), os.path.join(caseDir, caseId + "_CTA.nii.gz"))
    shutil.copyfile(os.path.join(outputDir, caseId + ".nii.gz"), os.path.join(caseDir, caseId + ".nii.gz"))

    print("Ensembling took {tiemx} s".format(timex = (time() - start)))
    print("                                   ")