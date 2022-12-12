import numpy as np
import nibabel as nib
import os
from scipy.ndimage import gaussian_laplace
from skimage.measure import regionprops, label

from time import time

def cerebral_cropping(case_dir):
    '''
        The cerebralCropper enables to generate a reduced bounding box from the original CTA and the mannually annotated nifti
        by using a Laplacian of Gaussian filter (scypy) representing the cerebral region. For the further processign this is 
        required to perform a more focused segmentation of the cerebral arteries. 

        Parameters
        ----------
        casePath :: string or path-like object
                Path to the CTA image

        Returns
        ----------
            The extracted bounding boxes are saved inside the directory in a folder called 'cerebralArteries'

    '''
    time0 = time()
    # Define path to CTA
    case_id = os.path.basename(case_dir)
    print()
    print('Cropping ' + case_id + 'CTA and NCCT')
    # Load CTA
    full_CTA = nib.load(os.path.join(case_dir, case_id + '.nii.gz'))
    CTA = full_CTA.get_fdata()

    # Label = nib.load(os.path.join(case_dir, case_id +'_label.nii.gz')).get_fdata()
    NCCT = nib.load(os.path.join(case_dir, case_id +'_NCCT.nii.gz')).get_fdata()
    TH = nib.load(os.path.join(case_dir, case_id +'_TH.nii.gz')).get_fdata()
    
    # Obtain affine matrix 
    aff = np.array(full_CTA.affine)

    # Apply laplacian-gaussian filter to half of the CTA image
    x0, y0, z0 = CTA.shape
    
    CTA2 = CTA[:, :, int(z0/2):]
    x1, y1, z1 = CTA2.shape
    
    laplacian = gaussian_laplace(CTA2, sigma = 0.0001, mode = "nearest")
    
    # Get cranium "binary" mask
    tolerance = 0.43 * np.ptp(laplacian)
    threshold = np.min(laplacian) + tolerance
    cranium_mask = np.where(laplacian<=threshold, np.max(CTA), 0)
    
    # Get largest connected ccomponent
    def getLargestCC(segmentation):
        labels = label(segmentation)
        assert( labels.max() != 0 ) # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        return largestCC  

    label_mask = getLargestCC(cranium_mask)
    x, y, z = np.nonzero(label_mask) 

    cropp_point = int(z0/2) + min(z)
    
    CTA_array = CTA[:,:,cropp_point:]
    NCCT_array = NCCT[:,:,cropp_point:]
    TH_array = np.flip(TH[:,:,cropp_point:], axis = 0)

    CTA = nib.Nifti1Image(CTA_array, aff)
    NCCT_LABEL = nib.Nifti1Image(NCCT_array, aff)
    TH_LABEL = nib.Nifti1Image(TH_array, aff)

    if not os.path.exists(os.path.join(case_dir, 'cerebralArteries')): os.mkdir(os.path.join(case_dir, 'cerebralArteries'))

    # nib.save(CTA, os.path.join(case_dir, 'cerebralArteries', case_id + '.nii.gz'))
    # nib.save(NCCT_LABEL, os.path.join(case_dir, 'cerebralArteries', case_id + '_NCCT.nii.gz'))
    nib.save(TH_LABEL, os.path.join(case_dir, 'cerebralArteries', case_id + '_TH.nii.gz'))
    print('Done in {}'.format(np.round(time()-time0, 2)))
    print()
    

    # parser = argparse.ArgumentParser()

    # parser.add_argument('-casePath', '--casePath', type=str, required=True, 
    #     help='path binary nifti to be processed. Required.')

    # args = parser.parse_args()

    # casePath = args.casePath

    # caseDir = os.path.abspath(os.path.dirname(casePath))
    # caseId = f"{casePath[-15:-7]}"

    # full_CTA = nib.load(casePath)
    # affine = full_CTA.affine
    # full_CTA = full_CTA.get_fdata()
    # label = nib.load(casePath[:-7]+'_label.nii.gz').get_fdata()

    # # Obtain affine matrix 
    # aff = np.array(nib.load(casePath).affine)
    # # Load volume and associate to node
    # slicer.util.loadLabelVolume(casePath)
    # masterVolumeNode = getNode(os.path.basename(casePath[:-7])) # caseId is the file name without the extension (.nii.gz

    # MOD = slicer.modules.swissskullstripper
    # # Set parameters
    # parameters = {}
    # parameters["patientVolume"] = masterVolumeNode
    # outputModelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
    # parameters["patientOutputVolume"] = outputModelNode
    # outputModelNode1 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    # parameters['patientMaskLabel'] = outputModelNode1

    # cliNode = slicer.cli.runSync(MOD, None, parameters)

    # Volumearray = slicer.util.arrayFromVolume(outputModelNode1)

    # x, y, z = np.nonzero(Volumearray) 
    # cropp_point = min(x)

    # # CTA_array = slicer.util.arrayFromVolume(masterVolumeNode)
    # # CTA_array = np.swapaxes(CTA_array[cropp_point:,:,:], 0, 2)
    # CTA_array = full_CTA[:,:,cropp_point:]
    # label_array = label[:,:,cropp_point:]
    # CTA = nib.Nifti1Image(CTA_array, affine)
    # CTA_LABEL = nib.Nifti1Image(label_array, affine)
    # if not os.path.exists(os.path.join(caseDir, 'cerebralArteries')): os.mkdir(os.path.join(caseDir, 'cerebralArteries'))
    # nib.save(CTA, os.path.join(caseDir, 'cerebralArteries', caseId + '.nii.gz'))
    # nib.save(CTA_LABEL, os.path.join(caseDir, 'cerebralArteries', caseId + '_label.nii.gz'))

