import os
import numpy as np
import nibabel as nib
from pathlib import Path

def make_folders(feat_dir, mask_dir, item):
    FEATURES_SAVE_PATH = os.path.join(feat_dir, item)
    MASK_GM_SAVE_PATH = os.path.join(mask_dir, 'GM', item)
    MASK_WM_SAVE_PATH = os.path.join(mask_dir, 'WM', item)
    MASK_CSF_SAVE_PATH = os.path.join(mask_dir, 'CSF', item)
    Path(FEATURES_SAVE_PATH).mkdir(parents=True, exist_ok=True)
    Path(MASK_GM_SAVE_PATH).mkdir(parents=True, exist_ok=True)
    Path(MASK_WM_SAVE_PATH).mkdir(parents=True, exist_ok=True)
    Path(MASK_CSF_SAVE_PATH).mkdir(parents=True, exist_ok=True)

def get_training_sub_volumes(image, image_affine, label, label_affine, 
                   subvol_dest_folder, submask_dest_folder, classes=1,
                   orig_x = 240, orig_y = 240, orig_z = 48, 
                   output_x = 80, output_y = 80, output_z = 16,
                   stride_x = 40, stride_y = 40, stride_z = 8,
                   background_threshold=0.1):

    X = None
    y = None
    
    tries = 0

    for i in range(0, orig_z-output_z+1, stride_z):#(0, orig_x-output_x+1, output_x-1):
        for j in range(0, orig_y-output_y+1, stride_y):
            for k in range(0, orig_x-output_x+1, stride_x):

                # extract relevant area of label
                y = label[k: k + output_x,
                          j: j + output_y,
                          i: i + output_z]
                
                volume = output_x*output_y*output_z

                # compute the background ratio 
                bgrd_ratio = np.sum(y==classes)/volume

                # if background ratio is above the threshold, use that sub-volume. otherwise continue and try another sub-volume
                if bgrd_ratio > background_threshold: # Lo que se quiere segmentar se al menos threshold%
                    #print(bgrd_ratio)
                    tries += 1
                    X = image[k: k + output_x,
                          j: j + output_y,
                          i: i + output_z]

                    nii_subvolume = nib.Nifti1Image(X.astype(np.int16), image_affine)
                    nib.save(nii_subvolume, os.path.join(subvol_dest_folder, 'subvolume'+str(tries)+'.nii'))
                    nii_submask = nib.Nifti1Image(y.astype(np.int16), label_affine)
                    nib.save(nii_submask, os.path.join(submask_dest_folder, 'submask'+str(tries)+'.nii'))

def get_test_subvolumes(image, image_affine, label, label_affine, 
                        subvol_dest_folder, submask_dest_folder, 
                        orig_x = 240, orig_y = 240, orig_z = 48, 
                        output_x = 80, output_y = 80, output_z = 16,
                        stride_x = 80, stride_y = 80, stride_z = 16):
    
    X = None
    y = None
    
    tries = 0

    for i in range(0, orig_z-output_z+1, stride_z):#(0, orig_x-output_x+1, output_x-1):
        for j in range(0, orig_y-output_y+1, stride_y):
            for k in range(0, orig_x-output_x+1, stride_x):
                tries += 1

                # extract relevant area of label
                y = label[k: k + output_x,
                          j: j + output_y,
                          i: i + output_z]

                X = image[k: k + output_x,
                      j: j + output_y,
                      i: i + output_z]

                nii_subvolume = nib.Nifti1Image(X, image_affine)
                nib.save(nii_subvolume, os.path.join(subvol_dest_folder, 'subvolume'+str(tries)+'.nii'))
                nii_submask = nib.Nifti1Image(y, label_affine)
                nib.save(nii_submask, os.path.join(submask_dest_folder, 'submask'+str(tries)+'.nii'))

                  
def get_all_training_sub_volumes(image, label_GM, label_WM, label_CSF, affine, 
                   subvol_dest_folder, submask_gm_folder, #classes=1,
                   submask_wm_folder, submask_csf_folder,
                   orig_x = 240, orig_y = 240, orig_z = 48, 
                   output_x = 80, output_y = 80, output_z = 16,
                   stride_x = 40, stride_y = 40, stride_z = 8,
                   background_threshold=0.1):

    X = None
    y = None
    
    tries = 0

    for i in range(0, orig_z-output_z+1, stride_z):#(0, orig_x-output_x+1, output_x-1):
        for j in range(0, orig_y-output_y+1, stride_y):
            for k in range(0, orig_x-output_x+1, stride_x):

                # extract relevant area of label
                X = image[k: k + output_x,
                          j: j + output_y,
                          i: i + output_z]
                
                volume = output_x*output_y*output_z

                # compute the background ratio
                bgrd_ratio = np.sum(np.where(X>0, 1, 0))/volume 
                #bgrd_ratio = np.sum(y)/volume #np.sum(y==classes)/volume

                # if background ratio is above the threshold, use that sub-volume. otherwise continue and try another sub-volume
                if bgrd_ratio > background_threshold: # Lo que se quiere segmentar se al menos threshold%
                    #print(bgrd_ratio)
                    tries += 1

                    y_GM = label_GM[k: k + output_x,
                                    j: j + output_y,
                                    i: i + output_z]

                    y_WM = label_WM[k: k + output_x,
                                    j: j + output_y,
                                    i: i + output_z]

                    y_CSF = label_CSF[k: k + output_x,
                                      j: j + output_y,
                                      i: i + output_z]

                    nii_subvolume = nib.Nifti1Image(X.astype(np.int16), affine)
                    nib.save(nii_subvolume, os.path.join(subvol_dest_folder, 'subvolume'+str(tries)+'.nii'))

                    nii_submask_gm = nib.Nifti1Image(y_GM.astype(np.int16), affine)
                    nib.save(nii_submask_gm, os.path.join(submask_gm_folder, 'submask'+str(tries)+'.nii'))

                    nii_submask_wm = nib.Nifti1Image(y_WM.astype(np.int16), affine)
                    nib.save(nii_submask_wm, os.path.join(submask_wm_folder, 'submask'+str(tries)+'.nii'))

                    nii_submask_csf = nib.Nifti1Image(y_CSF.astype(np.int16), affine)
                    nib.save(nii_submask_csf, os.path.join(submask_csf_folder, 'submask'+str(tries)+'.nii'))

def get_all_test_subvolumes(image, label_GM, label_WM, label_CSF, affine, 
                        subvol_folder, submask_gm_folder,
                        submask_wm_folder, submask_csf_folder, 
                        orig_x = 240, orig_y = 240, orig_z = 48, 
                        output_x = 80, output_y = 80, output_z = 16,
                        stride_x = 80, stride_y = 80, stride_z = 16):
    
    X = None
    y = None
    
    tries = 0

    for i in range(0, orig_z-output_z+1, stride_z):#(0, orig_x-output_x+1, output_x-1):
        for j in range(0, orig_y-output_y+1, stride_y):
            for k in range(0, orig_x-output_x+1, stride_x):
                tries += 1

                # extract relevant area of label

                y_GM = label_GM[k: k + output_x,
                                j: j + output_y,
                                i: i + output_z]

                y_WM = label_WM[k: k + output_x,
                                j: j + output_y,
                                i: i + output_z]

                y_CSF = label_CSF[k: k + output_x,
                                  j: j + output_y,
                                  i: i + output_z]

                X = image[k: k + output_x,
                          j: j + output_y,
                          i: i + output_z]

                nii_subvolume = nib.Nifti1Image(X.astype(np.int16), affine)
                nib.save(nii_subvolume, os.path.join(subvol_folder, 'subvolume'+str(tries)+'.nii'))

                nii_submask_gm = nib.Nifti1Image(y_GM.astype(np.int16), affine)
                nib.save(nii_submask_gm, os.path.join(submask_gm_folder, 'submask'+str(tries)+'.nii'))

                nii_submask_wm = nib.Nifti1Image(y_WM.astype(np.int16), affine)
                nib.save(nii_submask_wm, os.path.join(submask_wm_folder, 'submask'+str(tries)+'.nii'))

                nii_submask_csf = nib.Nifti1Image(y_CSF.astype(np.int16), affine)
                nib.save(nii_submask_csf, os.path.join(submask_csf_folder, 'submask'+str(tries)+'.nii'))
