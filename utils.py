import os
from multiprocessing import Pool
from timeit import default_timer as timer

import numpy as np 
import nibabel as nib
import tensorflow as tf

from preprocess.get_subvolume import get_training_sub_volumes

def make_dirs():
    processed_paths = {}
    processed_sample_paths = {}
    PARENT_DIR = os.getcwd()
    DATABASE_DIR = os.path.join(PARENT_DIR, "NFBS_Dataset")
    SAMPLES_FOLDERS = next(os.walk(DATABASE_DIR))[1]
    
    processed_paths["PROCESSED_DIR"] = os.path.join(PARENT_DIR, "processed")
    PROCESSED_DIR = processed_paths["PROCESSED_DIR"]
    processed_paths["SUBVOLUME_FOLDER"] = os.path.join(PROCESSED_DIR,"subvolumes")
    processed_paths["SUBVOLUME_MASK_FOLDER"] = os.path.join(PROCESSED_DIR,"subvolumes_masks")

    for path in processed_paths:
        dir_exists = os.path.exists(processed_paths[path])
        if not dir_exists:
            os.makedirs(processed_paths[path])

    processed_paths.pop("PROCESSED_DIR")
    
    for path in processed_paths:
        for sample in SAMPLES_FOLDERS:
            sample_dir = os.path.join(processed_paths[path], sample)
            dir_exists = os.path.exists(sample_dir)
            if not dir_exists:
                os.makedirs(sample_dir)
        processed_sample_paths[path] = processed_paths[path]
    processed_sample_paths["DATABASE_DIR"] = DATABASE_DIR
    processed_sample_paths["SAMPLES"] = SAMPLES_FOLDERS
    return processed_sample_paths

paths = make_dirs()

def create_file_name(sample, mask=False):
    if mask:
        file_name = "sub-"+sample+"_ses-NFB3_T1w_brainmask.nii.gz"
    else:
        file_name = "sub-"+sample+"_ses-NFB3_T1w.nii.gz"
    file_name = os.path.join(sample, file_name)
    return file_name

def get_sub_volumes(sample):
        img = os.path.join(paths["DATABASE_DIR"], create_file_name(sample, False))
        img_mask = os.path.join(paths["DATABASE_DIR"], create_file_name(sample, True))
        img = nib.load(img)
        img_mask = nib.load(img_mask)
        image = img.get_fdata()
        image_mask = img_mask.get_fdata()
        SAVE_PATH_SUBVOLUME = os.path.join(paths["SUBVOLUME_FOLDER"], sample)
        SAVE_PATH_SUBMASK = os.path.join(paths["SUBVOLUME_MASK_FOLDER"], sample)
        get_training_sub_volumes(image, img.affine, image_mask, img_mask.affine, 
                                        SAVE_PATH_SUBVOLUME, SAVE_PATH_SUBMASK, 
                                        classes=1, 
                                        orig_x = 256, orig_y = 256, orig_z = 192, 
                                        output_x = 128, output_y = 128, output_z = 16,
                                        stride_x = 32, stride_y = 32, stride_z = 8,
                                        background_threshold=0.1)


if __name__ == '__main__':
    start = timer()
    with Pool(6) as pool:
        pool.map(get_sub_volumes, paths["SAMPLES"])

    end = timer()
    
    print("Elapsed time {}".format(end-start))