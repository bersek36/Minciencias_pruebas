import numpy as np 
np.set_printoptions(precision=2, suppress=True)

import os
import nibabel as nib
from nibabel.testing import data_path

example_ni1 = os.path.join(data_path, "example_nifti2.nii.gz")
n1_img = nib.load(example_ni1)
n1_img