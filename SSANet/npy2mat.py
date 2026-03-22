import numpy as np
from scipy.io import savemat

filename = "/data/code/ds_code/SSANet/Rec/datasets-hys1_data_45.mat.npy"

data = np.load(filename)

savemat('/data/code/ds_code/SSANet/Rec/mat1/datasets-hys1_data_45.mat', {'Xim': data})