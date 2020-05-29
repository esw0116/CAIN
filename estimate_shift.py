import os
import glob
import imageio
import numpy as np
import pandas as pd
from skimage import feature


data_root = 'data/dataset/Project_Samsung'

train_folder = os.path.join(data_root, 'elegance', 'train/LR_bicubic/X2_denoise')
val_folder = os.path.join(data_root, 'elegance', 'val/LR_bicubic/X2_denoise')


file_list = []
subfolders = sorted(glob.glob(train_folder + '/*'))
print(subfolders)
for subfolder in subfolders:
    print(subfolder)
    flist = sorted(glob.glob(subfolder + '/*'))
    df = pd.DataFrame(columns=['Y', 'X', 'Total_Y', 'Total_X'])
    total_shift = np.array([0, 0])
    max_shift = np.array([0, 0])
    min_shift = np.array([0, 0])
    for i in range(len(flist)-1):
        if i == 0:
            src_img = imageio.imread(flist[i])
            trg_img = imageio.imread(flist[i+1])
        else:
            src_img = trg_img
            trg_img = imageio.imread(flist[i+1])

        shift, _, _ = feature.register_translation(src_img, trg_img)
        total_shift[0] += shift[0]
        total_shift[1] += shift[1]

        df.loc[i+1] = [shift[0], shift[1], total_shift[0], total_shift[1]]

        if max_shift[0] < total_shift[0]:
            max_shift[0] = total_shift[0]
        if max_shift[1] < total_shift[1]:
            max_shift[1] = total_shift[1]
        if min_shift[0] > total_shift[0]:
            min_shift[0] = total_shift[0]
        if min_shift[1] > total_shift[1]:
            min_shift[1] = total_shift[1]

    df.loc['Max'] = [0, 0, max_shift[0], max_shift[1]]
    df.loc['Min'] = [0, 0, min_shift[0], min_shift[1]]

    df.to_csv(os.path.join(subfolder, 'shift.csv'), sep=',')
