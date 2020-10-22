import glob
import os
import imageio
import numpy as np
import pandas as pd

#train_folder = os.path.join('data/dataset/Project_Samsung', 'elegance', 'train/LR_bicubic/X2_denoise')
#val_folder = os.path.join('data/dataset/Project_Samsung', 'elegance', 'val/LR_bicubic/X2_denoise')

#train_savefolder = os.path.join('data/dataset/Project_Samsung', 'elegance', 'train/LR_bicubic/X2_denoise_crop')
#val_savefolder = os.path.join('data/dataset/Project_Samsung', 'elegance', 'val/LR_bicubic/X2_denoise_crop')

train_folder = os.path.join('data/dataset/Project_Samsung', 'elegance', 'train/LR_bicubic/X2')
val_folder = os.path.join('data/dataset/Project_Samsung', 'elegance', 'val/LR_bicubic/X2')

train_savefolder = os.path.join('data/dataset/Project_Samsung', 'elegance', 'train/LR_bicubic/X2_crop')
val_savefolder = os.path.join('data/dataset/Project_Samsung', 'elegance', 'val/LR_bicubic/X2_crop')

if not os.path.exists(train_savefolder):
    os.mkdir(train_savefolder)

if not os.path.exists(val_savefolder):
    os.mkdir(val_savefolder)


for folder, savefolder in zip([train_folder, val_folder], [train_savefolder, val_savefolder]):
    subfolders = sorted(glob.glob(folder + '/*'))
    for subfolder in subfolders:
        print(subfolder)
        basename = os.path.basename(subfolder)
        save_subfolder = os.path.join(savefolder, basename)
        if not os.path.exists(save_subfolder):
            os.mkdir(save_subfolder)
        subfolder_file_list = sorted(glob.glob(subfolder + '/*.tif'))
        shift_csv = subfolder + '/shift.csv'
        shift_df = pd.read_csv(shift_csv, index_col=0)
        max_y, min_y = shift_df.loc['Max']['Total_Y'], -shift_df.loc['Min']['Total_Y']
        max_x, min_x = shift_df.loc['Max']['Total_X'], -shift_df.loc['Min']['Total_X']

        for i in range(len(subfolder_file_list)):
            if i == 0:
                crop_edge = [int(max_y), int(min_y), int(max_x), int(min_x)]
            else:
                cur_y, cur_x = shift_df.iloc[i-1]['Total_Y'], shift_df.iloc[i-1]['Total_X']
                crop_edge = [int(max_y-cur_y), int(min_y+cur_y), int(max_x-cur_x), int(min_x+cur_x)]

            raw_image = imageio.imread(subfolder_file_list[i])
            h, w = raw_image.shape
            cropped_image = raw_image[crop_edge[0]:h-crop_edge[1], crop_edge[2]:w-crop_edge[3]]
            #print(crop_edge)
            #print(raw_image.shape, cropped_image.shape)

            img_name = os.path.basename(subfolder_file_list[i])
            imageio.imwrite(os.path.join(save_subfolder, img_name), cropped_image)
