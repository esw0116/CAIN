import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random

class Eleganceset(Dataset):
    def __init__(self, data_root, is_training, denoise):
        self.data_root = data_root
        self.training = is_training

        if denoise:
            train_folder = os.path.join(self.data_root, 'elegance', 'train/LR_bicubic/X2_denoise')
            val_folder = os.path.join(self.data_root, 'elegance', 'val/LR_bicubic/X2_denoise')
        else:
            train_folder = os.path.join(self.data_root, 'elegance', 'train/LR_bicubic/X2')
            val_folder = os.path.join(self.data_root, 'elegance', 'val/LR_bicubic/X2')

        self.file_list = []
        if self.training:
            subfolders = sorted(glob.glob(train_folder + '/*'))
            for subfolder in subfolders:
                subfolder_file_list = sorted(glob.glob(subfolder + '/*'))
                subfolder_file_list = subfolder_file_list[2:-2]
                self.file_list.extend(subfolder_file_list)

        else:
            subfolders = sorted(glob.glob(val_folder + '/*'))
            for subfolder in subfolders:
                subfolder_file_list = sorted(glob.glob(subfolder + '/*'))
                subfolder_file_list = subfolder_file_list[:-2]
                subfolder_file_list = subfolder_file_list[2::4]
                self.file_list.extend(subfolder_file_list)
        
        self.transforms = transforms.Compose([
            transforms.RandomCrop(1024),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
            transforms.ToTensor()
        ])
        

    def __getitem__(self, index):
        '''
        if self.training:
            imgpath = os.path.join(self.image_root, self.trainlist[index])
        else:
            imgpath = os.path.join(self.image_root, self.testlist[index])
        imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png']
        '''
        current_img_path = self.file_list[index]

        current_idx_string = current_img_path[-7:-4]
        current_idx = int(current_idx_string)

        pp_idx_string = '{:03d}'.format(current_idx - 2)
        pp_img_path = current_img_path.replace(current_idx_string, pp_idx_string)

        past_idx_string = '{:03d}'.format(current_idx - 1)
        past_img_path = current_img_path.replace(current_idx_string, past_idx_string)

        future_idx_string = '{:03d}'.format(current_idx + 1)
        future_img_path = current_img_path.replace(current_idx_string, future_idx_string)

        ff_idx_string = '{:03d}'.format(current_idx + 2)
        ff_img_path = current_img_path.replace(current_idx_string, ff_idx_string)

        if self.training:
            seed = random.random()
            # if seed < 0.5: +-1 feature, else +-2 feature.
            if seed < 0.5:
                imgpaths = [past_img_path, current_img_path, future_img_path]
            else:
                imgpaths = [pp_img_path, current_img_path, ff_img_path]

            # Load images
            img1 = Image.open(imgpaths[0])
            img2 = Image.open(imgpaths[1])
            img3 = Image.open(imgpaths[2])
            # Data augmentation
            seed = random.randint(0, 2 ** 32)
            random.seed(seed)
            img1 = self.transforms(img1)
            random.seed(seed)
            img2 = self.transforms(img2)
            random.seed(seed)
            img3 = self.transforms(img3)
            # Random Temporal Flip
            if random.random() >= 0.5:
                img1, img3 = img3, img1
                imgpaths[0], imgpaths[2] = imgpaths[2], imgpaths[0]
            imgs = [img1, img2, img3]

        else:
            imgpaths = [pp_img_path, past_img_path, current_img_path, future_img_path, ff_img_path]

            # Load images
            img1 = Image.open(imgpaths[0])
            img2 = Image.open(imgpaths[1])
            img3 = Image.open(imgpaths[2])
            img4 = Image.open(imgpaths[3])
            img5 = Image.open(imgpaths[4])

            T = transforms.ToTensor()
            img1 = T(img1)
            img2 = T(img2)
            img3 = T(img3)
            img4 = T(img4)
            img5 = T(img5)

            imgs = [img1, img2, img3, img4, img5]
        
        return imgs, imgpaths

    def __len__(self):
        return len(self.file_list)


def get_loader(mode, data_root, batch_size, shuffle, num_workers, denoise=True, test_mode=None):
    if mode == 'train':
        is_training = True
    else:
        is_training = False
    dataset = Eleganceset(data_root, is_training=is_training, denoise=denoise)
    if mode == 'train':
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    else:
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
