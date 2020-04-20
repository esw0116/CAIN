import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random

class Eleganceset(Dataset):
    def __init__(self, data_root, is_training):
        self.data_root = data_root
        self.training = is_training

        train_folder = os.path.join(self.data_root, 'elegance', 'train/LR_bicubic/X2')
        val_folder = os.path.join(self.data_root, 'elegance', 'val/LR_bicubic/X2')

        self.folder = train_folder if self.training else val_folder

        subfolders = sorted(glob.glob(self.folder+ '/*'))

        self.file_list = []

        for subfolder in subfolders:
            subfolder_file_list = sorted(glob.glob(subfolder + '/*'))
            subfolder_file_list = subfolder_file_list[2:-2]
            self.file_list.extend(subfolder_file_list)
        
        self.transforms = transforms.Compose([
            transforms.RandomCrop(512),
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

        seed = random.randint(0, 100)
        if seed % 2 == 0:
            past_idx_string = '{:03d}'.format(current_idx - 1)
            past_img_path = current_img_path.replace(current_idx_string, past_idx_string)

            future_idx_string = '{:03d}'.format(current_idx + 1)
            future_img_path = current_img_path.replace(current_idx_string, future_idx_string)

        else:
            past_idx_string = '{:03d}'.format(current_idx - 2)
            past_img_path = current_img_path.replace(current_idx_string, past_idx_string)

            future_idx_string = '{:03d}'.format(current_idx + 2)
            future_img_path = current_img_path.replace(current_idx_string, future_idx_string)

        imgpaths = [past_img_path, current_img_path, future_img_path]

        # Load images
        img1 = Image.open(imgpaths[0])
        img2 = Image.open(imgpaths[1])
        img3 = Image.open(imgpaths[2])

        # Data augmentation
        if self.training:
            seed = random.randint(0, 2**32)
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
        else:
            T = transforms.ToTensor()
            img1 = T(img1)
            img2 = T(img2)
            img3 = T(img3)

        imgs = [img1, img2, img3]
        
        return imgs, imgpaths

    def __len__(self):
        return len(self.file_list)
        '''
        if self.training:   
            return len(self.trainlist)
        else:
            return len(self.testlist)
        return 0
        '''

def get_loader(mode, data_root, batch_size, shuffle, num_workers, test_mode=None):
    if mode == 'train':
        is_training = True
    else:
        is_training = False
    dataset = Eleganceset(data_root, is_training=is_training)
    if mode == 'train':
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    else:
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
