from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import pdb
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
import random
from kitti_transform import *
# import scipy.misc


class depthDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, transform=None):
        self.frame = pd.read_csv(csv_file, header=None)
        self.transform = transform

    def __getitem__(self, idx):
        image_name = self.frame.loc[idx, 0]
        depth_name = self.frame.loc[idx, 1]

        path = './datasets/kitti/'
        image = Image.open(path + image_name)
        depth = Image.open(path + depth_name)

        sample = {'image': image, 'depth': depth}
        if self.transform:
            sample = self.transform(sample)
            
        return sample

    def __len__(self):
        return len(self.frame)


def getTrainingData(batch_size=64,csv_file='./datasets/kitti_train_sparse.csv'):

    __imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([
            [-0.5675,  0.7192,  0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948,  0.4203],
        ])
    }
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}

    transformed_training = depthDataset(csv_file,
                                        transform=transforms.Compose([                    
                                            RandomCrop([480,320],[240,160]),
                                            RandomHorizontalFlip(),
                                            ToTensor(),
                                            Lighting(0.1, __imagenet_pca[
                                                'eigval'], __imagenet_pca['eigvec']),
                                            ColorJitter(
                                                brightness=0.4,
                                                contrast=0.4,
                                                saturation=0.4,
                                            ),
                                            Normalize(__imagenet_stats['mean'],
                                                      __imagenet_stats['std'])
                                        ]))

    dataloader_training = DataLoader(transformed_training, batch_size,
                                     shuffle=True, num_workers=4, pin_memory=True)

    return dataloader_training



def getTestingData(batch_size=64):

    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}

    __imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([
            [-0.5675,  0.7192,  0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948,  0.4203],
        ])
    }
    # scale = random.uniform(1, 1.5)
    transformed_testing = depthDataset(csv_file='./datasets/kitti_val_select.csv',
                                       transform=transforms.Compose([
                                           ToTensor(),
                                           Lighting(0.1, __imagenet_pca[
                                                'eigval'], __imagenet_pca['eigvec']),
                                           Normalize(__imagenet_stats['mean'],
                                                     __imagenet_stats['std'])
                                       ]))

    dataloader_testing = DataLoader(transformed_testing, batch_size,
                                    shuffle=False, num_workers=4, pin_memory=True)

    return dataloader_testing
