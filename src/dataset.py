'''
DEFINES: how data is 
    * loaded
    * indexed 
    * preprocessed (transforms, augmentation, etc.)
'''
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image

##### CONSTANTS #####
IMAGENET_MEAN = [0.485, 0.456, 0.406]         # Mean of ImageNet dataset (used for normalization)
IMAGENET_STD = [0.229, 0.224, 0.225]          # Std of ImageNet dataset (used for normalization)

#### DATASETS ###
class RetinaDataset(Dataset):
    def __init__(self, folder_dir, dataframe, image_size, normalization=True):
        '''

        Arguments:
            folder_dir  -- string path to data folder
            dataframe   -- pandas data frame
            image_size  -- int, size of image to be resized/cropped before DNN
            normalization   -- whether applying mean and std of ImageNet
        '''
        self.image_paths = []
        self.image_labels = []

        # Define transformation sequence
        image_transformation = [
            transforms.Resize(image_size),
            transforms.ToTensor()
        ]

        if normalization:
            image_transformation.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))

        self.image_transformation = transforms.Compose(image_transformation)

        # Get all image paths and image labels from dataframe
        for index, row in dataframe.iterrows():
            image_path = os.path.join(folder_dir, row.filename)
            self.image_paths.append(image_path)
            labels = []
            for col in row[1:]: # TODO: consider optimize this
                if col == 1:
                    labels.append(1)
                else:
                    labels.append(0)
            self.image_labels.append(labels)
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # read image
        image_path = self.image_paths[index]
        image_data = Image.open(image_path).convert("RGB") # convert image to RGB

        # 
        image_data = self.image_transformation(image_data)

        return image_data, torch.FloatTensor(self.image_labels[index]), image_path
