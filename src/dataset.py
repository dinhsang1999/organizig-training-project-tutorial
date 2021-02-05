'''
DEFINES: how data is 
    * loaded
    * indexed 
    * preprocessed (transforms, augmentation, etc.)
'''
import torchvision.transforms as transforms
import torch.utils.data import Dataset

##### CONSTANTS #####
IMAGENET_MEAN = [0.485, 0.456, 0.406]         # Mean of ImageNet dataset (used for normalization)
IMAGENET_STD = [0.229, 0.224, 0.225]          # Std of ImageNet dataset (used for normalization)

#### DATASETS ###
class RetinaDataset(Dataset):
    def __init__(self, folder_dir, dataframe, image_size, normalization=True):
        pass
