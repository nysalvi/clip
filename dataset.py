from torchvision.io import read_image
from torch import Dataset
from pathlib import Path
from sympy import Union
from ast import Tuple
import pandas as pd
import os 

class TextImagePairSet(Dataset):
    def __init__(self, annotations_dir:str, img_dir:str, transforms=None, target_transforms=None):
        """
        Args
        -------------------------------------------------------------------------------------
        annotations_folder : string to a directory containing 'train_labels.csv', 'dev_labels.csv' and 'test_labels.csv' with name and label as columns 
        img_dir : string to a directory with train, dev and test folders.
        transforms : transforms to images 
        target_transforms : transforms to labels 
        """        
        self.train_labels = pd.read_csv(os.path.join(annotations_dir, "train_labels.csv"))
        self.dev_labels = pd.read_csv(os.path.join(annotations_dir, "dev_labels.csv"))
        self.test_labels = pd.read_csv(os.path.join(annotations_dir, "test_labels.csv"))

        self.img_dir = img_dir
        self.transforms = transforms
        self.target_transforms = target_transforms

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join()
        image = 
        label = 
        #os.path.join()