from torchvision.io import read_image
from torch import Dataset
from pathlib import Path
from sympy import Union
from ast import Tuple
import pandas as pd
import os 


class TextImagePairSet(Dataset):
    def __init__(self, annotations_file:str, img_dir:str, transform=None, target_transform=None):
        """
        Args
        -------------------------------------------------------------------------------------
        annotations_folder : string to a directory containing csv with file ´name´ and ´y´ as columns
        img_dir : string to a directory with images folder.
        transforms : transforms to images 
        target_transforms : transforms to labels 
        """        
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_dir[idx, 0])
        image = read_image(img_path)
        label = os.path.join(self.img_dir, self.img_dir[idx, 1])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label