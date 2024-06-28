from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode
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
        annotations_file : annotations csv with ´name´, ´y´ and ´mask´ as columns
        img_dir : string to a directory with images folder.
        transforms : transforms to images 
        target_transforms : transforms to labels 
        """        
        self.annotations_csv = pd.read_csv(annotations_file, sep=';')
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.annotations_csv)

    def __getitem__(self, idx):                        
        img_path = os.path.join(self.img_dir, self.annotations_csv.iloc[idx, 0])
        
        image = read_image(img_path, ImageReadMode.RGB)
        label = self.annotations_csv.iloc[idx, 2]
        y = self.annotations_csv.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)                
        return image, y, label