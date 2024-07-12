from transformers import CLIPImageProcessor, CLIPTokenizerFast
from torchvision.io import ImageReadMode
from torchvision.io import read_image
from torch.utils.data import Dataset
from pathlib import Path
from sympy import Union
from ast import Tuple
import pandas as pd
import torch
import os 


class CLIPSet(Dataset):

    def __init__(self, annotations_file:str, cfg):        
        pretrained = cfg['pretrained']
        img_json = cfg['img_processor']
        tokenizer_json = cfg['tokenizer']
                 
        self.annotations_csv = pd.read_csv(annotations_file, sep=';')
        self.img_dir = cfg['img_dir']
        self.transform = cfg['transform']
        self.target_transform = cfg['target_transform']                

        if img_json:        
            self.img_processor = CLIPImageProcessor.from_pretrained(pretrained_model_name_or_path=pretrained, config=img_json)
        else:
            self.img_processor = CLIPImageProcessor.from_pretrained(pretrained_model_name_or_path=pretrained)
        if tokenizer_json:
            self.tokenizer = CLIPTokenizerFast.from_pretrained(pretrained_model_name_or_path=pretrained, config=tokenizer_json)
        else: 
            self.tokenizer = CLIPTokenizerFast.from_pretrained(pretrained_model_name_or_path=pretrained)



    def __len__(self):
        return len(self.annotations_csv)

    def __getitem__(self, idx):                        
        img_path = os.path.join(self.img_dir, self.annotations_csv.iloc[idx, 0])
        
        image = read_image(img_path, ImageReadMode.RGB)
        label = self.annotations_csv.iloc[idx, 2]
        y = self.annotations_csv.iloc[idx, 1]        
        if self.transform:
            image = self.transform(image)
        label = self.tokenizer(label)
        if self.target_transform:
            label = self.target_transform(label)   
        y = torch.tensor(y, dtype=torch.float64)
        return image, y.unsqueeze(0), label
    