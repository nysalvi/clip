#from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
#from transformers import CLIPConfig, CLIPTextConfig, CLIPVisionConfig 
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
#from transformers import CLIPTokenizerFast, CLIPImageProcessor
#from pre_trained import PRE_TRAINED, VALUES, FILES
from torchvision.transforms import v2
from pathlib import Path
import transformers
import torch
import json
import os

class MyCLIP(CLIPModel):
    def __init__(self):
        super().__init__()
        for x in self.parameters():
            x.requires_grad_(False)
        self.fn = torch.nn.Sequential([
            
        ])
    def forward(self, X):
        return super()(X)

