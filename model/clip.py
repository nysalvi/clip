from transformers import CLIPConfig, CLIPTextConfig, CLIPVisionConfig, CLIPModel, CLIPImageProcessor, CLIPTokenizer, CLIPTokenizerFast, CLIPProcessor
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from pre_trained import PRE_TRAINED, VALUES, FILES
from torchvision.transforms import v2
from pathlib import Path
import transformers
import torch
import json
import os

class CLIP(CLIPModel):
    def __init__(self, config:CLIPConfig):
        super().__init__(config)
        self.classification = torch.nn.Sequential([
            self.fc = torch.nn.Linear(in_features=768, out_features=1, bias=True)
        ])
    def forward(self, output):
        output = self(output)
        

