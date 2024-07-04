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
    def __init__(self, kargs):
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        print("genio")        
        super(MyCLIP, self).__init__(kargs)
        for x in self.parameters():
            x.requires_grad_(False)
        self.fn = torch.nn.Linear(1024, 1, bias=True)

    def forward(self, X):
        print(X.shape)
        if len(X.shape) == 4:
            size = X.shape[0]
        else: 
            size = 1
        #outputs = super(MyCLIP, self).forward({'pixel_values' : X, 'attention_mask' : [1, 1, 1, 1, 1, 1], 'input_ids' : [344, 123, 431, 316,643, 647]})
        #vision_text_output = torch.cat(outputs['vision_model_outputs'], outputs['text_model_outputs'])
        #return self.fn(vision_text_output)
        return self.fn(torch.randn(size, 1024).to('cuda:0'))

