from torchvision.transforms import v2
from torchvision import transforms
import transformers
import torchvision
import torch

AUGMENTATIONS = {
    "clip_preprocess" : v2.Compose([
        v2.Resize(224, interpolation=v2.InterpolationMode.BICUBIC),
        v2.CenterCrop((224, 224)),
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        v2.Normalize(transformers.image_utils.OPENAI_CLIP_MEAN, transformers.image_utils.OPENAI_CLIP_STD)
    ])
}



