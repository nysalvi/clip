from enum import Enum
import transformers

    
class TOKENIZER(Enum):
    CLIP = "openai/clip-vit-base-patch32"

class IMAGE_PROCESSOR(Enum):
    CLIP = "openai/clip-vit-base-patch32"

class FEATURE_EXTRACTOR(Enum):
    CLIP = "openai/clip-vit-base-patch32"

class PROCESSOR(Enum):
    CLIP = "openai/clip-vit-base-patch32"

class MODEL(Enum):
    CLIP = "openai/clip-vit-base-patch32"

class BACKBONE(Enum):
    CLIP = "openai/clip-vit-base-patch32"

class TEXT_CONFIG(Enum):
    CLIP = [transformers.CLIPTextConfig, "openai/clip-vit-base-patch32"]

