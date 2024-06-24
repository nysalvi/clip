from enum import Enum

PRE_TRAINED = [
    "VIT_BASE_PATCH16",
    "VIT_BASE_PATCH32",
    "VIT_LARGE_PATCH14",
    "VIT_LARGE_PATCH14_336",
    "QUALCOMM"
]
VALUES = [
    "openai/clip-vit-base-patch16",
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14",
    "openai/clip-vit-large-patch14-336",
    "qualcomm/OpenAI-Clip"
]

FILES = {
    "model" : "cfg.json",
    "tokenizer" : "tokenizer_cfg.json",
    "tokenizer_fast" : "tokenizer_fast_cfg.json",
    "vocabulary" : "vocabulary.json",
    "vocabulary_fast" : "vocabulary_fast.json",
    "txt" : "txt_cfg.json",
    "img_processor" : "img_processor_cfg.json",
    "vision" : "vision_cfg.json",
}
