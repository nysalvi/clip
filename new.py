from transformers import CLIPConfig, CLIPTextConfig, CLIPVisionConfig, CLIPModel, CLIPImageProcessor, CLIPTokenizer, CLIPTokenizerFast, CLIPProcessor
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from pre_trained import PRE_TRAINED, VALUES, FILES
from torchvision.transforms import v2
from itertools import product
from pathlib import Path
from PIL import Image
import argparse, os
import transformers
import json


#parser.add_argument("--txt_cfg", help="specify a config file to override default configs in CLIPTextConfig")                
#parser.add_argument("--vision_config", "-vis_cfg", help="specify a config json file to override default configs in CLIPVisionConfig")            
#parser.add_argument("--token_cfg", help="specify a config json file to override default configs in CLIPTokenizer")    
#parser.add_argument("--img_processor", "-img_proc",  help="specify a config json file to override default configs in CLIPImageProcessor")
#parser.add_argument("--model_cfg", help="specify a config json file to override default configs in CLIPConfig")    
    

def set_args():        
    parser = argparse.ArgumentParser(add_help=True)    
    parser.add_argument("--pretrained", '-pre', choices=[0, 1, 2, 3], help="Choose pre-trained model:" +
        "\n\t0 - {0}\n\t1 - {1}\n\t2 - {2}\n\t3 - {3}\nDefault: openai/clip-vit-base-patch32".format(*PRE_TRAINED, type=int), default=1)    
        
    parser.add_argument("--config", "-cfg", help="set configs file to be loaded", default=False)

    args = parser.parse_args()
    args.pretrained = VALUES[args.pretrained]    
    return args


def check_configs(path):
    if not path:
        return (False, False, False, False, False)        

    model_exists = f"{path}/model.json" if os.path.exists(f"{path}/model.json") else False
    processor_exists = f"{path}/processor.json" if os.path.exists(f"{path}/processor.json") else False    
    tokenizer_exists = f"{path}/tokenizer.json" if os.path.exists(f"{path}/tokenizer.json") else False

    model_cfg = open(model_exists).read() if model_exists else False
    processor_cfg = open(processor_exists).read() if processor_exists else False
    tokenizer_cfg = open(tokenizer_exists).read() if tokenizer_exists else False

    return (model_cfg, processor_cfg, tokenizer_cfg)


def load_configs(path):            
    model_exists, img_exists, tokenizer_exists = check_configs(path)
    
    model_json = list(json.loads(model_exists).values()) if model_exists else [False]
    processor_json = list(json.loads(img_exists).values()) if img_exists else [False]
    tokenizer_json = list(json.loads(tokenizer_exists).values()) if tokenizer_exists else [False]
    
    return (model_json, processor_json, tokenizer_json)


def load_model(pretrained, model_json, img_json, tokenizer_json):
    if model_json:
        clip_cfg = CLIPConfig(*model_cfg)        
        model = CLIPModel.from_pretrained(pretrained_model_name_or_path=pretrained, config=clip_cfg)
    else:
        model = CLIPModel.from_pretrained(pretrained)

    if img_json:        
        processor = CLIPImageProcessor.from_pretrained(pretrained_model_name_or_path=pretrained, config=img_json)
    else:
        processor = CLIPImageProcessor.from_pretrained(pretrained_model_name_or_path=pretrained)
    if tokenizer_json:
        tokenizer = CLIPTokenizerFast.from_pretrained(pretrained_model_name_or_path=pretrained, config=tokenizer_json)
    else: 
        tokenizer = CLIPTokenizerFast.from_pretrained(pretrained_model_name_or_path=pretrained)

    return (model, processor, tokenizer)


if __name__ == "__main__":
    args = set_args()    
    model_cfg, processor_cfg, tokenizer_cfg = load_configs(f"./config/{args.config}")
        
    combinations = product(model_cfg, processor_cfg, tokenizer_cfg)
    for cfg, img, tokenizer in combinations:
        model, processor, tokenizer = load_model(args.pretrained, cfg, img, tokenizer)
             