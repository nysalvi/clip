from transformers import CLIPConfig, CLIPTextConfig, CLIPVisionConfig, CLIPModel, CLIPImageProcessor, CLIPTokenizer, CLIPTokenizerFast, CLIPProcessor
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from torchvision.transforms import v2
from pre_trained import PRE_TRAINED, VALUES, FILES
from itertools import product
from pathlib import Path
from PIL import Image
import argparse, os
import transformers
import json

def set_args(parser:argparse.ArgumentParser):        
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pretrained", '-pre', choices=[0, 1, 2, 3], help="Choose pre-trained model:" +
            "\n\t0 - {0}\n\t1 - {1}\n\t2 - {2}\n\t3 - {3}".format(*PRE_TRAINED, type=int))    
    group.add_argument("--default", action="store_const", const=1, 
            help="uses 'openai/clip-vit-base-patch32' as default for all classes")    
        
    parser.add_argument("--cfg_folder", 
        help="valid file names: model_cfg, img_cfg, tokenizer_cfg, tokenizerfast_cfg, txt_cfg and vision_cfg")
    parser.add_argument("--txt_cfg", help="specify a config file to override default configs in CLIPTextConfig")                
    parser.add_argument("--vision_config", "-vis_cfg", help="specify a config json file to override default configs in CLIPVisionConfig")            
    parser.add_argument("--token_cfg", help="specify a config json file to override default configs in CLIPTokenizer")    
    parser.add_argument("--img_processor", "-img_proc",  help="specify a config json file to override default configs in CLIPImageProcessor")
    parser.add_argument("--model_cfg", help="specify a config json file to override default configs in CLIPConfig")
    #parser.print_help()


def load_configs(cfg_folder, fast=""):
    path = f"./config/{cfg_folder}"

    model_cfg = open(f"{path}/model_cfg.json").read() if os.path.exists(f"{path}/model_cfg.json") else False
    img_cfg = open(f"{path}/img_cfg.json").read() if os.path.exists(f"{path}/img_cfg.json") else False    
    tokenizer_cfg = open(f"{path}/tokenizer_cfg.json").read() if os.path.exists(f"{path}/tokenizer_cfg.json") else False
    txt_cfg = open(f"{path}/txt_cfg.json").read() if os.path.exists(f"{path}/txt_cfg.json") else False
    vision_cfg = open(f"{path}/vision_cfg.json").read() if os.path.exists(f"{path}/vision_cfg.json") else False

    model_json = json.loads(model_cfg) if model_cfg else False
    img_json = json.loads(img_cfg) if img_cfg else False
    tokenizer_json = json.loads(tokenizer_cfg) if tokenizer_cfg else False
    txt_json = json.loads(txt_cfg) if txt_cfg else False
    vision_json = json.loads(vision_cfg) if vision_cfg else False

    return model_json, img_json, tokenizer_json, txt_json, vision_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)    
    set_args(parser)
    args = parser.parse_args()
    load_configs(args.cfg_folder, args.fast)