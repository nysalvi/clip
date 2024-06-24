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
        
    parser.add_argument("--folder", 
        help="valid file names: model_cfg, img_cfg, tokenizer_cfg, tokenizerfast_cfg, txt_cfg and vision_cfg", default=False)
    
    #parser.add_argument("--txt_cfg", help="specify a config file to override default configs in CLIPTextConfig")                
    #parser.add_argument("--vision_config", "-vis_cfg", help="specify a config json file to override default configs in CLIPVisionConfig")            
    #parser.add_argument("--token_cfg", help="specify a config json file to override default configs in CLIPTokenizer")    
    #parser.add_argument("--img_processor", "-img_proc",  help="specify a config json file to override default configs in CLIPImageProcessor")
    #parser.add_argument("--model_cfg", help="specify a config json file to override default configs in CLIPConfig")    
    
def check_configs(cfg_folder):
    if not cfg_folder:
        return (False, False, False, False, False)
    path = f"./config/{cfg_folder}"

    model_exists = True if os.path.exists(f"{path}/model_cfg.json") else False
    img_exists = True if os.path.exists(f"{path}/img_cfg.json") else False    
    tokenizer_exists = True if os.path.exists(f"{path}/tokenizer_cfg.json") else False

    


def load_configs(cfg_folder):
    if not cfg_folder:
        return (False, False, False, False, False)    
    path = f"./config/{cfg_folder}"
    
    model_cfg = open(f"{path}/model_cfg.json").read() if os.path.exists(f"{path}/model_cfg.json") else False
    img_cfg = open(f"{path}/img_cfg.json").read() if os.path.exists(f"{path}/img_cfg.json") else False    
    tokenizer_cfg = open(f"{path}/tokenizer_cfg.json").read() if os.path.exists(f"{path}/tokenizer_cfg.json") else False

    model_json = json.loads(model_cfg) if model_cfg else False
    img_json = json.loads(img_cfg) if img_cfg else False
    tokenizer_json = json.loads(tokenizer_cfg) if tokenizer_cfg else False


    txt_file = open(f"{path}/{model_cfg['text_config']}").read()
    txt_json = json.loads(txt_file)

    vision_file = open(f"{path}/{model_cfg['vision_config']}").read()
    vision_json = json.loads(vision_file)

    return (model_json, img_json, tokenizer_json, txt_json, vision_json)

#def load_model(model_json, img_json, tokenizer_json, txt_json, vision_json):
    #model = 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)    
    set_args(parser)
    args = parser.parse_args()
    #print(args.folder)
    #model_json, img_json, tokenizer_json, txt_json, vision_json = load_configs(args.folder)

