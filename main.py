from transformers import CLIPConfig, CLIPTextConfig, CLIPVisionConfig, CLIPModel, CLIPImageProcessor, CLIPTokenizer, CLIPTokenizerFast, CLIPProcessor
from itertools import product
from pathlib import Path
from PIL import Image
import transformers
import argparse, os
import json

def update_defaults():
    



def set_args(parser:argparse.ArgumentParser):
    parser.add_argument("--all_pretrained", '-all_pre', action="store_true", help="specify a string from model hub to be used in all config classes")
    parser.add_argument("--fast", action="store_true", help="enables fast tokenizer CLIPTokenizerFast variant")
    parser.add_argument("--all_default", '-all_D', action="store_true", help="enables all configs to be loaded with huggingface's default configs")

    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument("--txt_cfg", help="specify a config json file to override default configs in CLIPTextConfig")        
    group1.add_argument("--txt_pretrained", "-txt_pre", help="specify a string from model hub to load CLIPTextConfig")

    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument("--vis_cfg", help="specify a config json file to override default configs in CLIPVisionConfig")
    group2.add_argument("--vis_pretrained", "-vis_pre", help="specify a string from model hub to load CLIPVisionConfig")
    
    group3 = parser.add_mutually_exclusive_group()
    group3.add_argument("--token_vocabulary", "-tk_vocab", help="set path for loading own vocabulary")
    group3.add_argument("--token_pretrained", "-tk_pre", help="specify a string from model hub to load CLIPTokenizer/CLIPTokenizerFast")

    group4 = parser.add_mutually_exclusive_group()
    group4.add_argument("--img_processor", "-img_proc")
    group4.add_argument("--img_pretrained", "-img_pre", help="specify a string from model hub to load CLIPImageProcessor")

        
def get_configs(args:dict):
    text_config = {}
    vision_config = {}
    token_vocab = {}
    img_processor = {}

    #if args.all_default:

    if args.text_config:
        text_config = list(json.loads(open(f"./conf/{args.text_config}", 'r').read()).values() )    
    if args.vis_config:
        vision_config = list(json.loads(open(f"./conf/{args.vis_config}", 'r').read()).values() )    
    if args.token_vocab:
        token_vocab = list(json.loads(open(f"./conf/{args.token_vocab}", 'r').read()).values() )            
    if args.img_processor:
        img_processor = list(json.loads(open(f"./conf/{args.img_processor}", 'r').read()).values() )


    return (text_config, vision_config, token_vocab, img_processor)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    set_args(parser)
    
    args = parser.parse_args()
    configs = get_configs(args)
    
    #text_config, vision_config, token_vocab, img_processor
    configs = [x for x in product(*configs)]
    print(configs)
    exit()
    for config in configs:
        text_cfg = CLIPTextConfig(config[0])
        vis_cfg = CLIPVisionConfig(config[1])
        tokenizer = CLIPTokenizerFast(config[2]) if args.fast else CLIPTokenizer(config[2])
        img_processor = CLIPImageProcessor(config[3])
        clip_config = CLIPConfig(text_cfg, vis_cfg)


    #clip_config:CLIPConfig = CLIPConfig(text_config=pargs["text_config"], vision_config=pargs["vision_config"])
    #clip = CLIPModel(clip_config)
    
    
