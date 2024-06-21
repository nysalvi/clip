from transformers import CLIPConfig, CLIPTextConfig, CLIPVisionConfig, CLIPModel, CLIPImageProcessor, CLIPTokenizer, CLIPTokenizerFast, CLIPProcessor
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from itertools import product
from pathlib import Path
from PIL import Image
from clip import PRE_TRAINED
import transformers
import argparse, os
import json

def update_defaults():
    CLIPModel.config_class().to_json_file("./defaults/cfg.json")

    CLIPTextConfig.from_pretrained("openai/clip-vit-base-patch32").to_json_file("./defaults/text_config.json")
    CLIPVisionConfig.from_pretrained("openai/clip-vit-base-patch32").to_json_file("./defaults/vision_config.json")

    img_proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")#.to_json_file("./defaults/img_processor.json")
    img_proc_cfg = {
        "do_resize" : img_proc.do_resize,
        "size" : img_proc.size,
        "resample" : img_proc.resample,
        "do_center_crop" : img_proc.do_center_crop,
        "crop_size" : img_proc.crop_size,
        "do_rescale" : img_proc.do_rescale,
        "rescale_factor" : img_proc.rescale_factor,
        "do_normalize" : img_proc.do_normalize,
        "image_mean" : img_proc.image_mean,
        "image_std" : img_proc.image_std,
        "do_convert_rgb" : img_proc.do_convert_rgb
    }
    json.dump(img_proc_cfg, open("./defaults/img_processor.json"))

    tokenizer:CLIPTokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    vocab = vocab.get_vocab()
    json.dump(vocab, open("./defaults/tokenizer_vocab.json", "w"))    
    tokenizer = {"vocab_file" : tokenizer.get_vocab()}
    json.dump(tokenizer, open("./defaults/tokenizer.json", 'w'))

    tokenizer_fast = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
    json.dump(tokenizer_fast.get_vocab(), open("./defaults/tokenizerfast_vocab.json", "w"))
    tokfast_json = {"vocab_file" : tokenizer_fast.get_vocab()}
    json.dump(tokfast_json, open("./defaults/tokenizerfast.json", 'w'))

        

def set_args(parser:argparse.ArgumentParser):
    parser.add_argument("--pretrained", '-pre', help="choose between the following pre-trained values \n", type=int)
    parser.add_argument("--fast", action="store_true", help="enables fast tokenizer CLIPTokenizerFast variant")
    parser.add_argument("--defaults", action="store_true", help="uses 'openai/clip-vit-base-patch32' as default for all classes")
    
    parser.add_argument("--txt_cfg", help="specify a config json file to override default configs in CLIPTextConfig")        
    parser.add_argument("--txt_pretrained", "-txt_pre", help="specify a string from model hub to load CLIPTextConfig")
    
    parser.add_argument("--vision_config", "-vision_cfg", help="specify a config json file to override default configs in CLIPVisionConfig")
    parser.add_argument("--vision_pretrained", "-vision_pre", help="specify a string from model hub to load CLIPVisionConfig")
        
    parser.add_argument("--token_cfg", "-tk_cfg", help="specify a config json file to override default configs in CLIPTokenizer")
    parser.add_argument("--token_pretrained", "-tk_pre", help="specify a string from model hub to load CLIPTokenizer")
    
    parser.add_argument("--img_processor", "-img_proc")
    parser.add_argument("--img_pretrained", "-img_pre", help="specify a string from model hub to load CLIPImageProcessor")


def get_defaults(fast=False):
    cfg = json.load("./defaults/cfg.json")
    cfg_text = json.load("./defaults/text_config.json")
    cfg_vision = json.load("./defaults/vision_config.json")    
    cfg_token = json.load("./defaults/tokenizer_fast.json") if fast else json.load("./defaults/tokenizer.json")
    
        
def process_args(args:dict):
    get_defaults(args.fast)

    text_config = {}
    vision_config = {}
    token_vocab = {}
    img_processor = {}

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
    update_defaults()

    args = parser.parse_args()
    configs = process_args(args)
    
    #text_config, vision_config, token_vocab, img_processor
    configs = [x for x in product(*configs)]
    print(configs)
    exit()
    for config in configs:
        text_cfg = CLIPTextConfig(config[0])
        vision_config = CLIPVisionConfig(config[1])
        tokenizer = CLIPTokenizerFast(config[2]) if args.fast else CLIPTokenizer(config[2])
        img_processor = CLIPImageProcessor(config[3])
        clip_config = CLIPConfig(text_cfg, vision_config)

    
    
