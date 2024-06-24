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
    
    parser.add_argument("--fast", action="store_true", help="enables fast tokenizer CLIPTokenizerFast variant")

    parser.add_argument("--txt_cfg", help="specify a config json file to override default configs in CLIPTextConfig")                
    parser.add_argument("--vision_config", "-vis_cfg", help="specify a config json file to override default configs in CLIPVisionConfig")            
    parser.add_argument("--token_cfg", help="specify a config json file to override default configs in CLIPTokenizer")    
    parser.add_argument("--img_processor", "-img_proc",  help="specify a config json file to override default configs in CLIPImageProcessor")
    parser.add_argument("--model_cfg", help="specify a config json file to override default configs in CLIPConfig")
    #parser.print_help()


def update_pretrained_configs(pretrained:int):
    folder = PRE_TRAINED[pretrained]    
    os.makedirs(f"./defaults/{folder}/", exist_ok=True)

    pre_trained = VALUES[pretrained]
    model = FILES['model']
    txt_cfg = FILES['txt']
    vision_cfg = FILES['vision']
    tokenizer = FILES["tokenizer"]
    tokenizer_fast = FILES["tokenizer_fast"]
    vocabulary = FILES["vocabulary"]
    vocabulary_fast = FILES["vocabulary_fast"]
    img_processor_cfg = FILES["img_processor"]
        
    json.dump(CLIPModel.config_class().to_json_string(), open(f"./defaults/{folder}/{model}", 'w'))    
    json.dump(CLIPTextConfig.from_pretrained(pre_trained).to_json_string(), open(f"./defaults/{folder}/{txt_cfg}", 'w'))
    json.dump(CLIPVisionConfig.from_pretrained(pre_trained).to_json_string(), open(f"./defaults/{folder}/{vision_cfg}", 'w'))

    img_proc = CLIPImageProcessor.from_pretrained(pre_trained)    
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
    json.dump(img_proc_cfg, open(f"./defaults/{folder}/{img_processor_cfg}.json", 'w'))

    tok:CLIPTokenizer = CLIPTokenizer.from_pretrained(pre_trained)
    vocab = tok.get_vocab()
    json.dump(vocab, open(f"./defaults/{folder}/{vocabulary}", "w"))    
    tok = {"vocab_file" : f"./defaults/{folder}/{vocabulary}"}
    json.dump(tok, open(f"./defaults/{folder}/{tokenizer}", 'w'))

    tok_fast = CLIPTokenizerFast.from_pretrained(pre_trained)
    json.dump(tok_fast.get_vocab(), open(f"./defaults/{folder}/{vocabulary_fast}", "w"))
    tokfast_json = {"vocab_file" : f"./defaults/{folder}/{vocabulary_fast}"}
    json.dump(tokfast_json, open(f"./defaults/{folder}/{tokenizer_fast}", 'w'))
                
def process_args(args:dict):    

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

def get_defaults(pretrained, fast=False):
    folder = PRE_TRAINED[pretrained] 

    cfg = json.load(f"./defaults/{folder}/{FILES['model']}")
    cfg_text = json.load(f"./defaults/{folder}/{FILES['txt']}")
    cfg_vision = json.load(f"./defaults/{folder}/{FILES['vision']}")    
    cfg_token = json.load(f"./defaults/{folder}/{FILES["tokenizer_fast"]}") if fast else json.load(f"./defaults/{folder}/{FILES["tokenizer"]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)    
    set_args(parser)
    args = parser.parse_args()
    update_pretrained_configs(args.default if args.default else args.pre_trained)
    exit()    
        

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

    
    
