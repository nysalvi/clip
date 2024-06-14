from transformers import CLIPConfig, CLIPTextConfig, CLIPVisionConfig, CLIPModel, CLIPImageProcessor, CLIPTokenizer, CLIPTokenizerFast, CLIPProcessor
from PIL import Image
from pathlib import Path
import transformers
import argparse, os
import json

def parse(args:dict):
    n = {}
    if os.path.exists(f"./conf/{args.text_config}"):        
        n['text_config'] = json.loads(open(f"./conf/{args.text_config}", 'r').read())
    if os.path.exists(f"./conf/{args.vision_config}"):
        n['vision_config'] = json.loads(open(f"./conf/{args.vision_config}", 'r').read())
    if os.path.exists(f"./conf/{args.tokenizer_config}"):
        n['tokenizer_config'] = json.loads(open(f"./conf/{args.tokenizer_config}", 'r').read())        

    
    return n

def run_config(args, arg_name):
    func = args[arg_name]['function']
    config = args[arg_name]['config']
    return eval(f"{func}({config})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_str")
    parser.add_argument("--model_config")
    parser.add_argument("--text_config")
    parser.add_argument("--vision_config")
    parser.add_argument("--tokenizer_config")
    
    args = parser.parse_args()
    pargs = parse(args)
    
    clip_config:CLIPConfig = CLIPConfig(text_config=pargs["text_config"], vision_config=pargs["vision_config"])
    clip = CLIPModel(clip_config)
    
    
