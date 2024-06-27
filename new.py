from transformers import CLIPConfig, CLIPTextConfig, CLIPVisionConfig, CLIPModel, CLIPImageProcessor, CLIPTokenizer, CLIPTokenizerFast, CLIPProcessor
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from pre_trained import PRE_TRAINED, VALUES, FILES
from torchvision.transforms import v2
from torch.nn import CrossEntropyLoss
from trainer import CLIPTrainer
from torch.optim import AdamW
from itertools import product
from pathlib import Path
from PIL import Image
import transformers
import argparse, os
import datetime
import json

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
        return (False, False, False, False)        

    model_exists = f"{path}/model.json" if os.path.exists(f"{path}/model.json") else False
    model_cfg = open(model_exists).read() if model_exists else False

    processor_exists = f"{path}/processor.json" if os.path.exists(f"{path}/processor.json") else False    
    processor_cfg = open(processor_exists).read() if processor_exists else False

    tokenizer_exists = f"{path}/tokenizer.json" if os.path.exists(f"{path}/tokenizer.json") else False
    tokenizer_cfg = open(tokenizer_exists).read() if tokenizer_exists else False

    cfg = f"{path}/cfg.json" if os.path.exists(f"{path}/cfg.json") else False
    return (model_cfg, processor_cfg, tokenizer_cfg, cfg)


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

def set_logs_folder(date:datetime.datetime, configs:list=[False, False, False, False]):
    logs_folder = f".{os.sep}outputs{os.sep}{date.year}-{date.month}-{date.day}:{date.hour}:{date.minute}:{date.second}"

    if not os.path.exists(logs_folder): os.makedirs(logs_folder)

    if configs[0]:
        with open(f"{logs_folder}{os.sep}model.json") as model:
            model.write(configs[0])            
    if configs[1]:
        with open(f"{logs_folder}{os.sep}processor.json") as processor:
            processor.write(configs[1])            
    if configs[2]:
        with open(f"{logs_folder}{os.sep}tokenizer.json") as tokenizer:
            tokenizer.write(configs[2])
    if configs[3]:
        with open(f"{logs_folder}{os.sep}cfg.json") as cfg:
            cfg.write(configs[3])

    return logs_folder

if __name__ == "__main__":
    args = set_args()    
    architecture_cfg, img_processor_cfg, tokenizer_cfg, config = load_configs(f"./config/{args.config}")    
    combinations = product(architecture_cfg, img_processor_cfg, tokenizer_cfg, config)
    for model_cfg, processor_cfg, tok_cfg, cfg in combinations:
        date = datetime.datetime.now()
        logs_folder = set_logs_folder(date, [model_cfg, processor_cfg, tok_cfg, cfg])
        model, processor, tokenizer = load_model(args.pretrained, model_cfg, processor_cfg, tok_cfg)
        model = model.to("cuda")
        processor = processor.to("cuda")
        tokenizer = tokenizer.to("cuda")

        optim = AdamW(model.parameters(), 1e-5)
        trainer = CLIPTrainer(optim, CrossEntropyLoss(), None, logs_folder, total_epochs=10)

