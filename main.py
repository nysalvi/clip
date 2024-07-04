from transformers import CLIPConfig, CLIPTextConfig, CLIPVisionConfig, CLIPModel, CLIPImageProcessor, CLIPTokenizer, CLIPTokenizerFast, CLIPProcessor
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from pre_trained import PRE_TRAINED, VALUES, FILES
from generator.clip_generator import CLIPGenerator
from torchvision.transforms import v2
from torch.nn import CrossEntropyLoss
from trainers.trainer import Trainer
from torch.optim import AdamW
from itertools import product
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import transformers
import argparse, os
import datetime
import json

def set_args():        
    parser = argparse.ArgumentParser(add_help=True)    
    parser.add_argument("--pretrained", '-pre', choices=[0, 1, 2, 3], help="Choose pre-trained model:" +
        "\n\t0 - {0}\n\t1 - {1}\n\t2 - {2}\n\t3 - {3}\nDefault: openai/clip-vit-base-patch32".format(*PRE_TRAINED, type=int), default=1)    
        
    parser.add_argument("--config", "-cfg", help="set configs folder to be loaded", default=False)    
    parser.add_argument("--resume_training", "-r", help="set OUTPUTS folder to be resumed", default=False)    
    parser.add_argument("--last", "-l", help="try to resume last experiment", default=False)    

    args = parser.parse_args()
    assert os.path.exists(f"./config/{args.config}/cfg.json"), "you need a configuration file"
    args.pretrained = VALUES[args.pretrained]    
    return args

def set_logs_folder(date:datetime.datetime, configs:dict):
    logs_folder = f".{os.sep}outputs{os.sep}{date.year}-{date.month}-{date.day}:{date.hour}:{date.minute}:{date.second}"

    for k, v in configs.items():
        if v:
            if not os.path.exists(logs_folder): os.makedirs(logs_folder)
            with open(f"{logs_folder}{os.sep}{k}.json") as file:
                file.write(v)

    return logs_folder

if __name__ == "__main__":
    args = set_args()    
    path = f"./config/{args.config}"
    architecture_cfg, img_processor_cfg, tokenizer_cfg = CLIPGenerator.load_configs(path)    
    config = f"{path}/cfg.json" if os.path.exists(f"{path}/cfg.json") else [False]


    combinations = product(architecture_cfg, img_processor_cfg, tokenizer_cfg, config)
    for model_cfg, processor_cfg, tok_cfg, cfg in combinations:
        date = datetime.datetime.now()
        save = {"model_cfg" : model_cfg, "processor" : processor_cfg, "tok_cfg" : tok_cfg, "cfg" : cfg}
        logs_folder = set_logs_folder(date, save)
        model, processor, tokenizer = CLIPGenerator.load_model(args.pretrained, model_cfg, processor_cfg, tok_cfg)
        model = model.to("cuda")

        optim = AdamW(model.parameters(), 1e-5)
        trainer = Trainer(optim, CrossEntropyLoss(), None, logs_folder, total_epochs=10)        
