from transformers import CLIPConfig, CLIPModel, CLIPImageProcessor, CLIPTokenizerFast
from generators.base_generator import Generator
from trainers.clip_trainer import CLIPTrainer
from enum import Enum
import json
import os 


class CLIPGenerator(Generator):

    def check_configs(path):                    
        model_cfg = open(f"{path}/model.json").read() if os.path.exists(f"{path}/model.json") else False        
        processor_cfg = open(f"{path}/img_processor.json").read() if os.path.exists(f"{path}/img_processor.json") else False            
        tokenizer_cfg = open(f"{path}/tokenizer.json").read()  if os.path.exists(f"{path}/tokenizer.json") else False
        
        return model_cfg, processor_cfg, tokenizer_cfg

    def load_configs(path):            
        model_cfg, img_cfg, tokenizer_cfg = CLIPGenerator.check_configs(path)
        
        model_json = json.loads(model_cfg)
        model_json = False if len(model_json) == 0 else model_json

        processor_json = json.loads(img_cfg)
        processor_json = False if len(processor_json) == 0 else processor_json

        tokenizer_json = json.loads(tokenizer_cfg)
        tokenizer_json = False if len(tokenizer_json) == 0 else tokenizer_json

        return {'model' : model_json,'img_processor' : processor_json, 'tokenizer' : tokenizer_json}

    #def load_model(pretrained, model_json, img_json, tokenizer_json):            
    #    if model_json:
    #        print(model_json)
    #        clip_cfg = CLIPConfig(**model_json)        
    #        model = CLIPModel.from_pretrained(pretrained_model_name_or_path=pretrained, config=clip_cfg)
    #    else:
    #        model = CLIPModel.from_pretrained(pretrained_model_name_or_path=pretrained)
    #    if img_json:        
    #        img_processor = CLIPImageProcessor.from_pretrained(pretrained_model_name_or_path=pretrained, **img_json)
    #    else:
    #        img_processor = CLIPImageProcessor.from_pretrained(pretrained_model_name_or_path=pretrained)
    #    if tokenizer_json:
    #        tokenizer = CLIPTokenizerFast.from_pretrained(pretrained_model_name_or_path=pretrained, **tokenizer_json)
    #    else: 
    #        tokenizer = CLIPTokenizerFast.from_pretrained(pretrained_model_name_or_path=pretrained)
    #
    #    return (model, img_processor, tokenizer)


    def load_model(configs):            
        pretrained = configs['pretrained']
        model_json = configs['model']
        img_json = configs['img_processor']
        tokenizer_json = configs['tokenizer']

        if model_json:
            clip_cfg = CLIPConfig(model_json)        
            model = CLIPModel.from_pretrained(pretrained_model_name_or_path=pretrained, config=clip_cfg)
        else:
            model = CLIPModel.from_pretrained(pretrained)
        if img_json:        
            img_processor = CLIPImageProcessor.from_pretrained(pretrained_model_name_or_path=pretrained, config=img_json)
        else:
            img_processor = CLIPImageProcessor.from_pretrained(pretrained_model_name_or_path=pretrained)
        if tokenizer_json:
            tokenizer = CLIPTokenizerFast.from_pretrained(pretrained_model_name_or_path=pretrained, config=tokenizer_json)
        else: 
            tokenizer = CLIPTokenizerFast.from_pretrained(pretrained_model_name_or_path=pretrained)
        return (model, img_processor, tokenizer)
