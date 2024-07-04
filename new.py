#from metric import LR_SCHEDULERS, OPTIMIZERS, LOSSES
from generators.clip_generator import CLIPGenerator
from augmentations import AUGMENTATIONS
from _datasets.dataset import TextImagePairSet 
from trainers.trainer import Trainer
from metric_writer import Writer
from tqdm import tqdm
import argparse, os
import transformers
import torchvision
import _datasets
import datetime
import torch
import json

class ModuleToJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.nn.Module):
            return obj.__str__()        
        return super().default(obj)

def set_args():        
    parser = argparse.ArgumentParser(add_help=True)            
    parser.add_argument("--config", "-cfg", help="set configs folder to be loaded")    
    parser.add_argument("--resume_training", "-r", help="set OUTPUTS folder to be resumed", default=False)    
    parser.add_argument("--last", "-l", help="try to resume last experiment", default=False)        

    args = parser.parse_args()    
    path = f".{os.sep}config{os.sep}{args.config}"    
    assert os.path.exists(f"{path}{os.sep}model.json") and os.path.exists(f"{path}{os.sep}config.json"), "There is no model config"
    return args, path

def set_logs_folder(date:datetime.datetime, configs:dict):
    logs_folder = f".{os.sep}outputs{os.sep}{date.year}-{date.month}-{date.day};{date.hour}H{date.minute}M{date.second}S"

    for k, v in configs.items():
        if v:
            if not os.path.exists(logs_folder): os.makedirs(logs_folder)
            with open(f"{logs_folder}{os.sep}{k}.json", 'w') as file:
                if type(v) == dict:
                    file.write(json.dumps(v, cls=ModuleToJsonEncoder, indent=4).replace('\\n','\n'))
                else:
                    file.write(v)

    return logs_folder

def check_configs(cfg):
    keys = cfg.keys()
    if not "seed" in keys: cfg['seed'] = False
    if not "epoch" in keys: cfg['epoch'] = 0
    if not "device" in keys: cfg['device'] = 'cpu'
    if not "early_stop" in keys: cfg['early_stop'] = 0
    if not "lr_scheduler" in keys: cfg['lr_scheduler'] = False
    if not "stop" in keys: cfg['stop'] = 0
    if not "early_stop" in keys: cfg['early_stop'] = 0
    if not "max_metric" in keys: cfg['max_metric'] = 0
    
    assert "name" in keys, "Add a name to your model"
    assert "loss_fn" in keys, "Must specify a Loss function"
    assert "optimizer" in keys, "Must specify optimizer function"
    assert "path" in keys, "Must specify data path to save configs"
    assert "batch_size" in keys, "Must specify batch size"
    assert "total_epochs" in keys, "Must specify number of epochs to train"
    assert "metric" in keys, "Your model must have a function "

def load_dataset_transform(transform):
    if transform:
        return AUGMENTATIONS[transform]
    return False

def load_dataset(nameset, cfg:dict):
    cfg[nameset]['transform'] = load_dataset_transform(cfg[nameset]['transform'])
    cfg[nameset]['target_transform'] = load_dataset_transform(cfg[nameset]['target_transform'])
    return eval(cfg[nameset].pop('eval'))(**cfg[nameset])

if __name__ == "__main__":
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"    
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    args, path = set_args()    

    cfg = json.loads(open(f"{path}{os.sep}config.json", "r").read())    
    check_configs(cfg)
    
    trainLoader = load_dataset("train", cfg)
    devLoader = load_dataset("dev", cfg)
    testLoader = load_dataset("test", cfg)

    model_configs = CLIPGenerator.load_configs(path)
    model_configs['pretrained'] = cfg['pretrained']
    
    model, img_processor, tokenizer = CLIPGenerator.load_model(model_configs)    
    model = model.to(cfg['device'])
    params = [p for p in model.parameters() if p.requires_grad]    

    scheduler_cfg = cfg['lr_scheduler']
    optimizer_cfg = cfg['optimizer']    
    loss_cfg = cfg['loss_fn']

    optimizer = eval(optimizer_cfg.pop("eval"))(params=params, **optimizer_cfg)    
    loss = eval(loss_cfg.pop("eval"))(**loss_cfg)
    lr_scheduler = eval(scheduler_cfg.pop("eval"))(optimizer=optimizer, **scheduler_cfg)
    
    trainer_cfg = {
        **cfg,
        "optimizer" : optimizer, 
        "loss_fn" : loss, 
        "lr_scheduler" : lr_scheduler, 
        "device" : cfg['device'],
        "tokenizer" : tokenizer,
        "image_processor" : img_processor,        
    }

    trainer = Trainer(**trainer_cfg)        
    
    pbar = tqdm(range(cfg['epoch'], cfg['total_epochs'], 1))

    date = datetime.datetime.now()
    save = {"config" : cfg, "model" : model_configs}
    logs_folder = set_logs_folder(date, save)
    writer = Writer(cfg['name'], logs_folder)

    for i in pbar:                
        train_loss = trainer.train(model, trainLoader)
        dev_loss = trainer.validate(model, devLoader)
        test_loss = trainer.test(model, testLoader)        
        #pbar.format_dict['rate']
        #pbar.format_dict['elapsed']
        pbar.update()
        