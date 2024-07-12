from torch.utils.data import DataLoader
from _datasets.dataset import CLIPSet
from augmentations import AUGMENTATIONS
import _datasets
import datetime
import torch
import json
import os

class ModuleToJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.nn.Module):
            return obj.__str__()        
        return super().default(obj)

def get_dataset_transform(transform):
    if transform:
        return AUGMENTATIONS[transform]
    return False

def load_optimizer(optimizer_cfg:dict, model_params):
    return eval(optimizer_cfg.pop("eval"))(params=model_params, **optimizer_cfg)    

def load_lr_scheduler(lr_scheduler_cfg:dict, optimizer):
    return eval(lr_scheduler_cfg.pop("eval"))(optimizer=optimizer, **lr_scheduler_cfg)

def load_loss_fn(loss_fn_cfg:dict):
    return eval(loss_fn_cfg.pop("eval"))(**loss_fn_cfg)
    

def load_dataset(nameset, cfg:dict):    
    cfg[nameset]['transform'] = get_dataset_transform(cfg[nameset]['transform'])
    cfg[nameset]['target_transform'] = get_dataset_transform(cfg[nameset]['target_transform'])    
    dataset = eval(cfg[nameset].pop('eval'))(**cfg[nameset])    
    return DataLoader(dataset, batch_size=cfg['batch_size'])

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

