from metric import LR_SCHEDULERS, OPTIMIZERS, LOSSES
from generators.clip_generator import CLIPGenerator
from metric import Writer
from trainers.clip_trainer import CLIPTrainer
from tqdm import tqdm
import argparse, os
import datetime
import torch
import json
import sys

#sys.args['TF_ENABLE_ONEDNN_OPTS'] = '0'

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
    logs_folder = f".{os.sep}outputs{os.sep}{date.year}-{date.month}-{date.day}:{date.hour}:{date.minute}:{date.second}"

    for k, v in configs.items():
        if v:
            if not os.path.exists(logs_folder): os.makedirs(logs_folder)
            with open(f"{logs_folder}{os.sep}{k}.json") as file:
                file.write(v)
    return logs_folder

def check_configs(cfg):
    keys = cfg.keys()
    if not "seed" in keys: cfg['seed'] = False
    if not "epoch" in keys: cfg['epoch'] = 0
    if not "device" in keys: cfg['device'] = 'cpu'
    if not "early_stop" in keys: cfg['early_stop'] = 0

    assert "loss_fn" in keys, "Must specify a Loss function"
    assert "optimizer" in keys, "Must specify optimizer function"
    assert "path" in keys, "data path must be in config"
    assert "batch_size" in keys, "Must specify batch size"

if __name__ == "__main__":
    args, path = set_args()
    cfg = json.loads(open(f"{path}{os.sep}config.json", "r").read())    
    check_configs(cfg)
    trainLoader = eval(cfg['dataset'].pop('eval'))(**cfg['dataset'])


    model_configs = CLIPGenerator.load_configs(path)
    model_configs['pretrained'] = cfg['pretrained']
    model, img_processor, tokenizer = CLIPGenerator.load_model(model_configs)
    
    optimizer_cfg = cfg['optimizer']

    linear = torch.nn.Linear(512, 4, True)
    optimizer = eval(optimizer_cfg.pop("eval"))(params=linear.parameters(), **optimizer_cfg)
    
    loss_cfg = cfg['loss_fn']
    loss = eval(loss_cfg.pop("eval"))(**loss_cfg)
    scheduler_cfg = cfg['lr_scheduler']
    lr_scheduler = eval(scheduler_cfg.pop("eval"))(optimizer=optimizer, **scheduler_cfg)

    trainer_cfg = {
        "optimizer" : optimizer, 
        "loss_fn" : loss, 
        "lr_scheduler" : lr_scheduler, 
        "device" : cfg['device'],
        "tokenizer" : tokenizer,
        "image_processor" : img_processor
        }
    trainer = CLIPTrainer(**trainer_cfg)        
    
    pbar = tqdm(range(cfg['epoch'], cfg['total_epochs'], 1))
    for i in pbar:
        writer = Writer()

        pbar.update()
    #for i in tqdm(range(cfg['total_epochs'])):    
    print(trainer)
    #optim_name = OPTIMIZERS[optimizer_cfg.pop('eval')]
    #adam = optim_name(linear.parameters(), **optimizer_cfg)
    #loss_name = LOSSES[loss_cfg.pop('eval')]    
    #loss = loss_name(**loss_cfg)
    #scheduler_name = LR_SCHEDULERS[scheduler_cfg.pop("eval")]
    #scheduler = scheduler_name(optimizer=optimizer, **scheduler_cfg)
    print(type(optimizer))
    print(optimizer)
    print(type(loss))
    print(lr_scheduler)
    #print(model)
    #print(configs)
    #print(optim_name)    
    #print(adam)
    #print(loss_name)
    #print(loss)
    #print(scheduler_name)
    #print(scheduler)
    #all_configs = GENERATOR[config['model']].value[0].load_configs(path)    
    #date = datetime.datetime.now()
    #save = {"model_cfg" : model_cfg, "img_processor" : processor_cfg, "tok_cfg" : tok_cfg, "cfg" : cfg}
    #logs_folder = set_logs_folder(date, save)
    #model, img_processor, tokenizer = CLIPGenerator.load_model(args.pretrained, model_cfg, processor_cfg, tok_cfg)
    #model = model.to("cuda")
    #optim = AdamW(model.parameters(), 1e-5)
    #trainer = CLIPTrainer(optim, CrossEntropyLoss(), None, logs_folder, total_epochs=10)        
