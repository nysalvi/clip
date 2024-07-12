from _datasets.dataset import CLIPSet
from utils import load_loss_fn, load_lr_scheduler, load_optimizer, set_logs_folder, load_dataset
#from metric import LR_SCHEDULERS, OPTIMIZERS, LOSSES
from generators.clip_generator import CLIPGenerator
from trainers.trainer import Trainer
from metric_writer import Writer
from tqdm import tqdm
import argparse, os
import _datasets
import datetime
import json

def set_args():        
    parser = argparse.ArgumentParser(add_help=True)            
    parser.add_argument("--config", "-cfg", help="set configs folder to be loaded")    
    parser.add_argument("--resume_training", "-r", help="set OUTPUTS folder to be resumed", default=False)    
    parser.add_argument("--last", "-l", help="try to resume last experiment", default=False)        

    args = parser.parse_args()    
    path = f".{os.sep}config{os.sep}{args.config}"    
    assert os.path.exists(f"{path}{os.sep}model.json") and os.path.exists(f"{path}{os.sep}config.json"), "There is no model config"
    assert not args.last, "last run is not implemented yet"
    assert not args.resume_training, "resume training is not implemented yet"
    return args, path

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


if __name__ == "__main__":    
    args, path = set_args()    
    cfg = json.loads(open(f"{path}{os.sep}config.json", "r").read())    
    
    check_configs(cfg)
    
    model_configs = CLIPGenerator.load_configs(path)
    
    model_configs['pretrained'] = cfg['pretrained']
    optimizer_cfg = cfg['optimizer']    
    scheduler_cfg = cfg['lr_scheduler']
    loss_cfg = cfg['loss_fn']
    
    trainLoader = load_dataset("train", {**cfg, **model_configs})
    devLoader =  load_dataset("dev", {**cfg, **model_configs})
    testLoader =  load_dataset("test", {**cfg, **model_configs})
    
    #model, img_processor, tokenizer = CLIPGenerator.load_model(model_configs)    
    model = CLIPGenerator.load_model(model_configs).to(cfg['device'])        

    params = [p for p in model.parameters() if p.requires_grad]    
    optimizer = load_optimizer(optimizer_cfg, params)
    loss = load_loss_fn(loss_cfg)
    lr_scheduler = load_lr_scheduler(scheduler_cfg, optimizer)
    
    trainer_cfg = {        
        "optimizer" : optimizer, 
        "loss_fn" : loss, 
        "lr_scheduler" : lr_scheduler, 
        "device" : cfg['device']       
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
        
        it_second = pbar.format_dict['rate']
        time = pbar.format_dict['elapsed']
        
        pbar.update()
