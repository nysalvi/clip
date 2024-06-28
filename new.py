from metrics import LR_SCHEDULERS, OPTIMIZERS, LOSSES
from generator.clip_generator import CLIPGenerator
from trainer import Trainer
import argparse, os
import datetime
import torch
import json

def set_args():        
    parser = argparse.ArgumentParser(add_help=True)    
        
    parser.add_argument("--config", "-cfg", help="set configs folder to be loaded")    
    parser.add_argument("--resume_training", "-r", help="set OUTPUTS folder to be resumed", default=False)    
    parser.add_argument("--last", "-l", help="try to resume last experiment", default=False)    
    
    args = parser.parse_args()    

    path = f"./config/{args.config}"    

    assert os.path.exists(f"{path}/model.json") and os.path.exists(f"{path}/config.json"), "There is no model config"
    return args, path

def set_logs_folder(date:datetime.datetime, configs:dict):
    logs_folder = f".{os.sep}outputs{os.sep}{date.year}-{date.month}-{date.day}:{date.hour}:{date.minute}:{date.second}"

    for k, v in configs.items():
        if v:
            if not os.path.exists(logs_folder): os.makedirs(logs_folder)
            with open(f"{logs_folder}{os.sep}{k}.json") as file:
                file.write(v)
    return logs_folder


if __name__ == "__main__":
    args, path = set_args()
    configs = json.loads(open(f"{path}/config.json", "r").read())    
    model_configs = CLIPGenerator.load_configs(path)
    model_configs['pretrained'] = configs['pretrained']
    model, img_processor, tokenizer = CLIPGenerator.load_model(model_configs)
    
    optimizer_cfg = configs['optimizer']
    optim_name = OPTIMIZERS[optimizer_cfg.pop('name')]
    linear = torch.nn.Linear(512, 4, True)
    adam = optim_name(linear.parameters(), **optimizer_cfg)
    
    loss_cfg = configs['loss_fn']
    loss_name = LOSSES[loss_cfg.pop('name')]    
    loss = loss_name(**loss_cfg)
    scheduler_cfg = configs['lr_scheduler']
    scheduler_name = LR_SCHEDULERS[scheduler_cfg.pop("name")]
    scheduler = scheduler_name(optimizer=adam, **scheduler_cfg)


    print(model)
    print(configs)
    print(optim_name)    
    print(adam)
    print(loss_name)
    print(loss)
    print(scheduler_name)
    print(scheduler)

    trainer_cfg = {"optimizer" : adam, "loss_fn" : loss, "lr_scheduler" : scheduler, "device" : configs['device']}
    trainer = Trainer(trainer_cfg)        
    print(trainer)
    #all_configs = GENERATOR[config['model']].value[0].load_configs(path)    
    #date = datetime.datetime.now()
    #save = {"model_cfg" : model_cfg, "img_processor" : processor_cfg, "tok_cfg" : tok_cfg, "cfg" : cfg}
    #logs_folder = set_logs_folder(date, save)
    #model, img_processor, tokenizer = CLIPGenerator.load_model(args.pretrained, model_cfg, processor_cfg, tok_cfg)
    #model = model.to("cuda")

    #optim = AdamW(model.parameters(), 1e-5)
    #trainer = CLIPTrainer(optim, CrossEntropyLoss(), None, logs_folder, total_epochs=10)        
