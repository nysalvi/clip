from sklearn.metrics import confusion_matrix, mean_squared_error, mean_squared_log_error
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, max_error, log_loss, zero_one_loss
from torch.utils.tensorboard.writer import SummaryWriter, FileWriter
import pandas as pd
import torch

#LR_SCHEDULERS = {
#    "MultiplicativeLR" : torch.optim.lr_scheduler.MultiplicativeLR,
#    "LambdaLR" : torch.optim.lr_scheduler.LambdaLR,
#    "StepLR" : torch.optim.lr_scheduler.StepLR,
#    "CosineAnnealingLR" : torch.optim.lr_scheduler.CosineAnnealingLR,
#    "SequentialLR" : torch.optim.lr_scheduler.SequentialLR
#}
#
#OPTIMIZERS = {
#    "Adam" : torch.optim.Adam,
#    "AdamW" : torch.optim.AdamW,
#    "SGD" : torch.optim.SGD,
#    "SparseAdam" : torch.optim.SparseAdam
#}
#
#LOSSES = {
#    "MSELoss" : torch.nn.MSELoss,
#    "L2" : torch.nn.MSELoss,
#    "CrossEntropyLoss" : torch.nn.CrossEntropyLoss,
#    "BCEWithLogitsLoss" : torch.nn.BCEWithLogitsLoss,
#    "BCELoss" : torch.nn.BCELoss,
#    "CosineSimilarity" : torch.nn.CosineSimilarity,
#    "PairwiseDistance" : torch.nn.PairwiseDistance
#}

METRICS = [
    mean_squared_log_error,
    mean_squared_error,
    log_loss,
    f1_score,
    accuracy_score,
    max_error,
    zero_one_loss,
    precision_score,
    recall_score    
]

class Writer():
    def __init__(self, model_name, path):        
        self.name = model_name
        self.writer = SummaryWriter(path)
    def load_metrics(self, dataset, y_true, y_pred):                   
        values = {}
        for metric in METRICS:
            name = metric.__name__
            print(name)
            value = metric(y_true, y_pred)
            print(value)
            values.update({name : value})
            self.writer.add_scalars(dataset, {
                f"{self.model_name}/{name}" : value
            })        
        return values
    def write(self, tags:list, i):
        tag = ""
        for t in tags:
            tag+= f"/{t}"            
        self.writer.add_scalar(f"{self.name}{tag}", i)
        
    def save_model(model, optimizer, lr_scheduler, i):
        pd.DataFrame()
        