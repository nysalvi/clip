
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_squared_log_error
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, max_error, log_loss, zero_one_loss
from torch.utils.tensorboard.writer import SummaryWriter, FileWriter
#from torch.utils.tensorboard.summary import 
import pandas as pd
import torch
#TF_ENABLE_ONEDNN_OPTS=0
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
    recall_score,
    confusion_matrix
]

class Writer():
    def __init__(self, date, path):
        self.writer = SummaryWriter(path)
    def sklearn(y_true, y_pred, df=None):
        if not df:
            df = pd.DataFrame([])
        for metric in METRICS:
            df[metric.__name__] = metric(y_true, y_pred)
        return df
    
