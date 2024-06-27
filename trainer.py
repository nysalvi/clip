from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, f1_score, fbeta_score, recall_score, root_mean_squared_error, root_mean_squared_log_loss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
import os

class CLIPTrainer:
    def __init__(self, optimizer, loss, scheduler:None, logs_folder, total_epochs, resume_training=False):
        self.optimizer = optimizer
        self.loss = loss
        self.total_epochs = total_epochs
        self.scheduler = scheduler
        self.writer = SummaryWriter(log_dir=logs_folder)
    def train_epoch(self, model, trainLoader):
        epoch_loss = 0
        model.train()
        for X, y in trainLoader:
            X, y = X.to(), y.to()
            self.optimizer.zero_grad()            
            outputs = self.model(X)
            loss = self.loss(outputs, loss)            
            loss.backward()
            self.optimizer.step()
            epoch_loss+= loss
        return epoch_loss
    
    def validate(self, model, validateLoader):
        with torch.no_grad():
            for X, y in validateLoader:
                X, y = X.to("cuda"), y.to("cuda")

    def train(self, model, trainLoader):
        for i in tqdm(range(self.total_epochs)):
            losses = self.train_epoch(model, trainLoader)
            self.writer(i)