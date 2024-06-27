from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, f1_score, fbeta_score, recall_score, root_mean_squared error, root_mean_squared_log_loss
import os

class Trainer:
    def __init__(self, optimizer, scheduler:None, total_epochs, current_epoch=0):
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.scheduler = scheduler
        self.writer = SummaryWriter(log_dir="outputs")
    def train_epoch(self, model, trainLoader):
        model.train()
        #for X, y in trainLoader:
            