import torch
import os

class Trainer:
    def __init__(self, configs):
        self.optimizer = configs['optimizer']
        self.loss_fn = configs['loss_fn']
        self.scheduler = configs['lr_scheduler']        
        self.device = configs['device']

    def train(self, model, trainLoader):
        y_true, y_pred, y_score, losses = [], [], [], []        
        model.train()
        for X, y in trainLoader:            
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()            
            outputs = self.model(X)

            loss_all = self.loss_fn(outputs, y)            
            loss = torch.mean(loss_all)
            loss.backward()

            self.optimizer.step()            
            
            y_true+= y; y_score+= outputs; losses+= loss_all
            y_pred+= (outputs > 0.5) * 1.; 
        
        return {'loss' : losses, 'y_hat' : y_pred, 'scores' : y_score, 'y' : y_true}
    
    def validate(self, model, validateLoader):
        y_true, y_pred, y_score, losses = [], [], [], []
        losses = []        
        model.eval()
        with torch.no_grad():
            for X, y in validateLoader:
                X, y = X.to(self.device), y.to(self.device)

                outputs = model(X)
                loss_all = self.loss_fn(outputs, y)            

                y_true+= y; y_score+= outputs; losses+= loss_all
                y_pred+= (outputs > 0.5) * 1.
                
        return {'loss' : losses, 'y_hat' : y_pred, 'scores' : y_score, 'y' : y_true}                
    
    def test(self, model, testLoader):
        y_true, y_pred, y_score, losses = [], [], [], []
        losses = []        
        model.eval()
        with torch.no_grad():
            for X, y in testLoader:
                X, y = X.to(self.device), y.to(self.device)

                outputs = model(X)
                loss_all = self.loss_fn(outputs, y)            

                y_true+= y; y_score+= outputs; losses+= loss_all
                y_pred+= (outputs > 0.5) * 1.;  
               
        return {'loss' : losses, 'y_hat' : y_pred, 'scores' : y_score, 'y' : y_true}                
                