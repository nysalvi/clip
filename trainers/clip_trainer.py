import torch
import os

class CLIPTrainer:
    def __init__(self, **kargs):
        self.optimizer = kargs['optimizer']
        self.loss_fn = kargs['loss_fn']
        self.scheduler = kargs['lr_scheduler']        
        self.device = kargs['device']
        self.processor = kargs['image_processor']
        self.tokenizer = kargs['tokenizer']
        
    def train(self, model, trainLoader):
        y_true, y_pred, y_score, losses = [], [], [], []        
        model.train()
        for X, y, _ in trainLoader:            
            X = self.processor(X)
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()            
            outputs = self.model(X)

            loss = self.loss_fn(outputs, y)                        
            loss.backward()

            self.optimizer.step()            
            
            y_true+= y; y_score+= outputs; losses+= loss
            y_pred+= (outputs > 0.5) * 1.
        
        return {'loss' : losses, 'y_hat' : y_pred, 'scores' : y_score, 'y' : y_true}
    
    def validate(self, model, validateLoader):
        y_true, y_pred, y_score, losses = [], [], [], []
        losses = []        
        model.eval()
        with torch.no_grad():
            for X, y, _ in validateLoader:
                X = self.processor(X)
                X, y = X.to(self.device), y.to(self.device)

                outputs = model(X)
                loss = self.loss_fn(outputs, y)            

                y_true+= y; y_score+= outputs; losses+= loss
                y_pred+= (outputs > 0.5) * 1.
                
        return {'loss' : losses, 'y_hat' : y_pred, 'scores' : y_score, 'y' : y_true}                
    
    def test(self, model, testLoader):
        y_true, y_pred, y_score, losses = [], [], [], []
        losses = []        
        model.eval()
        with torch.no_grad():
            for X, y, _ in testLoader:
                X = self.processor(X)
                X, y = X.to(self.device), y.to(self.device)

                outputs = model(X)
                loss = self.loss_fn(outputs, y)            

                y_true+= y; y_score+= outputs; losses+= loss
                y_pred+= (outputs > 0.5) * 1.               
        return {'loss' : losses, 'y_hat' : y_pred, 'scores' : y_score, 'y' : y_true}                
                