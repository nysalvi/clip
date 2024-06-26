class Trainer:
    def __init__(self, optimizer, num_epochs, scheduler):
        self.optimizer = optimizer
        self.num_epochs = num_epochs

    def train_epoch(self, model, trainLoader):
        model.train()
        #for X, y in trainLoader:
            