class Trainer():
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        pass

    def _train_epoch(self, epoch):

        self.model.train()


    def _valid_epoch(self, epoch):
        pass

    def _progress(self, batch_idx):
        pass
