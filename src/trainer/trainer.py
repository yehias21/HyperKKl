class Trainer():
    def __init__(self, model, criterion, metric_ftns, optimizer, cfg,
                 data_loader, valid_data_loader=None, lr_scheduler=None):
        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.cfg = cfg
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler

    def _train_epoch(self, epoch):
        self.model.train()
        for idx, data in enumerate(self.data_loader):
            x_label, z_label, y_label, u_label, t = data
            self.optimizer.zero_grad()
            # data preparation

            z_pred, x_pred = self.model(x_label, z_label, y_label, t)
            loss = self.criterion(x_pred, x_dot, z_pred, z_dot, y_pred, x_label, x_dot, z_label, z_dot, y_label)
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
    def _valid_epoch(self, epoch):
        pass

    def _progress(self, batch_idx):
        pass

if __name__ == '__main__':
    pass