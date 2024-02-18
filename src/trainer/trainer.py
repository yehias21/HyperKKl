class Trainer:
    """
    This trainer suits the following paradigms:
    - Normal training
    - Curriculum learning
    - Hypernetwork training
    """
    def __init__(self, model, criterion, optimizer, cfg,
                 data_loader, valid_data_loader=None, lr_scheduler=None, pde_loss=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.cfg = cfg
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.pde_loss = pde_loss
    def _train_epoch(self, epoch):
        self.model.train()
        for idx, data in enumerate(self.data_loader):
            total_loss = 0
            x_label, z_label, y_label, u_label, t = data
            self.optimizer.zero_grad()
            z_pred, x_pred = self.model(x_label, z_label, y_label, t)
            total_loss += self.criterion(x_pred, x_label, z_pred, z_label)
            total_loss+= self.pde_loss( z_pred, x_label, y_label)
            total_loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

    def _valid_epoch(self, epoch):
        pass


if __name__ == '__main__':
    pass