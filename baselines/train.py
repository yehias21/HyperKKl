import hydra
from omegaconf import DictConfig, OmegaConf
import torch, logging
from src.data_loader.dataset import load_dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.utils.normalizer import Normalizer
from src.utils.loss import Criterion, PDELoss
from src.utils.logger import Logger
from src.model.model_generation import init_models
from src.trainer.trainer import Trainer


@hydra.main(config_path="config", config_name="duff", version_base=None)
def main(cfg: DictConfig) -> None:
    # 1. Log the configuration
    OmegaConf.to_yaml(cfg)
    logger = None

    # 2. Load train dataset
    train_loader = load_dataset(cfg.data, partition='train')
    normalizer = Normalizer(train_loader.dataset)

    # 3. Initialize model
    kkl_model = init_models(cfg.models)
    logging.info(f"The  model is {kkl_model}")

    pde_loss = None
    if cfg.trainer.method == 'supervised_pinn':
        pde_loss = PDELoss(train_loader.dataset.system.diff_eq, kkl_model.forward_mapper, train_loader.dataset.observer.A,
                           train_loader.dataset.observer.B, normalizer=normalizer)

    loss = Criterion(cfg.trainer.loss, cfg.trainer.method, pde_loss, normalizer=normalizer)
    kkl_model.set_normalizer(normalizer)
    optimizer = torch.optim.Adam(kkl_model.learnable_params, lr=cfg.trainer.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, threshold=0.0001, verbose=True)
    trainer = Trainer(cfg=cfg, model=kkl_model, criterion=loss, optimizer=optimizer, train_loader=train_loader,
                      lr_scheduler=scheduler, logger=logger, epochs=cfg.trainer.epochs, device= cfg.trainer.device)

    # 4. Train
    trainer.train()

    # 5. Save model
    # Todo: Save model, optimizer and scheduler

    # Test and Evaluate


if __name__ == "__main__":
    main()
