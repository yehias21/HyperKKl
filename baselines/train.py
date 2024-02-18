import hydra
from omegaconf import DictConfig, OmegaConf
from comet_ml import Experiment
from src.data_loader.dataset import load_dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch

from src.loss.pde_loss import Criterion, PDELoss
from src.model.model_generation import init_models
from src.trainer.trainer import Trainer


@hydra.main(config_path="config", config_name="duff", version_base=None)
def main(cfg: DictConfig) -> None:
    # 1. Log the configuration
    OmegaConf.to_yaml(cfg)
    # crd = OmegaConf.load(f"{cfg.credentials}")
    # experiment = Experiment(
    #     api_key=crd.comet_key,
    #     project_name=crd.project_name,
    #     workspace=crd.workspace
    # )
    # experiment.log_parameters(OmegaConf.to_container(cfg, resolve=True))

    # 2. Load train dataset
    # train_loader, val_loader = load_dataset(cfg.data, partition='train')

    # 3. Initialize model
    kkl_model = init_models(cfg.models)

    # 4. Train
    pde_loss = None
    if cfg.trainer.method == 'supervised_pinn':
        pde_loss = PDELoss(cfg.models.diff_eq, kkl_model.forward, cfg.models.A, cfg.models.B)
    loss = Criterion(cfg.trainer.loss, cfg.trainer.method, pde_loss)
    optimizer = torch.optim.Adam(kkl_model.learnable_params, lr=cfg.train.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, threshold=0.0001, verbose=True)
    trainer = Trainer(cfg, kkl_model, loss, optimizer, train_loader, val_loader, scheduler)
    # Todo: Save model, optimizer and scheduler

    # Test and Evaluate

    # Todo:
    # Write test_loader
    # Write evaluate function


if __name__ == "__main__":
    main()
