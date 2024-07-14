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


@hydra.main(config_path="config", config_name="lorenz", version_base=None)
def main(cfg: DictConfig) -> None:
    # 1. Log the configuration
    OmegaConf.to_yaml(cfg)
    logger = None

    # 2. Load train dataset
    train_loader = load_dataset(cfg.data, partition='train')


if __name__ == "__main__":
    main()