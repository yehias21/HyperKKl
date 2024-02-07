import hydra
from omegaconf import DictConfig, OmegaConf
from comet_ml import Experiment
from src.data_loader.dataset import load_dataset


@hydra.main(config_path="config", config_name="duff",  version_base=None)
def main(cfg: DictConfig) -> None:

    # Import configuration and initialize logging system
    OmegaConf.to_yaml(cfg)
    crd = OmegaConf.load(f"{cfg.credentials}")
    experiment = Experiment(
    api_key=crd.comet_key,
    project_name=crd.project_name,
    workspace=crd.workspace
    )

    # Load dataset
    trainloader = load_dataset(cfg.data, partition='train')

    # Initialize model
    

    # Train


    # Test and Evaluate
    testloader = load_dataset(cfg=cfg.data, partition='test')

    # Save model and plot


if __name__  == "__main__":
    main()