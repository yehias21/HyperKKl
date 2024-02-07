# imports for model generation
import copy
from omegaconf import DictConfig, OmegaConf
from mapper import KKLObserverNetwork
import hydra
from models import get_model
from hypernetwork import HyperNetwork, get_decoder


@hydra.main(config_path="/media/yehias21/DATA/projects/KKL observer/hyperkkl/baselines/config/", config_name="duff",
            version_base=None)
def init_models(cfg: DictConfig) -> KKLObserverNetwork:
    # Configuration initialization
    OmegaConf.to_yaml(cfg)
    cfg = cfg.models
    hypernetwork = {'forward': None, 'inverse': None}

    # Initialize the forward and inverse mapper
    forward_mapper = get_model(cfg.forward_mapper)
    inverse_mapper = get_model(cfg.inverse_mapper)

    # Initialize the hypernetwork
    if cfg.hypernetwork is not None:
        encoder = get_model(cfg.hypernetwork.encoder)

        update_method_backprop = cfg.forward_mapper.update_method == "backprop"
        if not update_method_backprop:
            fm_dict = forward_mapper.state_dict()
            if not cfg.hypernetwork.shared:
                # Clone encoder module
                fm_enc = copy.deepcopy(encoder)
            else:  # Shared hypernetwork
                fm_enc = encoder
            decoder = get_decoder(cfg.hypernetwork.decoder, fm_dict, input_size=cfg.hypernetwork.encoder.output_size)
            hypernetwork['forward'] = HyperNetwork(fm_enc, decoder)

        update_method_backprop_inverse = cfg.inverse_mapper.update_method == "backprop"
        if not update_method_backprop_inverse:
            im_dict = inverse_mapper.state_dict()
            if not cfg.hypernetwork.shared:
                # Clone encoder module
                im_enc = copy.deepcopy(encoder)
            else:  # Shared hypernetwork
                im_enc = encoder
            decoder = get_decoder(cfg.hypernetwork.decoder, im_dict, input_size=cfg.hypernetwork.encoder.output_size)
            hypernetwork['inverse'] = HyperNetwork(im_enc, decoder)

    kkl_observer_network = KKLObserverNetwork(forward_mapper, inverse_mapper, hypernetwork)
    return kkl_observer_network

if __name__ == '__main__':
    init_models()