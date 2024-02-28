import torch, torch.nn as nn
import numpy as np, math
from omegaconf import DictConfig


def rank_mlp(input_, output, ratio):
    # Return the low-rank of sub-matrices given the compression ratio
    # minimum possible parameter
    r1 = int(np.ceil(np.sqrt(output)))
    r2 = int(np.ceil(np.sqrt(input_)))
    r = np.min((r1, r2))
    # maximum possible rank
    # To solve it we need to know the roots of quadratic equation: 2*r*(m+n)=m*n
    r3 = math.floor((output * input_) / (2 * (output + input_)))
    rank = math.ceil((1 - ratio) * r + ratio * r3)
    return rank


def get_rank(layer, param, ratio):
    if 'mlp' in layer:
        rank = rank_mlp(param.size(0), param.size(1), ratio)
        out_size = rank * (param.size(0) + param.size(1))
    else:
        raise NotImplementedError
    return out_size


class LoraDecoder(torch.nn.Module):
    def __init__(self, model_dict, cfg):
        super().__init__()
        self.source_dict = model_dict
        self.decoder = {}
        for layer, param in model_dict.items():
            out_size = get_rank(layer, param, cfg.rank_ratio)
            self.decoder[layer] = torch.nn.Linear(cfg.input_size, out_size)

    def forward(self, x):
        results = {}
        for layer in self.decoder.keys():
            x = self.decoder[layer](x)
            # Todo:
            # how to slice the rank matrix and reshape it to the original shape
        return results


def get_decoder(cfg: DictConfig, model_dict: dict):
    match cfg.method.lower():
        case "lora":
            return LoraDecoder(model_dict, cfg)
        case "full":
            raise NotImplementedError("full decoder not implemented yet")
        case "chunked":
            raise NotImplementedError("Chunked decoder not implemented yet")
        case _:
            raise ValueError(f"Unknown decoder type: {cfg.method}")


class HyperNetwork(torch.nn.Module):

    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module) -> None:
        super(HyperNetwork, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x) -> dict[str, torch.Tensor]:
        embd = self.encoder(x)
        weights = self.decoder(embd)
        return weights
