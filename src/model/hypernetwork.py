import torch
import numpy as np, math
from omegaconf import DictConfig


def rank_calc(input_, output, ratio):
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


def get_decoder(cfg: DictConfig, model_dict: dict, input_size: int):
    match cfg.method.lower():
        case "lora":
            decoder = {}
            for layer, param in model_dict.items():
                rank = rank_calc(param.size(0), param.size(1), cfg.rank_ratio)
                out_size = rank * (param.size(0) + param.size(1))
                decoder[layer] = torch.nn.Linear(input_size, out_size)
        case "full":
            ...
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
