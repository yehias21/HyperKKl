from typing import Optional
import torch.nn as nn
from hypernetwork import HyperNetwork


class KKLObserverNetwork(nn.Module):
    def __init__(self, forward_mapper, inverse_mapper,
                 hypernetwork: dict[str, Optional[HyperNetwork]]):
        super(KKLObserverNetwork, self).__init__()
        self.forward_mapper = forward_mapper
        self.inverse_mapper = inverse_mapper
        self.normalizer = None
        self.hypernetwork = hypernetwork

    def forward(self, inputs: dict):
        if self.hypernetwork['forward'] is not None:
            try:
                weights = self.hypernetwork['forward'](inputs['hyper_input'])
                z_hat = self.forward_mapper(inputs['x'], weights)
            except Exception as e:
                print(f"Forward mapper hypernetwork failed with error: {e}")
        else:
            z_hat = self.forward_mapper(inputs['x'])

        if self.hypernetwork['inverse'] is not None:
            try:
                weights = self.hypernetwork['inverse'](inputs['hyper_input'])
                x_hat = self.inverse_mapper(inputs['z'], weights)
            except Exception as e:
                print(f"Inverse mapper hypernetwork failed with error: {e}")
        else:
            x_hat = self.inverse_mapper(inputs['z'])

        return z_hat, x_hat
