from typing import Optional
import torch.nn as nn
from src.model.hypernetwork import HyperNetwork


class KKLObserverNetwork(nn.Module):
    def __init__(self, forward_mapper, inverse_mapper,
                 forward_hypernetwork: Optional[HyperNetwork],
                 inverse_hypernetwork: Optional[HyperNetwork]):
        super(KKLObserverNetwork, self).__init__()
        self.forward_mapper = forward_mapper
        self.inverse_mapper = inverse_mapper
        self.forward_hypernetwork = forward_hypernetwork
        self.inverse_hypernetwork = inverse_hypernetwork
        self.normalizer = None

    @property
    def learnable_params(self):
        # Todo: Need to be reviewed
        # Parameters from forward_mapper and inverse_mapper
        params = list(self.forward_mapper.parameters()) + list(self.inverse_mapper.parameters())

        # Parameters from forward hypernetwork (if it exists)
        if self.forward_hypernetwork is not None:
            params.extend(list(self.forward_hypernetwork.parameters()))

        # Parameters from inverse hypernetwork (if it exists)
        if self.inverse_hypernetwork is not None:
            params.extend(list(self.inverse_hypernetwork.parameters()))

        return params

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def forward(self, inputs: dict):
        if self.forward_hypernetwork is not None:
            try:
                weights = self.forward_hypernetwork(inputs['exo_input'])
                z_hat = self.forward_mapper(inputs['x_states'], weights)
            except Exception as e:
                print(f"Forward mapper hypernetwork failed with error: {e}")
        else:
            z_hat = self.forward_mapper(inputs['x_states'])

        if self.inverse_hypernetwork is not None:
            try:
                weights = self.inverse_hypernetwork(inputs['exo_input'])
                x_hat = self.inverse_mapper(inputs['z_states'], weights)
            except Exception as e:
                print(f"Inverse mapper hypernetwork failed with error: {e}")
        else:
            x_hat = self.inverse_mapper(inputs['z_states'])

        return z_hat, x_hat
