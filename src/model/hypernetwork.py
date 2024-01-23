import torch
from typing import TYPE_CHECKING, Optional
import numpy as np
class encoder(torch.nn.Module):
    '''
        TODO
        Encoder for the hypernetwork
    '''
    def __init__(self, in_dim, neurons: list, activation = 'relu'):
        super(encoder, self).__init__()
        self.layers = torch.nn.ModuleList() 
        # TODO: add the layers to the module list
    def forward(self, x):
        pass


class decoder(torch.nn.Module):
    '''
        TODO
        Decoder for the hypernetwork
    '''
    def __init__(self, in_dim, neurons: list, activation = 'relu'):
        super(decoder, self).__init__()
        self.layers = torch.nn.ModuleList() 

    def forward(self, x):
        pass


class HyperNetwork(torch.nn.Module):
    '''
        TODO
        Vanilla HyperNetwork, explain the structure of the network and add params if needed
    '''
    def __init__(self, exoin_dim, inner_dim: list[int], params,activation = 'relu'):
        super(HyperNetwork, self).__init__()
        self.encoder = torch.nn.ModuleList() 
        for i in range(len(inner_dim)):
            if i == 0:
                self.layers.append(torch.nn.Linear(exoin_dim, inner_dim[i]))
                self.layers.append(torch.nn.ReLU())
            else:
                self.layers.append(torch.nn.Linear(inner_dim[i-1], inner_dim[i]))
                self.layers.append(torch.nn.ReLU())

        self.decoder = {}
        #  params are the parameters of the main network, loop on it and create a feedforward layer that regress the weight of each layer of the main network
        for name, size in params.items():
            out_size = np.prod(size)
            self.decoder[name]=torch.nn.Linear(inner_dim[-1], out_size)
    def forward(self, x):
        res ={}
        x = self.encoder(x)
        for zz in self.decoder.items():
            pass

