from typing import Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class SimTime:
    """
    Time interval for the simulation
    t0: Initial time
    tn: Final time
    eps: time resolution
    """
    t0: float
    tn: float
    eps: float

@dataclass
class SinParam:
    """
    Parameters for the sin function
    A: Amplitude
    w: Frequency
    phi: Phase
    """
    A: float
    w: float

@dataclass
class SigParam:
    """
    list of sin functions
    """
    sig_name: str
    sig_params: list[SinParam]

@dataclass
class ICParam:
    """
    Parameters for the initial conditions
    sampler: sampler name
    sample_space: sampler_space limit
    seed: random seed used
    samples: number of samples
    """
    sampler: str
    seed: int
    samples: int 
    sample_space: Optional[np.ndarray] = np.empty((5,4))
@dataclass
class SysDim:
    """
    System dimensions
    """
    x_dim: int
    z_dim: int
    y_dim: int

@dataclass
class NoiseParam:
    """
    Parameters for the noise
    """
    sys_noise: Optional[dict]
    out_noise: Optional[dict]
    inp_noise: Optional[dict]


