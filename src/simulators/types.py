from typing import Optional, Any, Dict
from dataclasses import dataclass
import numpy as np


@dataclass
class SysParam:
    """
    System parameters
    """
    C: list[float]
    ObservableIndex: list[int]
    system_coeff: Optional[Dict[str, Any]] = None


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


## Input Signal dataclasses

@dataclass
class SigParam:

    signal_type: str
    signal_data: Dict
